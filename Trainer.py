import os
import random
import torch
from peft import get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from loss import (
    group_advantages,
    sequences_log_probs,
    GRPOLoss,
)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from prompt_template import apply_apply_template
from llm import LocalVLLM
from Scorer import LocalScorer
import logging
import pandas as pd
import shutil

class Trainer:
    def __init__(self,
                 base_model_name,
                 lora_cfg,
                 train_instructions,
                 val_instructions,
                 strategies,
                 victim_model_name,
                 output_path,
                 checkpoint_path,
                 devices,
                 lr=1e-5,
                 num_step=300,
                 rollouts_per_step=64,
                 num_generation=16,
                 num_select=16,
                 kl_weight=1e-2,
                 low_clip_eps=0.2,
                 high_clip_eps=0.28,
                 L_max=2048,
                 L_cache=512,
                 train_batch_size=2,
                 accumulation_steps=32,
                 gradient_updates=1,
                 max_norm=1.0,
                 resume=False,
                 resume_step=0):
        self.cpu_device = torch.device('cpu')
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.training_info_path = os.path.join(self.output_path, 'training_info')
        os.makedirs(self.training_info_path, exist_ok=True)

        self.resume = resume
        self.resume_step = resume_step
        self.step = resume_step

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.apply_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=devices[0],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if self.resume:
            self.apply_model = PeftModel.from_pretrained(
                self.apply_model,
                model_id=os.path.join(self.checkpoint_path, f"step_{self.step}"),
                is_trainable=True,
                low_cpu_mem_usage=True,
            )
        else:
            self.apply_model = get_peft_model(self.apply_model, lora_cfg)
        self.apply_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.base_vllm = LLM(model=base_model_name, dtype='bfloat16', gpu_memory_utilization=0.7, max_model_len=4096, enable_lora=True, device=devices[0])

        self.kl_weight = kl_weight
        self.low_clip_eps, self.high_clip_eps = low_clip_eps, high_clip_eps
        self.L_max, self.L_cache = L_max, L_cache
        self.num_step = num_step
        self.train_instructions = train_instructions
        self.val_instructions = val_instructions
        self.strategies = strategies
        self.rollouts_per_step = rollouts_per_step
        self.num_generation = num_generation
        self.num_select = num_select
        self.train_batch_size = train_batch_size
        self.accumulation_steps = accumulation_steps
        self.gradient_updates = gradient_updates
        self.max_norm = max_norm

        self.optimizer = AdamW(self.apply_model.parameters(), lr=lr)
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=160,
            num_training_steps=self.num_step * 16,
        )
        if self.resume:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_path, f"step_{self.step}/optimizer.pt"), map_location=self.apply_model.device))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_path, f"step_{self.step}/scheduler.pt"), map_location=self.apply_model.device))
            self.get_data(self.train_instructions, (resume_step + 1) * rollouts_per_step % len(self.train_instructions))

        self.victim_model = LocalVLLM(victim_model_name, gpu_memory_utilization=0.3, device=devices[1])
        self.scorer = LocalScorer(model=self.base_vllm, tokenizer=self.tokenizer)
        
        # Initialize lora file for vLLM generation
        if self.resume:
            name, id, path = f"step_{self.step}", self.step + 1, os.path.join(self.checkpoint_path, f"step_{self.step}")
            self.cur_lora_request = LoRARequest(name, id, path)
        else:
            self.cur_lora_request, self.step = None, 0
            self.save_checkpoint(self.step)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger('training_log')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(output_path, "training_log.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def train(self):
        if not self.resume:
            self.eval()
        for step in range(self.resume_step + 1, self.num_step):
            self.step += 1
            self.train_apply_model()
            self.eval()
    
    def train_apply_model(self):
        print(f"------training step {self.step}------apply model training stage------")
        replay_buffer = ReplayBuffer()
        objective = GRPOLoss(low_clip_eps=self.low_clip_eps, high_clip_eps=self.high_clip_eps, kl_weight=self.kl_weight)

        behaviors = self.get_data(self.train_instructions, self.rollouts_per_step)
        strategies = random.sample(self.strategies, self.rollouts_per_step)
        replay_buffer.clear()
        
        all_behaviors, all_strategies = [], []
        for behavior, strategy in zip(behaviors, strategies):
            all_behaviors.extend([behavior] * self.num_generation)
            all_strategies.extend([strategy] * self.num_generation)
        
        completions, length = self.generate_prompt(all_strategies, all_behaviors, filter=False, require_length=True)

        all_completions, all_length, all_sequence_ids, all_action_mask = [], [], [], []
        for i in range(len(behaviors)):
            start_i = i * self.num_generation
            end_i = start_i + self.num_generation
            sequence_ids, action_mask = self.rollout(strategies[i], behaviors[i], completions[start_i: end_i])
            all_sequence_ids.append(sequence_ids)
            all_completions.append(completions[start_i: end_i])
            all_length.append(length[start_i: end_i])
            all_action_mask.append(action_mask)
        
        all_rewards = self.reward_func(behaviors, all_completions, all_length)

        self.apply_model.eval()
        for i in range(len(all_rewards)):
            experience = self.get_experience(self.apply_model, all_rewards[i], all_sequence_ids[i], all_action_mask[i])
            replay_buffer.append(experience.to(self.cpu_device))
        torch.cuda.empty_cache()

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=join_experience_batch,
        )

        self.apply_model.train()
        for gradient_update in range(self.gradient_updates):
            for i, exp in enumerate(experience_sampler):
                exp: Experience
                exp = exp.to(self.apply_model.device)

                log_probs = sequences_log_probs(
                    self.apply_model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask, cbs=2
                )
                if self.kl_weight > 0.0:
                    loss, kl = objective(log_probs=log_probs, experience=exp)
                else:
                    loss = objective(log_probs=log_probs, experience=exp)
                
                loss = loss / self.accumulation_steps

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={exp.advantages}")
                    continue

                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    grad_norm = clip_grad_norm_(self.apply_model.parameters(), max_norm=self.max_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        self.save_checkpoint(self.step)
    
    def get_data(self, data, num):
        if num == 0:
            return None
        sub_data = data[:num]
        data.extend(sub_data)
        del data[:num]
        return sub_data
    
    def generate_prompt(self, strategies, instructions, filter=True, require_length=False):
        prompts_token_ids = []
        for strategy, instruction in zip(strategies, instructions):
            prompt = apply_apply_template(self.tokenizer, strategy, instruction)
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
            input_ls = input_ids[0].tolist()
            if len(input_ls) > 4096:
                input_ls = input_ls[:4096]
            prompt_token_ids = TokensPrompt(prompt_token_ids=input_ls)
            prompts_token_ids.append(prompt_token_ids)
        
        if filter:
            sampling_params = SamplingParams(temperature=0.6, top_k=20, top_p=0.95, max_tokens=4096)
        else:
            sampling_params = SamplingParams(temperature=1.0, max_tokens=4096)
        outputs = self.base_vllm.generate(prompts_token_ids, sampling_params, lora_request=self.cur_lora_request)
        completions = [output.outputs[0].text for output in outputs]
        if require_length:
            length = [len(output.outputs[0].token_ids) for output in outputs]

        if not filter:
            if require_length:
                return completions, length
            return completions
        
        prompts = []
        for completion in completions:
            pos = completion.find('</think>') + len('</think>')
            completion_without_think = completion[pos: ]
            pos1 = completion_without_think.find('[START OF PROMPT]')
            pos2 = completion_without_think.find('[END OF PROMPT]')
            if pos1 != -1 and pos2 != -1:
                pos1 += len('[START OF PROMPT]')
                prompts.append(completion_without_think[pos1: pos2].strip())
            else:
                prompts.append("")
        if require_length:
            return prompts, length
        return prompts

    def rollout(self, strategy, instruction, competitions):
        prompt = apply_apply_template(self.tokenizer, strategy, instruction)
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
        sequence = []
        for competition in competitions:
            sequence.append(prompt + competition)
        
        sequence_ids = self.tokenizer(sequence, add_special_tokens=False, return_tensors='pt', padding=True, padding_side='right')['input_ids']

        pad_token_id = self.tokenizer.eos_token_id
        action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        action_mask[:,input_ids.shape[1]:] = True
        action_mask[sequence_ids == pad_token_id] = False
        action_mask = action_mask[:, 1:]

        return sequence_ids, action_mask
    
    def reward_func(self, instructions, all_completions, all_length):
        valid_instructions, valid_completions, valid_length, valid_idx = [], [], [], []
        for i, completions in enumerate(all_completions):
            for j, completion in enumerate(completions):
                pos = completion.find('</think>') + len('</think>')
                completion_without_think = completion[pos: ]
                pos1 = completion_without_think.find('[START OF PROMPT]')
                pos2 = completion_without_think.find('[END OF PROMPT]')
                if pos1 != -1 and pos2 != -1:
                    pos1 += len('[START OF PROMPT]')
                    valid_instructions.append(instructions[i])
                    valid_completions.append(completion_without_think[pos1: pos2].strip())
                    valid_length.append(all_length[i][j])
                    valid_idx.append((i, j))
        
        responses = self.victim_model.generate(valid_completions)
        rewards = self.scorer.batch_scoring(valid_instructions, valid_completions, responses)
        for i in range(len(rewards)):
            rewards[i] = max(0.0, rewards[i] + self.length_penalty(valid_length[i]))

        all_rewards = [torch.zeros(self.num_generation, 1, dtype=torch.float) for i in range(len(instructions))]
        for i in range(len(rewards)):
            all_rewards[valid_idx[i][0]][valid_idx[i][1]] = rewards[i]
        return all_rewards

    def length_penalty(self, length):
        if length <= self.L_max - self.L_cache:
            return 0.0
        elif length > self.L_max:
            return -3.0
        else:
            return ((self.L_max - self.L_cache) - length) * 3.0 / self.L_cache

    def get_experience(self, model, rewards, sequence_ids, action_mask):
        select_indices = self.select_samples(rewards)
        rewards, sequence_ids, action_mask = rewards[select_indices], sequence_ids[select_indices], action_mask[select_indices]
        with torch.no_grad():
            pad_token_id = self.tokenizer.eos_token_id
            advantages = group_advantages(rewards)
            sequence_ids = sequence_ids.to(model.device)
            action_mask = action_mask.to(model.device)
            attention_mask = sequence_ids != pad_token_id
            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
                cbs=2
            )
            if self.kl_weight > 0.0:
                with model.disable_adapter():
                    log_probs_ref = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                        cbs=2
                    )
            else:
                log_probs_ref = None

        return Experience(
            sequences=sequence_ids,
            action_log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            returns=rewards,
            advantages=advantages,
            attention_mask=attention_mask,
            action_mask=action_mask,
        )

    def select_samples(self, rewards):
        levels = [[] for i in range(4)]
        for i in range(rewards.shape[0]):
            if rewards[i].item() >= 0.0 and rewards[i].item() < 0.75:
                levels[0].append(i)
            elif rewards[i].item() >= 0.75 and rewards[i].item() < 1.5:
                levels[1].append(i)
            elif rewards[i].item() >= 1.5 and rewards[i].item() < 2.25:
                levels[2].append(i)
            else:
                levels[3].append(i)
        
        if len(levels[0]) + len(levels[1]) >= self.num_select // 2 and len(levels[2]) + len(levels[3]) >= self.num_select // 2:
            low_num, high_num = self.num_select // 2, self.num_select // 2
        elif len(levels[0]) + len(levels[1]) < self.num_select // 2:
            low_num = len(levels[0]) + len(levels[1])
            high_num = self.num_select - low_num
        else:
            high_num = len(levels[2]) + len(levels[3])
            low_num = self.num_select - high_num

        indices = []
        
        if low_num > 0:
            if len(levels[0]) == 0:
                indices.extend(random.sample(levels[1], low_num))
            elif len(levels[1]) == 0:
                indices.extend(random.sample(levels[0], low_num))
            elif len(levels[0]) >= low_num // 2 and len(levels[1]) >= (low_num + 1) // 2:
                indices.extend(random.sample(levels[0], low_num // 2))
                indices.extend(random.sample(levels[1], (low_num + 1) // 2))
            elif len(levels[0]) < low_num // 2:
                indices.extend(levels[0])
                indices.extend(random.sample(levels[1], low_num - len(levels[0])))
            else:
                indices.extend(levels[1])
                indices.extend(random.sample(levels[0], low_num - len(levels[1])))

        if high_num > 0:
            if len(levels[2]) == 0:
                indices.extend(random.sample(levels[3], high_num))
            elif len(levels[3]) == 0:
                indices.extend(random.sample(levels[2], high_num))
            elif len(levels[2]) >= high_num // 2 and len(levels[3]) >= (high_num + 1) // 2:
                indices.extend(random.sample(levels[2], high_num // 2))
                indices.extend(random.sample(levels[3], (high_num + 1) // 2))
            elif len(levels[2]) < high_num // 2:
                indices.extend(levels[2])
                indices.extend(random.sample(levels[3], high_num - len(levels[2])))
            else:
                indices.extend(levels[3])
                indices.extend(random.sample(levels[2], high_num - len(levels[3])))
        
        return indices

    def eval(self):
        print(f"------training step {self.step}------eval stage------")
        instructions = []
        for instruction in self.val_instructions:
            instructions.extend([instruction] * 5)
        strategies = self.strategies[:len(instructions)]
        jbk_prompts, length = self.generate_prompt(strategies, instructions, require_length=True)
        responses = self.victim_model.generate(jbk_prompts)
        rewards = self.scorer.batch_scoring(instructions, jbk_prompts, responses)
        rewards_mean = sum(rewards) / len(rewards)
        length_mean = sum(length) / len(length)
        self.logger.info(f"step {self.step}------mean length: {length_mean:.2f}------mean reward: {rewards_mean:.4f}")
        pd.DataFrame({'reward': rewards, 'instruction': instructions, 'strategy': strategies, 'jbk_prompt': jbk_prompts, 'response': responses}).to_csv(os.path.join(self.training_info_path, f'step_{self.step}.csv'), index=None)


    def save_checkpoint(self, step):
        self.apply_model.save_pretrained(os.path.join(self.checkpoint_path, f"step_{step}/"))
        self.tokenizer.save_pretrained(os.path.join(self.checkpoint_path, f"step_{step}/"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoint_path, f"step_{step}/optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(self.checkpoint_path, f"step_{step}/scheduler.pt"))
        name, id, path = f"step_{step}", step + 1, os.path.join(self.checkpoint_path, f"step_{step}")
        self.cur_lora_request = LoRARequest(name, id, path)


