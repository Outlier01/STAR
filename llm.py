import torch
from vllm import LLM as vllm
from vllm import SamplingParams, TokensPrompt
from transformers import AutoTokenizer
from utils import (
    apply_chat_template
)
from openai import OpenAI
import logging
import concurrent.futures
import time

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.5,
                 system_message=None,
                 device='auto'
                 ):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.model = vllm(self.model_path, dtype='bfloat16', gpu_memory_utilization=gpu_memory_utilization, max_model_len=2048, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.system_message = system_message

    def generate_batch(self, prompts, temperature=0, max_tokens=1024):
        prompts_token_ids = []
        for prompt in prompts:
            full_prompt = apply_chat_template(self.tokenizer, prompt, self.system_message)
            input_ids = self.tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")['input_ids']
            input_ls = input_ids[0].tolist()
            if len(input_ls) > 2048:
                input_ls = input_ls[:2048]
            prompt_token_ids = TokensPrompt(prompt_token_ids=input_ls)
            prompts_token_ids.append(prompt_token_ids)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(prompts_token_ids, sampling_params)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs

class OpenAILLM:
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(api_key = api_key, base_url='')
        self.model_path = model_path

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        # {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                contents = [results.choices[i].message.content for i in range(n)]
                for i in range(len(contents)):
                    if contents[i] == None:
                        contents[i] = " "
                return contents
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_list = []
            for prompt in prompts:
                future = executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                        max_trials, failure_sleep_time)
                future_list.append(future)
                # time.sleep(2)

            for future in future_list:
                results.extend(future.result())
        return results