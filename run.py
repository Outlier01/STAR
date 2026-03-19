import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import argparse
import random
import torch
from Trainer import Trainer
import pandas as pd
from peft import LoraConfig

def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the train")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--victim_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    # parser.add_argument("--victim_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./output/RL")
    parser.add_argument("--checkpoint_path", type=str, default="./output/RL/checkpoint")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_step", type=int, default=300)
    parser.add_argument("--rollouts_per_step", type=int, default=64)
    parser.add_argument("--num_generation", type=int, default=16)
    parser.add_argument("--num_select", type=int, default=16)
    parser.add_argument("--kl_weight", type=float, default=1e-2)
    parser.add_argument("--low_clip_eps", type=float, default=0.2)
    parser.add_argument("--high_clip_eps", type=float, default=0.28)
    parser.add_argument("--L_max", type=int, default=3072)
    parser.add_argument("--L_cache", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--gradient_updates", type=int, default=1)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    init_rng(args.seed)
    devices = [torch.device(f'cuda:{i}') for i in range(8)]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_instructions = pd.read_csv(os.path.join(args.data_path, 'DAN_train.csv'))['0'].tolist()
    val_instructions = pd.read_csv(os.path.join(args.data_path, 'DAN_val.csv'))['0'].tolist()
    strategies = pd.read_csv(os.path.join(args.data_path, 'strategies_500.csv'))['0'].tolist()
    trainer = Trainer(base_model_name=args.base_model,
                      lora_cfg=lora_config,
                      train_instructions=train_instructions,
                      val_instructions=val_instructions,
                      strategies=strategies,
                      victim_model_name=args.victim_model,
                      output_path=args.output_path,
                      checkpoint_path=args.checkpoint_path,
                      devices=devices,
                      lr=args.lr,
                      num_step=args.num_step,
                      rollouts_per_step=args.rollouts_per_step,
                      num_generation=args.num_generation,
                      num_select=args.num_select,
                      kl_weight=args.kl_weight,
                      low_clip_eps=args.low_clip_eps,
                      high_clip_eps=args.high_clip_eps,
                      L_max=args.L_max,
                      L_cache=args.L_cache,
                      train_batch_size=args.train_batch_size,
                      accumulation_steps=args.accumulation_steps,
                      gradient_updates=args.gradient_updates,
                      max_norm=args.max_norm,
                      resume=args.resume,
                      resume_step=args.resume_step)
    trainer.train()