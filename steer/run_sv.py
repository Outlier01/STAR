import os
import argparse
import random
import torch
from SVTrainer import SVTrainer
import pandas as pd

def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the train")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--data_path", type=str, default="./strategies")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--add_layer_low", type=int, default=12)
    parser.add_argument("--add_layer_high", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    init_rng(args.seed)
    devices = [torch.device(f'cuda:{i}') for i in range(8)]

    data = []
    for i in range(100):
        data.append(pd.read_csv(os.path.join(args.data_path, f'strategy_{i + 1}.csv'))['0'].tolist())

    trainer = SVTrainer(model_name=args.base_model,
                        data=data,
                        output_path=args.output_path,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        epoch=args.epoch,
                        add_layer_low=args.add_layer_low,
                        add_layer_high=args.add_layer_high,
                        device=devices[6])
    trainer.run()