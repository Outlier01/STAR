import torch
from utils import load_model_and_tokenizer, get_strategy_prompt
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
import os

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, add_activations):
        super().__init__()
        self.block = block
        self.add_activations = torch.nn.parameter.Parameter(add_activations)

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        modified_hidden_states = output[0] + self.add_activations
        return (modified_hidden_states,) + output[1:] 

class SVTrainer:
    def __init__(self,
                 model_name,
                 data,
                 output_path,
                 lr=1e-3,
                 batch_size=4,
                 epoch=50,
                 add_layer_low=12,
                 add_layer_high=24,
                 device='auto'):
        self.device = device
        self.model, self.tokenizer = load_model_and_tokenizer(model_name, device=device)
        self.data = data
        self.output_path = output_path

        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.add_layer_low = add_layer_low
        self.add_layer_high = add_layer_high

    def run(self):
        for i, strategies in enumerate(self.data):
            for j in range(self.add_layer_low, self.add_layer_high):
                self.model.model.layers[j] = BlockOutputWrapper(self.model.model.layers[j], torch.zeros(2560, dtype=torch.bfloat16, device=self.device))
            for param in self.model.parameters():
                param.requires_grad = False
            for j in range(self.add_layer_low, self.add_layer_high):
                self.model.model.layers[j].add_activations.requires_grad = True

            dataloader = DataLoader(strategies, batch_size=self.batch_size, shuffle=True)
            optimizer = AdamW([self.model.model.layers[j].add_activations for j in range(self.add_layer_low, self.add_layer_high)], lr=self.lr)
            training_step = ((len(dataloader) + self.batch_size - 1) // self.batch_size) * self.epoch
            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=training_step * 0.1,
                num_training_steps=training_step,
            )
            prompt = get_strategy_prompt(self.tokenizer)
            prompt_token_length = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)

            for epoch in range(self.epoch):
                all_loss = []
                for batch_data in dataloader:
                    full_texts = []
                    for single_data in batch_data:
                        answer = single_data + self.tokenizer.eos_token
                        full_texts.append(prompt + answer)
                    inputs = self.tokenizer(full_texts, add_special_tokens=False, return_tensors='pt', padding=True, padding_side='right').to(self.model.device)
                    labels = inputs['input_ids'].clone()
                    labels[:, :prompt_token_length] = -100

                    optimizer.zero_grad() 
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    all_loss.append(loss.item())
                print(f"epoch: {epoch + 1}, mean loss: {sum(all_loss) / len(all_loss)}")
            
            steer_vectors = []
            for j in range(self.add_layer_low, self.add_layer_high):
                steer_vectors.append(self.model.model.layers[j].add_activations.clone().detach().to('cpu'))
            torch.save(steer_vectors, os.path.join(self.output_path, f'steer_vector_{i + 1}.pt'))

            for j in range(self.add_layer_low, self.add_layer_high):
                self.model.model.layers[j] = self.model.model.layers[j].block