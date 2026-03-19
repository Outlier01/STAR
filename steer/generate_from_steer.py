from utils import load_model_and_tokenizer, get_strategy_prompt
import torch
import os
import numpy as np
from sklearn.decomposition import PCA
from SVTrainer import BlockOutputWrapper
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda:0')

def check(strategy):
    if len(strategy) < 18 or strategy[:18] != "**Strategy Name:**":
        return False
    if strategy.find("**Underlying Principle:**") != -1 and strategy.find("**Description:**") != -1:
        return True
    return False

all_steer_vectors = []
fns = os.listdir('./new_output_sv')
for fn in fns:
    fpath = os.path.join('./new_output_sv', fn)
    all_steer_vectors.append(torch.stack(torch.load(fpath)).view(-1))
data_matrix = torch.stack(all_steer_vectors).to(torch.float16).numpy()

pca = PCA(n_components=0.99)
pca.fit(data_matrix)

n_components_chosen = pca.n_components_
stdevs = np.sqrt(pca.explained_variance_)

model_name = "Qwen/Qwen3-4B"
model, tokenizer = load_model_and_tokenizer(model_name, device=device)

prompt = get_strategy_prompt(tokenizer)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
generation_kwargs = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "max_new_tokens": 1000,
    "do_sample": True,
}

strategies = []
for i in tqdm(range(500)):
    response = None
    for j in range(10):
        try:
            random_coords = np.random.randn(1, n_components_chosen)
            scaled_coords = random_coords * stdevs
            new_tensors_np = pca.inverse_transform(scaled_coords)
            steer_vectors = 1.0 * torch.from_numpy(new_tensors_np).view(-1, 2560).to(torch.bfloat16).to(device)
            for k in range(12, 24):
                model.model.layers[k] = BlockOutputWrapper(model.model.layers[k], steer_vectors[k - 12].to(device))
            outputs = model.generate(**inputs, **generation_kwargs)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for k in range(12, 24):
                model.model.layers[k] = model.model.layers[k].block
            assert (outputs.shape[1] <= 800 and check(response))
            break
        except AssertionError as error:
            response = None
    if response != None:
        strategies.append(response)

    pd.DataFrame(strategies).to_csv(f'./strategies_steer_trans.csv', index=None)
    
print(f"total generate {len(strategies)} strategies")