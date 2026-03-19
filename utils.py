import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def load_model_and_tokenizer(model_name, quantization_config=None, device='cuda'):
    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            low_cpu_mem_usage=True,
            quantization_config = quantization_config,
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def apply_chat_template(tokenizer, text, system_prompt=None, add_generation_prompt=True, enable_thinking=True):
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
    else:
        messages = [{"role": "user", "content": text}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
    return full_prompt