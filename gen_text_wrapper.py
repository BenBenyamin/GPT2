import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from model import GPT2 as CustomGPT2Model  # your own model
import torch
import random
import numpy as np


# Set prompt and generation settings
prompt = "Explain in simple terms how quantum entanglement works. Use metaphors and examples a 10-year-old could understand."
max_length = 150
temperature = 0.7
top_k = 100

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# -------------------------------------
# Load Hugging Face official GPT-2
# -------------------------------------
hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda')
hf_model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
hf_output = hf_model.generate(
    **inputs,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    do_sample=True,
    num_return_sequences=1
)

print("\n Hugging Face GPT-2 Output:\n")
print(tokenizer.decode(hf_output[0], skip_special_tokens=True))

# -------------------------------------
# Wrap custom GPT2 model
# -------------------------------------

# Create a dummy config
custom_config = GPT2Config(
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_positions=1024,
    n_ctx=1024,
    vocab_size=50257
)

class CustomGPT2Wrapper(GPT2LMHeadModel):
    def __init__(self, config, custom_model):
        super().__init__(config)
        self.transformer = custom_model  # just reusing internal variable name

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = self.transformer(input_ids)
        return CausalLMOutput(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

# Instantiate model
custom_model = CustomGPT2Model(
    n_blocks=12,
    seq_len=1024,
    n_embd=768,
    n_head=12,
    vocab_size=50257,
    dropout=0.1
)
custom_model.load_state_dict(torch.load("myGPT2_from_pretrained.pth", weights_only=True))
custom_model.eval().to('cuda')

wrapped_model = CustomGPT2Wrapper(custom_config, custom_model).to('cuda')

# Run generation
custom_output = wrapped_model.generate(
    **inputs,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    do_sample=True,
    num_return_sequences=1
)

print("\n Custom GPT2 Output:\n")
print(tokenizer.decode(custom_output[0], skip_special_tokens=True))
