from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from tqdm import tqdm

"""
This script loads the FineWeb-Edu dataset (sample-10BT), splits it into training and validation sets,
then tokenizes the validation set using the GPT-2 tokenizer and saves it as a .pt file.
"""

# Load dataset from Hugging Face
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token as eos_token for GPT2 compatibility

# Create train/val split
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Tokenize validation dataset and save
tokens = []
attention_masks = []
for example in tqdm(val_dataset, desc="Tokenizing validation dataset"):
    encoding = tokenizer(
        example['text'],
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=1024,
    )
    tokens.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

# Combine into single tensors and save
tokens = torch.cat(tokens, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
torch.save({'input_ids': tokens, 'attention_mask': attention_masks}, "/data/users/ofh8750/val_tokenized.pt")
