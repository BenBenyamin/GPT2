from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from tqdm import tqdm

# Load dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Split into train and val
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# # Process val dataset
# tokens = []
# attention_masks = []
# for example in tqdm(val_dataset, desc="Tokenizing validation dataset"):
#     encoding = tokenizer(
#         example['text'],
#         return_tensors='pt',
#         truncation=True,
#         padding='max_length',
#         max_length=1024,
#     )
#     tokens.append(encoding['input_ids'])
#     attention_masks.append(encoding['attention_mask'])

# tokens = torch.cat(tokens, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# torch.save({'tokens': tokens, 'attention_masks': attention_masks}, "val_tokenized.pt")

# Now tokenize the train dataset and split it into 4 files on the fly
num_splits = 4
split_size = len(train_dataset) // num_splits

tokens = []
attention_masks = []

for idx, example in enumerate(tqdm(train_dataset, desc="Tokenizing train dataset")):

    encoding = tokenizer(
        example['text'],
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=1024,
    )
    tokens.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

    # Save upon reaching split_size items
    if (idx + 1) % split_size == 0:
        split_id = idx // split_size
        input_ids = torch.cat(tokens, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)
        torch.save(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            f'train_tokenized_{split_id}.pt'
        )
        tokens = []
        attention_masks = []

# Handle remaining examples (if any)
if tokens:
    split_id = num_splits - 1
    input_ids = torch.cat(tokens, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    torch.save(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        f'train_tokenized_{split_id}.pt'
    )
