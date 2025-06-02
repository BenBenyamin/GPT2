import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

class FineWebEdu(Dataset):
    """
    Custom PyTorch Dataset for loading tokenized training and validation data
    from the FineWebEdu dataset, supporting distributed training by chunking.

    Args:
        split (str): Either 'train' or 'val' to determine the dataset split.
        agent_num (int, optional): ID of the current agent (used in multi-agent training).
        n_chunks (int, optional): Number of data chunks to load per agent for training.
    """
    def __init__(self, split, agent_num=None, n_chunks=None):
        if split == "val" or n_chunks is None:
            self.data = torch.load("val_tokenized.pt")
        elif split == "train":
            input_ids = []
            masks = []
            self.data = {}

            for i in range(agent_num * n_chunks, (agent_num + 1) * n_chunks):
                data = torch.load(f"train_tokenized_{i}.pt")
                input_ids.append(data["input_ids"])
                masks.append(data["attention_mask"])

            self.n_chunks = n_chunks
            self.data["input_ids"] = torch.cat(input_ids, dim=0)
            self.data["attention_mask"] = torch.cat(masks, dim=0)

            print(f"Agent {agent_num} has loaded files {agent_num * n_chunks} - {(agent_num + 1) * n_chunks - 1}")

    @property
    def shape(self):
        """
        Returns the shape of the dataset in terms of number of sequences and tokens.

        Returns:
            torch.Size: Shape of input_ids tensor.
        """
        return self.data["input_ids"].shape

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of token sequences.
        """
        return self.shape[0]

    def __getitem__(self, index):
        """
        Returns input and target token sequences for next-token prediction.

        Args:
            index (int): Index of the data sample to fetch.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of input tokens and target tokens.
        """
        tokens = self.data["input_ids"][index]
        inputs = tokens[:-1]
        targets = tokens[1:]
        return inputs, targets
