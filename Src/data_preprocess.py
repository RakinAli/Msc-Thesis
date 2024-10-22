# src/data_preprocess.py
import os
import torch
from torch.utils.data import Dataset, DataLoader


class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        # Get unique characters from text
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(self.chars)}
        # Get vocab size
        self.vocab_size = len(self.chars)
        self.seq_length = seq_length
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        # Return the number of sequences in the dataset
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Return a single sequence
        x = torch.tensor(self.data[idx : idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(
            self.data[idx + 1 : idx + self.seq_length + 1], dtype=torch.long
        )
        return x, y

# Function to load data
def load_data(file_path, seq_length, batch_size):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    dataset = CharDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader, dataset


if __name__ == "__main__":
    # Example usage
    dataloader, dataset = load_data(
        "../data/harry_potter.txt", seq_length=1024, batch_size=64
    )
    print(f"Vocab Size: {dataset.vocab_size}")
    print(f"Total Sequences: {len(dataset)}")
