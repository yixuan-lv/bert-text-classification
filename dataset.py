import torch
from torch.utils.data import Dataset

class ToutiaoDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.long)}