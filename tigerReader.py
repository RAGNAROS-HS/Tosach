import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd  # or line-by-line parsing for efficiency

class SynthTigerDataset(Dataset):
    def __init__(self, root_dir, gt_file='gt.txt', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # List of (full_img_path, label)
        with open(f'{root_dir}/{gt_file}', 'r') as f:
            for line in f:
                img_rel_path, label = line.strip().split('\t')
                self.data.append((f'{root_dir}/{img_rel_path}', label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label  # Tokenize label as needed for your model

# Usage
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = SynthTigerDataset(root_dir='./synthtiger', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
