"""
Simple dataset for LoRA fine-tuning
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class GhibliMarketDataset(Dataset):
    def __init__(self, data_dir, instance_token="<sks>", resolution=512):
        self.data_dir = Path(data_dir)
        self.instance_token = instance_token
        self.resolution = resolution
        
        # Find image files
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_paths = []
        for ext in valid_exts:
            self.image_paths.extend(self.data_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.data_dir.glob(f"*{ext.upper()}"))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_paths)} training images")
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
        
        self.prompt = f"a busy market, in {instance_token} style"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        
        return {
            'pixel_values': image,
            'prompt': self.prompt
        }


def collate_fn(examples):
    pixel_values = torch.stack([ex['pixel_values'] for ex in examples])
    prompts = [ex['prompt'] for ex in examples]
    return {'pixel_values': pixel_values, 'prompts': prompts}


def create_dataloader(data_dir, batch_size=1, instance_token="<sks>", 
                     resolution=512, shuffle=True):
    dataset = GhibliMarketDataset(data_dir, instance_token, resolution)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     collate_fn=collate_fn, pin_memory=True)