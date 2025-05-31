import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, annotations_df, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0].replace(';0', ''))
        image = Image.open(img_path).convert("RGB")  
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


######## A voir pour implementer un héritage CustomDataloader(Dataloader) pour extraire les patchs qu'on veut cf partie de Jean ?

class Paintings_Dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0].replace(';0', ''))
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
    

class CustomDataLoader(DataLoader):
    pass