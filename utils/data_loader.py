import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm
import torchvision
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import utils.transformations_init


class CustomDataset(Dataset):
    def __init__(self, annotations_df, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path.replace(';0', ''))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


######## A voir pour implementer un héritage CustomDataloader(Dataloader) pour extraire les patchs qu'on veut cf partie de Jean ?

annotations_file = '/Users/celio/Documents/Classeur1.csv'
img_dir = '/Users/celio/Documents/smalldata'
annotations_df = pd.read_csv(annotations_file, sep=';')


train_df, test_df = train_test_split(annotations_df, test_size=0.2, 
    stratify=annotations_df.iloc[:, 1],  # colonne labels
    random_state=42) # random state pour les tests


train_dataset = CustomDataset(train_df, img_dir)
test_dataset = CustomDataset(test_df, img_dir)


im, label = test_dataset[20]
print(im.shape, ' ', label)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


