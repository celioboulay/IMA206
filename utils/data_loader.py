import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch
from tqdm import tqdm
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils.transformations_init


class ArtDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        base_name = os.path.basename(img_path)
        image_key = os.path.splitext(base_name)[0]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(e)
            return None, image_key

        if self.transform:
            image = self.transform(image)

        return image, image_key


class Data_loader:
    def __init__(self, file_paths, method):
        self.file_paths = file_paths
        self.transform = method
        self.art_dataset = ArtDataset(file_paths=self.file_paths, transform=self.transform)
        self.images = []
        self.dict = {}
        self._load_all_images_to_memory()


    def _load_all_images_to_memory(self):
        for idx in tqdm(range(len(self.art_dataset)), desc="Loading images"):
            img_tensor, image_key = self.art_dataset[idx]
            if img_tensor is not None:
                self.images.append(img_tensor)
                self.dict[image_key] = img_tensor

    def display_image(self, im):
        pass

    def extract_patch(self, image, method=None):
        if method is None:
            method = self.transform
        patch = method(image)
        print(patch.shape)
        return patch

all_files_paths = glob.glob('/Users/celio/Documents/database/**/*.jpg', recursive=True)

data_loader = Data_loader(all_files_paths[:20], method=utils.transformations_init.transform_pas_ouf)

print(data_loader.dict['bouguereau_9'])

im = data_loader.images[0]