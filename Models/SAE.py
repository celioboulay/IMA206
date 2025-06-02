import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utils.transformations_init import transform_center_256


class SAE(nn.Module): # pour l'instant jsp si faut vraiemnt appeler Stacked AE mais vsy
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 100, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, 5),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(200, 100, 5),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(200, 100, 5)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    

# Trying the SAE 
dataset = datasets.ImageFolder('Data/smalldata/', transform=transform_center_256)
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


train_features, train_labels = next(iter(dataloader))
img = train_features[0].squeeze()
img = img[0, :, :]
label = train_labels[0]