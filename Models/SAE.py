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
import torch
from tqdm import tqdm
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utils.transformations_init import transform_center_256
from sequentials import Autoencoder

######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


'arXiv:1711.08763v1 [cs.CV] 23 Nov 2017'
'https://stackoverflow.com/questions/51253611/how-to-get-an-autoencoder-to-work-on-a-small-image-dataset'


################## à mettre dans sequentials

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*64*64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*64*64),
            nn.ReLU(),
            nn.Unflatten(1, (64,64,64)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def get_embedding(self, x):
        return self.encoder(x)
    
############### 



# Trying the SAE 
dataset = datasets.ImageFolder('Data/smalldata/', transform=transform_center_256)
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = Autoencoder(latent_dim=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



n_epochs = 3
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for data, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.size(0)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader.dataset):.6f}")
