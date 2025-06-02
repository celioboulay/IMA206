import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from tqdm import tqdm

from utils.transformations_init import transform_center_256
from Models.sequentials import AE

'https://stackoverflow.com/questions/51253611/how-to-get-an-autoencoder-to-work-on-a-small-image-dataset'

######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = datasets.ImageFolder('Data/smalldata1/', transform=transform_center_256)
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


####### Loading AutoEncoder
model = AE(latent_dim=128)
model.to(device)


####### Training AE
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100  
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

torch.save(model.state_dict(), "Models/autoencoder.pth")  # on save le model


######### Test et display pour verif
data_iter = iter(dataloader)
images, _ = next(data_iter)
images = images.to(device)

model.eval()
with torch.no_grad():
    reconstructions = model(images)


def imshow(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)

n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    plt.subplot(2, n, i+1)
    plt.imshow(imshow(images[i]))
    plt.axis('off')
    plt.subplot(2, n, n+i+1)
    plt.imshow(imshow(reconstructions[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()
