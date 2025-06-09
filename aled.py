import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from Models.sequentials import CNN_2
from utils.transformations_init import *


########### Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



########### Donnees
'''   pour  datasets.ImageFolder il faut la structure suivante
Data/dossier/
    peintre1/
        im1
        im2
    peintre2/
        im1
        ....
'''
dataset = datasets.ImageFolder('Data/smalldata/', transform=transform_high_gradient)

train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('data loaded')



#### model

model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)



    ######## Validation
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')



model.eval()
##### test
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

with torch.no_grad():
    for ax in axes.flat:
        idx = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[idx]
        image_input = image.unsqueeze(0).to(device)

        output = model(image_input)
        _, pred = torch.max(output, 1)

        # Image en numpy pour affichage
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        ax.imshow(img_np)
        ax.set_title(f"True: {dataset.classes[label]}\nPred: {dataset.classes[pred.item()]}")
        ax.axis('off')

plt.tight_layout()
plt.show()
