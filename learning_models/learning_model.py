import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
from utils.data_loader import *    # notamment Custom loader dataset
import pandas as pd
from Models.sequentials import SimpleCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms



annotations_file = 'Data/Classeur1.csv'
img_dir = 'Data/smalldata'
annotations_df = pd.read_csv(annotations_file, sep=';')


train_df, test_df = train_test_split(annotations_df, test_size=0.2, 
    stratify=annotations_df.iloc[:, 1],  # colonne labels
    random_state=42) # random state pour les tests



# Normalisation à appliquer sur le DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CustomDataset(train_df, img_dir, transform=transform)
test_dataset = CustomDataset(test_df, img_dir, transform=transform)



train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)




nb_channels = 3
nb_classes = 3
learning_rate = 1e-3
n_epochs = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

model = SimpleCNN(nb_channels, nb_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

######################################## Entrainement du modele

train_loss = []

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
        data, target = data.to(device), target.to(device) # pour avoir les donnees et le modele sur le meme device


        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()  # computes dloss/dx for every parameter x which has requires_grad=True
        optimizer.step() # updates the value of x using x.grad

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted==target).sum().item()
        total += target.size(0)

    train_accuracy = correct / total
    avg_train_loss = train_loss / len(train_loader)

    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')



