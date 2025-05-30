import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
from utils.data_loader import *    # notamment Custom loader dataset
import pandas as pd
from Models.sequentials import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.transformations_init import *
from sklearn.model_selection import train_test_split

###############################################################
# Learning models with our data to find interesting features
###############################################################

annotations_file = 'Data/Classeur1.csv'
img_dir = 'Data/smalldata'
annotations_df = pd.read_csv(annotations_file, sep=';')


train_df, test_df = train_test_split(annotations_df, test_size=0.2, 
    stratify=annotations_df.iloc[:, 1])

######## Test train separes, donc on peut operer directement sur les Customdatasets pour generer plusieurs inputs


########## Mettre ci dessous dans une pipeline a part
#### First data_set
train_dataset_resized = CustomDataset(train_df, img_dir, transform=transform_center_256)
test_dataset_resized = CustomDataset(test_df, img_dir, transform=transforms.Compose([transforms.CenterCrop((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]))


####### Exemple
'''plt.figure(figsize=(16, 4))
for i in range(5):
    img, label = train_dataset_resized[i]
    img = img.permute(1, 2, 0)  # [C,H,W] -> [H,W,C]
    img = img * 0.5 + 0.5
    plt.subplot(1, 5, i+1)
    plt.imshow(img.numpy())
    plt.title(f"{label}")
    plt.axis('off')

plt.tight_layout()
plt.show()'''
###################################





batch_size=4

train_loader = CustomDataLoader(train_dataset_resized, batch_size=batch_size, shuffle=True)
test_loader = CustomDataLoader(test_dataset_resized, batch_size=batch_size, shuffle=False)




nb_channels = 3
nb_classes = 3
learning_rate = 1e-3
n_epochs = 15

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")



########################################

model = CNN3(nb_classes).to(device)


##################################################################
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



##################################### Eval
model.eval()

classes = ['Delacroix', 'Manet', 'Monet'] 

dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

images = images.cpu().numpy()

'''fig = plt.figure(figsize=(8, 3))
for idx in range(len(images)):
    ax = fig.add_subplot(1, len(images), idx+1)
    img = np.transpose(images[idx], (1, 2, 0))
    img = (img * 0.5) + 0.5  # denormalisation
    ax.imshow(img)
    ax.set_title(f'True: {classes[labels[idx]]}\nPred: {classes[predicted[idx]]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
'''


########### Activation maps
'https://discuss.pytorch.org/t/visualize-feature-map/29597/3'

'''activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.features[0].register_forward_hook(get_activation('conv1'))
data, label = test_dataset_resized[0]
data_unsqueezed = data.unsqueeze(0).to(device)
output = model(data_unsqueezed)
act = activation['conv1'].squeeze()

num_maps_to_show = 16
rows, cols = 4, 4
fig, axes = plt.subplots(rows, cols+1, figsize=(15, 12))

img_orig = data.permute(1, 2, 0).cpu().numpy()
img_orig = (img_orig * 0.5) + 0.5
axes[0, 0].imshow(img_orig)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')
for r in range(1, rows):
    axes[r, 0].axis('off')

for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        if idx < act.size(0):
            axes[i, j+1].imshow(act[idx].cpu().numpy(), cmap='viridis')
            axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f'Feature map {idx+1}')
        else:
            axes[i, j+1].axis('off')

plt.tight_layout()
plt.show()
'''