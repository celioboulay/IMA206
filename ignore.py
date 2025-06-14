import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from Models.sequentials import AE
from utils.transformations_init import *



######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



######## Chargement des donnees
'''   pour  datasets.ImageFolder il faut la structure suivante
Data/dossier/
    peintre1/
        im1
        im2
    peintre2/
        im1
        ....
'''
dataset = datasets.ImageFolder('Data/smalldata/', transform=transform_local_center)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print('data loaded')



####### Collecting input features with a pre trained network
## Ca va se resumer a import SAE puis charger les poids du SAE pre entraine et passer les donnees a travers

f_theta = AE(latent_dim=100) # meme latent_dim que le modele ci dessous
f_theta.load_state_dict(torch.load("Models/autoencoder.pth", map_location=device))
f_theta = f_theta.to(device)
f_theta.eval()


with torch.no_grad():
    features_list = []
    for images_batch, _ in dataloader:
        images_batch = images_batch.to(device)
        outputs = f_theta.get_embedding(images_batch)
        features_list.append(outputs.cpu())
        
features_tensor = torch.cat(features_list, dim=0)  # shape: (num_images, latent_dim)
print(features_tensor.shape) # features_tensor embedded data_points, maintenant on fait strandard k-means pour l'init des clusters dans Z



############### Clusters initialization with K-means
from sklearn.cluster import KMeans

n_clusters_init = 2

kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(features_tensor.numpy())
cluster_centers_init = kmeans.cluster_centers_


######## Maintenant qu'on a cette initialisation, on se refere a la partie 3.1 du papier
######## on run dec jusqua converence
from dec.dec import TMM

tmm = TMM(n_clusters=n_clusters_init)
cluster_centers = cluster_centers_init

nb_epochs=50
learning_rate = 1e-5

optimizer = torch.optim.Adam(f_theta.parameters(), lr=learning_rate)
cluster_centers_torch = torch.tensor(cluster_centers_init, device=device, dtype=torch.float)


f_theta.train()
for epoch in range(nb_epochs):
    epoch_loss = 0
    for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{nb_epochs}"):
        images = images.to(device)
        z = f_theta.get_embedding(images)
        q = tmm.compute_soft_assignment(z, cluster_centers_torch, alpha=tmm.alpha)
        p = tmm.compute_target_distribution(q)
        loss = tmm.KL(p, q)
        epoch_loss += loss.item() * images.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader.dataset):.6f}")


###### On compute les embeddings finaux et dernier k-means

with torch.no_grad():
    final_features_list = []
    for images, _ in dataloader:
        images = images.to(device)
        z = f_theta.get_embedding(images)
        final_features_list.append(z.cpu())

final_embeddings = torch.cat(final_features_list, dim=0)

# Nouveau K-means sur embeddings finaux
print('final KMeans')
final_kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(final_embeddings.numpy())  # pas besoin de repasser sur cpu
final_labels = final_kmeans.labels_


####### visualize(final_features, dim=2)
# from utils.visualize import ce qu'il faut

fig, axs = plt.subplots(5, 7, figsize=(10, 7))
image_paths = [path for (path, _) in dataset.imgs]
cluster_to_indices = {c: [] for c in range(5)}
for idx, label in enumerate(final_labels):
    if label in cluster_to_indices:
        cluster_to_indices[label].append(idx)
for row, cluster_id in enumerate(range(5)):
    indices = cluster_to_indices[cluster_id]
    random.shuffle(indices)
    for col, idx in enumerate(indices[:7]):
        img = plt.imread(image_paths[idx])
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
        axs[row, col].set_title(os.path.basename(image_paths[idx]), fontsize=6)
plt.tight_layout()
plt.show()