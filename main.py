import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import *
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
dataset = datasets.ImageFolder('Data/smalldata1/', transform=transform_center_256)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # batch size ? = f(len(dataset)) surement



####### Collecting input features with a pre trained network
from Models.SAE import SAE

f_theta = SAE(NotImplemented)
f_theta.to(device)
f_theta.eval()


with torch.no_grad():
    features_list = []
    for images_batch, _ in dataloader:
        images_batch = images_batch.to(device)
        outputs = f_theta.get_embedding(images_batch)
        features_list.append(outputs.cpu())
        
features_tensor = torch.cat(features_list, dim=0)  # shape: (num_images, 512)
print(features_tensor.shape) # features_tensor embedded data_points, maintenant on fait strandard k-means pour l'init des clusters dans Z



############### Clusters initialization with K-means
from sklearn.cluster import KMeans

n_clusters_init = 10

kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(features_tensor.numpy())
cluster_centers_init = kmeans.cluster_centers_


######## Maintenant qu'on a cette initialisation, on se refere a la partie 3.1 du papier
######## on run dec jusqua converence
from dec.dec import TMM

tmm = TMM(n_clusters=n_clusters_init)
cluster_centers = cluster_centers_init

nb_epochs=15
learning_rate = 1e-3

optimizer = torch.optim.Adam(f_theta.parameters(), lr=learning_rate)
cluster_centers_torch = torch.tensor(cluster_centers_init, device=device, dtype=torch.float)


f_theta.train()
for epoch in tqdm(range(nb_epochs)):
    for images, _ in dataloader:
        images = images.to(device)
        z = f_theta.get_embedding(images)
        q = tmm.compute_soft_assignment(z, cluster_centers_torch, alpha=tmm.alpha)
        p = tmm.compute_target_distribution(q)
        loss = tmm.KL(p, q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



###### On compute les embeddings finaux et dernier k-means

with torch.no_grad():
    final_features_list = []
    for images, _ in dataloader:
        images = images.to(device)
        z = f_theta.get_embedding(images)
        final_features_list.append(z.cpu())

final_embeddings = torch.cat(final_features_list, dim=0)

# Nouveau K-means sur embeddings finaux
final_kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(final_embeddings.numpy())  # pas besoin de repasser sur cpu
final_labels = final_kmeans.labels_


####### visualize(final_features, dim=2)
# from utils.visualize import ce qu'il faut