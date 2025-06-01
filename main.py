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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



####### Collecting input features with a pre trained network
########## A remplacer par un stacked auto encodeur construit de la bonne maniere (a definir en fonction de ce que nous donne l'apprentissage sur les images)
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)     #### tant qu'on a pas le SAE on fait avec resNet 18......
f_theta = nn.Sequential(*list(model.children())[:-1])

f_theta.eval()
f_theta.to(device)

features_list = []

with torch.no_grad():
    for images_batch, _ in dataloader:
        images_batch = images_batch.to(device)
        outputs = f_theta(images_batch) 

        outputs = outputs.view(outputs.size(0), -1) 

        features_list.append(outputs.cpu()) 
        
features_tensor = torch.cat(features_list, dim=0)  # shape: (num_images, 512)
print(features_tensor.shape) # features_tensor embedded data_points, maintenant on fait strandard k-means pour l'init des clusters dans Z



############### Clusters initialization with K-means
from sklearn.cluster import KMeans

n_clusters_init = 10

kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(features_tensor.numpy())
cluster_centers_init = kmeans.cluster_centers_

for cluster in enumerate(cluster_centers_init):  # cluster_centers_init K initial centroids in feature space Z
        print(cluster[1].shape)



######## Maintenant qu'on a cette initialisation, on se refere a la partie 3.1 du papier
######## on run dec jusqua converence
from dec.dec import TMM

nb_epochs=15
learning_rate = 1e-3

optimizer = torch.optim.Adam(f_theta.parameters(), lr=learning_rate)

'''   f_theta.train() a mettre je sais plus ou mais chill
for epoch in range(nb_epochs):
    for images, _ in dataloader:
        z = f_theta(images)
        q_ij = compute_soft_assignment(z, cluster_centers)  # via Student's t-distribution
        p_ij = compute_target_distribution(q_ij)
        loss = kl_divergence(p_ij, q_ij)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''



###### On compute les embeddings finaux et dernier k-means

final_embeddings = f_theta(NotImplemented) # TMM.call ou jsp quoi pour avoir le dernier etat des features
final_labels = kmeans.predict(final_embeddings.detach().numpy())



####### visualize(final_features, dim=2)
# from utils.visualize import ce qu'il faut