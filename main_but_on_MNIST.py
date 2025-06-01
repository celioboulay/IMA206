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

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from Models.sequentials import Autoencoder



######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)



####### Collecting input features with a pre trained network
f_theta = Autoencoder(latent_dim=64).to(device)
f_theta.load_state_dict(torch.load("Models/autoencoder_mnist.pth", map_location=device))
f_theta.eval() 


with torch.no_grad():
    features_list = []
    for images_batch, _ in dataloader:
        images_batch = images_batch.to(device)
        outputs = f_theta.get_embedding(images_batch)
        features_list.append(outputs.cpu())

        
features_tensor = torch.cat(features_list, dim=0) 
print(features_tensor.shape) # features_tensor embedded data_points, maintenant on fait strandard k-means pour l'init des clusters dans Z



############### Clusters initialization with K-means
from sklearn.cluster import KMeans

n_clusters_init = 10

kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(features_tensor.numpy())
cluster_centers_init = kmeans.cluster_centers_  # cluster_centers_init K initial centroids in feature space Z



######## Maintenant qu'on a cette initialisation, on se refere a la partie 3.1 du papier
######## on run dec jusqua converence
# pour le classique avec 10 clusters, 0 epoch forcement c'est parfait
# pour n_clusters_init  = 9 il regroupe 7 et 1

from dec.dec import TMM

tmm = TMM(n_clusters=n_clusters_init)
cluster_centers = cluster_centers_init

nb_epochs=10
learning_rate = 1e-3

optimizer = torch.optim.Adam(f_theta.encoder.parameters(), lr=learning_rate)
cluster_centers_torch = torch.tensor(cluster_centers_init, device=device, dtype=torch.float) # sklearn calcule sur cpu

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


n_clusters = len(np.unique(final_labels))
final_embeddings_np = final_embeddings.numpy()

for c in range(n_clusters):
    idxs = np.where(final_labels == c)[0]
    cluster_embeddings = final_embeddings_np[idxs]
    cluster_mean = cluster_embeddings.mean(axis=0)

    plt.figure(figsize=(10,2))
    for i, idx in enumerate(idxs[:5]):
        plt.subplot(1, 5, i+1)
        img = dataset[idx][0].squeeze().numpy()  # image MNIST (1,28,28) -> (28,28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Idx:{idx}")
        plt.axis('off')
    plt.suptitle(f"Cluster {c}")
    plt.show()
