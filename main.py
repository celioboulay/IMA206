import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import KMeans
from tqdm import tqdm

from dec.dec import TMM
from utils.data_loader import *
from utils.transformations_init import *
import torchvision.transforms.functional as TF



########### Chargement des donnees

dataset = datasets.ImageFolder('Data/smalldata1/', transform=transform_center_256)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


####### Collecting input features with a pre trained network
########## A remplacer par un stacked auto encodeur construit de la bonne maniere (a definir en fonction de ce que nous donne l'apprentissage sur les images)


weights = ResNet18_Weights.DEFAULT  #### tant qu'on a pas le SAE on fait avec resNet 18......
model = resnet18(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])

model.eval()
model.to(device)

features_list = []

with torch.no_grad():
    for images_batch, _ in dataloader:
        images_batch = images_batch.to(device)
        outputs = model(images_batch) 

        outputs = outputs.view(outputs.size(0), -1) 

        features_list.append(outputs.cpu()) 
        
features_tensor = torch.cat(features_list, dim=0)  # shape: (num_images, 512)
print(features_tensor.shape)


# features_tensor embedded data_points, maintenant on fait strandard k-means pour l'init des clusters dans Z


############### Clusters initialization with K-means

n_clusters_init = 10

kmeans = KMeans(n_clusters=n_clusters_init, random_state=0).fit(features_tensor.numpy())
cluster_centers_init = kmeans.cluster_centers_

for cluster in enumerate(cluster_centers_init):
        print(cluster[1].shape)

####### cluster_centers_init K initial centroids in feature space Z
# Maintenant qu'on a cette initialisation, on se refere a la partie 3.1 du papier

pre_trained_SAE = NotImplemented

init = pre_trained_SAE(features)


######## on run dec jusqua converence

TMM = dec.TMM()

TMM.forward()
### en gros la boucle



final_features = NotImplemented # TMM.call ou jsp quoi pour avoir le dernier etat des features

######### plot des resultats en dim2 ou 3

# visualize(final_features, dim=2)