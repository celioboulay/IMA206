
import os
import glob
import random
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap


import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from sklearn.cluster import KMeans

from resnet import extract_features
import visualize



file_paths = glob.glob('./Data/temp/*.jpg')
random.shuffle(file_paths)
file_paths_subset = file_paths#[:2000]   # paintings

embeddings = np.array([extract_features(fp) for fp in tqdm(file_paths_subset)]) # tenseur de taille N,2048 contenant pour chaque image les features associées
print(embeddings.shape)
painters = [os.path.basename(fp).split('_')[0] for fp in file_paths_subset] # tenseur de taille N,1 contenant le nom du peintre associé à chaque tableau




painter_to_indices = {}  # on va regarder quels indices correspondent à des tableau du même peintre
for idx, painter in enumerate(painters):
    if painter not in painter_to_indices:
        painter_to_indices[painter] = []
    painter_to_indices[painter].append(idx)

edge_index = []      # on trace une edge entre deux tableaux du même auteur
for indices in painter_to_indices.values():
    for i in indices:
        for j in indices:
            if i != j:
                edge_index.append([i, j])



edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()   # indexes of pairs of connected edges
x = torch.tensor(embeddings, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
data.x = F.normalize(data.x, p=2, dim=-1)



input_dim = data.x.shape[1]


n_clusters=3


kmeans = KMeans(n_clusters=n_clusters, random_state=0) # changer de méthode
labels = kmeans.fit_predict(data.x)


visualize.plot_all_clusters_images(file_paths_subset, labels, n_clusters=n_clusters, n_images=7, img_size=(2, 1))