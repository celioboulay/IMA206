import os
import glob
import random
import numpy as np
from tqdm import tqdm


import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE

from sklearn.cluster import KMeans

from resnet import extract_features
import gcn
import visualize



file_paths = glob.glob('./Data/wikiart/*.jpg')
random.shuffle(file_paths)
file_paths_subset = file_paths[:700]

embeddings = np.array([extract_features(fp) for fp in tqdm(file_paths_subset)]) # tenseur de taille N,2048 contenant pour chaque image les features associées
painters = [os.path.basename(fp).split('_')[0] for fp in file_paths_subset] # tenseur de taille N,1 contenant le nom du peintre associé à chaque tableau

print(embeddings.shape)






painter_to_indices = {}  # on va regarder quels indices correspondent à des tableau du même peintre
for idx, painter in enumerate(painters):
    if painter not in painter_to_indices:
        painter_to_indices[painter] = []
    painter_to_indices[painter].append(idx)

edge_index = []      # pour les edges du graph en init
for indices in painter_to_indices.values():
    for i in indices:
        for j in indices:
            if i != j:
                edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
x = torch.tensor(embeddings, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
data.x = F.normalize(data.x, p=2, dim=-1)




input_dim = data.x.shape[1]
model = GAE(gcn.GCNEncoder(input_dim, 128))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr par défaut


gcn.train_loop(model, optimizer, data, epochs=400, print_every=50)



model.eval()
z = model.encode(data.x, data.edge_index).detach().cpu().numpy()

kmeans = KMeans(n_clusters=30, random_state=0)
labels = kmeans.fit_predict(z)




visualize.plot_all_clusters_images(file_paths_subset, labels, n_clusters=6, n_images=6, img_size=(2, 1))