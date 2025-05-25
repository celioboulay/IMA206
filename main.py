import os
import glob
import random
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap


from sklearn.manifold import TSNE
import matplotlib.cm as cm


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
file_paths_subset = file_paths[:500]   # paintings

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
model = GAE(gcn.GCNEncoder(input_dim, 128))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # lr pas opti



#model.train()
#gcn.train_model(model, optimizer, data, epochs=100, print_every=50)

model.eval()
z_initial = model.encode(data.x, data.edge_index).detach().cpu().numpy()
reducer = umap.UMAP(n_components=3, random_state=42)
z_initial_3d = reducer.fit_transform(z_initial)

max_range = (z_initial_3d.max(0) - z_initial_3d.min(0)).max() / 2
mid = z_initial_3d.mean(0)
limits = [(mid[i] - max_range, mid[i] + max_range) for i in range(3)]


epochs = 100
for epoch in range(1, epochs + 1):
    loss = gcn.train_one_epoch(model, optimizer, data)
    if epoch % 1 == 0:
        model.eval()
        z = model.encode(data.x, data.edge_index).detach().cpu().numpy()
        z_3d = reducer.transform(z)
        visualize.plot_umap(z_3d, painters, epoch, limits)
        model.train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')


##########################################################################################


model.eval()
z = model.encode(data.x, data.edge_index).detach().cpu().numpy()   # same shape as x

print(z.shape)




#############################################################################



n_clusters=10


kmeans = KMeans(n_clusters=n_clusters, random_state=0) # changer de méthode
labels = kmeans.fit_predict(z)


visualize.plot_all_clusters_images(file_paths_subset, labels, n_clusters=n_clusters, n_images=8, img_size=(2, 1))