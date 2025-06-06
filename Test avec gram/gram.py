import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from utils.data_loader import *
from utils.transformations_init import *
from utils.visualize import confusion_print
from utils.functions import *
from Models.sequentials import VGGFeatures

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from sklearn.cluster import AgglomerativeClustering, SpectralClustering


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
data_dir='Data/smalldata1/'
dataset = datasets.ImageFolder(root=data_dir, transform=transform_myriam_center_224)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

total_painters = len(dataset.classes)


##### Model
selected_layers = ['8', '15', '22']
model = VGGFeatures(selected_layers)
model.eval()


##### features from VGG
all_features = []
all_labels= []
with torch.no_grad():
    for images, labels in tqdm(DataLoader(dataset, batch_size=32, shuffle=False), desc="computing features from VGG"):
        feats = model(images)
        gram_vectors = []
        for feat in feats:
            gram = gram_matrix(feat)
            flattened = gram.view(gram.size(0), -1)
            gram_vectors.append(flattened)
        all_grams_batch = torch.cat(gram_vectors, dim=1)
        all_features.append(all_grams_batch)
        all_labels.append(labels)

all_features = torch.cat(all_features, dim=0).cpu().numpy()


##### Clustering

n_clusters = total_painters
print(n_clusters)
y_true_tensor = torch.cat(all_labels)
y_true = y_true_tensor.numpy()


clustering_agg = AgglomerativeClustering(n_clusters=n_clusters)
labels_agg = clustering_agg.fit_predict(all_features)
print('agg done')

clustering_spec = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
labels_spec = clustering_spec.fit_predict(all_features)

##### Matrices de confusion
cm_agg = confusion_matrix(y_true, labels_agg)
confusion_print(cm_agg)

cm_spec = confusion_matrix(y_true, labels_spec)
confusion_print(cm_spec)