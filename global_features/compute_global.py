import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import torch
from torchvision import transforms
from PIL import Image
import timm
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

from utils.transformations_init import *
from init_global import init
from simCLR_like import global_SSL

def load_model(device, model_path):
    model = timm.create_model('vit_large_patch16_224', pretrained=False)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def process_features(features_tensor):  # pour ramener les features extraites a la forme choisie, a savoir un .pt pour le moment
    embeddings=features_tensor
    print(embeddings.shape())
    return embeddings


def compute(data_path, device, embedding_dir=current_dir):

    dataset = datasets.ImageFolder(data_path, transform=transform_local_center)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    f_theta_2 = timm.create_model('vit_large_patch16_224', pretrained=True) # modele temporaire pour simplifier le squelette
    # f_theta_2 = load_model(device, model)
    f_theta_2.to(device)
    f_theta_2.eval()

    # global SLL args a passer dans compute
    f_theta_2 = global_SSL(data_path, device, f_theta_2, n_epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-6, temperature=0.5)  # data_path parce quon va charger un dataset different
    features_tensor = init(dataloader, device, model=f_theta_2)
    embeddings = process_features(features_tensor)
    torch.save(embeddings, embedding_dir+"/global_embeddings.pt")