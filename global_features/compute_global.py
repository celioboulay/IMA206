import os
import torch
from torchvision import transforms
from PIL import Image
import timm
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

from utils.transformations_init import *
from init_global import init
from simCLR_like import global_SSL


def compute_features(model, device): # dans compute_global parce que peut etre ca depend de la scale
    pass



def process_features(features_tensor):  # pour ramener les features extraites a la forme choisie, a savoir un .pt pour le moment
    
    embeddings=None

    return embeddings


def compute(data_path, embedding_dir, device):

    f_theta_2 = timm.create_model('vit_large_patch16_224', pretrained=True) # modele temporaire pour simplifier le squelette
    # f_theta_2 = load_model(device, model)
    f_theta_2.to(device)
    f_theta_2.eval()

    features_tensor = init(data_path, device, model=f_theta_2)
    global_SSL(data_path, features_tensor, device, f_theta_2)
    features_tensor = compute_features(data_path)
    embeddings = process_features(features_tensor)
    torch.save(embeddings, embedding_dir+"/global_embeddings.pt")