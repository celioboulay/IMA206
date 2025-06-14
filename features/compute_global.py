import os
import torch
from torchvision import transforms
from PIL import Image
import timm
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
from utils.transformations_init import *



def compute_features(model): # dans compute_global parce que peut etre ca depend de la scale
    pass

def init_global(data_path, device, model):

    dataset = datasets.ImageFolder(data_path, transform=transform_local_center)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    with torch.no_grad():
        features_list = []
        for images_batch, _ in dataloader:
            images_batch = images_batch.to(device)
            outputs = model.get_embedding(images_batch)
            features_list.append(outputs.cpu())
            
    features_tensor = torch.cat(features_list, dim=0)

    return features_tensor


def global_SSL():  # simCLR-like implementation for fine tunning f_theta_2
    pass


def process_features(features_tensor):  # pour ramener les features extraites a la forme choisie, a savoir un .pt pour le moment
    
    embeddings=None

    return embeddings


def compute(data_path, embedding_dir, device):

    f_theta_2 = timm.create_model('vit_large_patch16_224', pretrained=True) # modele temporaire pour simplifier le squelette
    f_theta_2.to(device)
    f_theta_2.eval()


    features_tensor = init_global(data_path, device, f_theta_2)
    global_SSL(data_path, features_tensor, device, f_theta_2)
    features_tensor = compute_features(data_path)
    embeddings = process_features(features_tensor)
    torch.save(embeddings, embedding_dir+"/global_embeddings.pt")