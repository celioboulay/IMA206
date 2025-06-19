import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from utils.transformations_init import *

from torchvision.datasets import ImageFolder
from tqdm import tqdm

import cv2

def imread_unicode(path):
    """Lit une image même si le chemin contient des caractères spéciaux."""
    try:
        with open(path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Failed to read {path} with error: {e}")
        return None
    
def get_embeddings(dataloader, data_path, embedding_dir, device, model):

    with torch.no_grad():
        features_list = []
        model.eval()
        i=-1
        for images_batch, _ in tqdm(dataloader, desc="Extracting global features"):
                i+=1
                images_batch = images_batch.to(device)
                if hasattr(model, 'forward_features'):
                    outputs = model.forward_features(images_batch)
                    output=torch.mean(outputs.squeeze(0), axis=0)  
                else:
                    outputs = model(images_batch)
                #outputs = model.get_embedding(images_batch)
                for _, image in enumerate(images_batch):
                    image_np = image.cpu().numpy().transpose(1, 2, 0)
                    image_np = (image_np * 255).astype(np.uint8)
                    save_path = os.path.join(embedding_dir, f"image_{i}.pt")
                torch.save(output, save_path)
        
            
'''    with torch.no_grad() :
    #On prend toutes les images une par une pour en extraire les patchs
        image_extensions = (".jpg", ".jpeg", ".png")
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    print(f"Found image: {image_path}")

                    #Extraction des patchs de l'image
                    image = imread_unicode(image_path)
                    if image is None:
                        print(f"Could not read image: {data_path}") 

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    patch_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)

                    z, _ = model.forward_features(patch_tensor) 
                    z = z.squeeze(0)  

                    embedding.append(z)

                    # Concatène tous les embbedding de patchs
                    merged_embedding = torch.cat(embedding, dim=0)  # shape: (5*d,)


                    # On sauvegarde l'embedding dans le dossier spécifié
                    embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(file)[0]}_embedding.pt")
                    torch.save(merged_embedding, embedding_path)

'''