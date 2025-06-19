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


def get_embeddings(dataloader, device, model, batch_size=16):

    with torch.no_grad():
        features_list = []
        for images_batch, _ in tqdm(dataloader, desc="Extracting global features"):
            images_batch = images_batch.to(device)
            if hasattr(model, 'forward_features'):
                outputs = model.forward_features(images_batch)
            else:
                outputs = model(images_batch)
            #outputs = model.get_embedding(images_batch)
            features_list.append(outputs.cpu())
            
    features_tensor = torch.cat(features_list, dim=0)

    return features_tensor




def process_features(features_tensor):  # pour ramener les features extraites a la forme choisie, a savoir un .pt pour le moment
    embeddings=features_tensor
    return embeddings
