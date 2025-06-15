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


def init(dataloader, device, model):

    with torch.no_grad():
        features_list = []
        for images_batch, _ in dataloader:
            images_batch = images_batch.to(device)
            outputs = model.get_embedding(images_batch)
            features_list.append(outputs.cpu())
            
    features_tensor = torch.cat(features_list, dim=0)

    return features_tensor
