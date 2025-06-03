import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from tqdm import tqdm

from utils.transformations_init import *
import torchvision.models as models
from sequentials import *



######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#### On charge un dataset par transformation
dir = 'Data/smalldata1/'
dataset_1 = datasets.ImageFolder(dir, transform = None)
dataset_2 = datasets.ImageFolder(dir, transform = None)
dataset_3 = datasets.ImageFolder(dir, transform = None)

dataloader_1 = DataLoader(dataset_1, batch_size=32, shuffle=True)
dataloader_2 = DataLoader(dataset_2, batch_size=32, shuffle=True)
dataloader_3 = DataLoader(dataset_3, batch_size=32, shuffle=True)
#######


AE_1 = AE()
AE_2 = AE()
AE_3 = AE()



