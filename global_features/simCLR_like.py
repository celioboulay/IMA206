import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import torch
from tqdm import tqdm

from simCLR_like import dataset_simCLR, nt_xent_loss, augment_transform


def global_SSL(data_path, features_tensor, device, f_theta, n_epochs=10):  # simCLR-like implementation for fine tunning f_theta_2
    '''
    in-place modification de f_theta_2, a voir si on met compute feature a la fin comme prevu ou si c'est plus simple de le mettre dans la boucle principale
    '''
    dataset = dataset_simCLR.SimCLRDataset(root_dir=data_path, transform=augment_transform.ssl_transform)
    
    pass