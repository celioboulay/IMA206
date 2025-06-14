import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import features
from load_data import load_data
import utils
from clustering import cluster


######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def parse_args():
    parser = argparse.ArgumentParser(description="a")
    
    # Exemple d'arguments
    parser.add_argument('--data_path', type=str, required=True, help='a')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='a')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--verbose', action='store_true', help='logs')

    args = parser.parse_args()
    return args



def main():    # python main.py --data_path ./Data/ --output_dir ./embeddings --verbose
    args = parse_args()
    if args.verbose:
        print("Args:", args)

    if args.verbose: return 0 # temp

    data = load_data(args.data_path)   # image_Folder
    local_embeddings = features.compute_local(data)    # compute local effectue aussi simCLR
    global_embeddings = features.compute_global(data)   
    embeddings = utils.merge_embeddings(local_embeddings, global_embeddings)
    clusters = cluster(embeddings, method='dec')

    utils.visualize(clusters)


if __name__ == "__main__":
    main()