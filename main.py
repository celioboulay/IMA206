import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import features
import utils
from clustering import cluster


######## Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def parse_args():
    parser = argparse.ArgumentParser(description="args")
    
    parser.add_argument('--data_path', type=str, required=True, help='images folder path')
    parser.add_argument('--embedding_dir', type=str, default='./embeddings', help='embeddings folder path')
    parser.add_argument('--recompute_embeddings', type=bool, default=True, help='do we recompute embeddings or just do clustering?')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--verbose', action='store_true', help='logs')

    args = parser.parse_args()
    return args



def main():    # python main.py --data_path ./Data/ --embedding_dir ./embeddings --verbose
    args = parse_args()
    if args.verbose:
        print("Args:", args)


    if args.recompute_embeddings:

        features.compute_global.compute(args.data_path, args.embedding_dir)  # f_theta_1   # compute local effectue aussi simCLR
        features.compute_local.compute(args.data_path, args.embedding_dir)  # f_theta_2

    
    utils.merge_embeddings(args.embedding_dir) # f_alpha
    clusters = cluster(args.embedding_dir, method='dec')    # dec agit sur f_alpha

    utils.visualize(clusters)


if __name__ == "__main__":
    main()