import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from global_features import compute_global
from local_features import compute_local
import utils
from clusering.clustering import cluster



def parse_args():
    parser = argparse.ArgumentParser(description="args")
    
    parser.add_argument('--data_path', type=str, required=True, help='images folder path')
    parser.add_argument('--embedding_dir', type=str, default='./embeddings', help='embeddings folder path')
    parser.add_argument('--recompute_embeddings', type=bool, default=True, help='do we recompute embeddings or just do clustering?')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--force_cpu', type=bool, default=False, help='if cuda unavailable')
    parser.add_argument('--verbose', action='store_true', help='logs')
    # parser --n_epochs

    args = parser.parse_args()
    return args



def main():    # python main.py --data_path ./Data/ --embedding_dir ./embeddings --verbose
    args = parse_args()
    if args.verbose:
        print("Args:", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu: device=torch.device("cpu")


    if args.recompute_embeddings:
        compute_local.compute(args.data_path, args.embedding_dir, device)  # f_theta_1
        compute_global.compute(args.data_path, args.embedding_dir, device)  # f_theta_2

    
    utils.merge_embeddings(args.embedding_dir, device, merged_embeddings_path='./clustering/z_merged.pt') # f_alpha  # verif les paths
    clusters = cluster(args.embedding_dir, device, method='dec')    # dec agit sur f_alpha

    utils.visualize(clusters, device)


if __name__ == "__main__":
    main()