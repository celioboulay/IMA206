import cv2
import patch_extraction as ext
import VAE
import os
import torch
import numpy as np
from VAE.models.vanilla_vae import VanillaVAE


def extract_patch(image):

    # Compute scores
    patch_scores = ext.compute_gradient_scores(image, patch_size=64, stride=64//2)

    # Keep top 10%
    n_top = int(len(patch_scores)*0.90)
    top_patches = ext.get_top_patches(patch_scores, top_k=n_top)

    # Extract patches
    return top_patches, ext.extract_patches_array_with_dog(image, top_patches, 
                                                          34, top_k=5, apply_dog_to_patches=True,
                                                          dog_sigma1= 1.0, dog_sigma2=2.0)


def compute(data_path, embedding_dir, device) :
    pass