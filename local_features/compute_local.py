import cv2
import patch_extraction as ext
import VAE
import os
import torch
import numpy as np


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)  # [B, C, H*W]
    G = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    return G / (c * h * w)  # Normalisation


def compute(data_path, file, embedding_dir, device, patch_size=32, nb_patches=5) :

    #Extraction des patchs de l'image

    image = cv2.imread(data_path)
    if image is None:
        return print(f"Could not read image: {data_path}") 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Subfolder for each image's top patches
    base_name = os.path.splitext(os.path.basename(file))[0]
    subfolder = os.path.join(embedding_dir, base_name)
    os.makedirs(subfolder, exist_ok=True)

        # 1. Compute scores
    patch_scores = ext.compute_gradient_scores(image, patch_size=patch_size, stride=patch_size//2)

        # 2. Keep top 30%
    n_top = int(len(patch_scores) * 0.66)
    top_patches = ext.get_top_patches(patch_scores, top_k=n_top)

        # 3. Extract patches
    selected_patches = ext.extract_patches_array_with_dog(image, top_patches, 
                                                                  patch_size, top_k=nb_patches, apply_dog_to_patches=True,
                                                                  dog_sigma1= 1.0, dog_sigma2=2.0)

        # 4. Save patches
    for i, patch in enumerate(selected_patches):
        patch_filename = f"{base_name}_top_{i}.jpg"
        patch_path = os.path.join(subfolder, patch_filename)
        cv2.imwrite(patch_path)
    

    #Passage dans la VAE
    embedding = []
    for patch in selected_patches:
        #Passage dans la VAE
        embedded_patch = 0 #A MODIFIER
        embedded_patch = torch.tensor(embedded_patch)

        #Matrice de Gram de la VAE
        embedded_gram = gram_matrix(embedded_patch)
        embedding.append(embedded_gram)

    #Moyenne des patchs extraits
    return np.mean(embedding, axis=0)