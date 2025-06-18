import cv2
import patch_extraction as ext
import VAE
import os
import torch
import numpy as np
from VAE.models.vanilla_vae import VanillaVAE


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)  # [B, C, H*W]
    G = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    return G / (c * h * w)  # Normalisation

def extract_patch(image):

    # Compute scores
    patch_scores = ext.compute_gradient_scores(image, patch_size=64, stride=64//2)

    # Keep top 10%
    n_top = int(len(patch_scores)*0.90)
    top_patches = ext.get_top_patches(patch_scores, top_k=n_top)

    # Extract patches
    return ext.extract_patches_array_with_dog(image, top_patches, 
                                                          34, top_k=5, apply_dog_to_patches=True,
                                                          dog_sigma1= 1.0, dog_sigma2=2.0)
#on modifiera cette méthode pour obtenir des patchs aléatoirement plutôt que prendre les 5 meilleurs


def compute(data_path, embedding_dir, device) :

    #On initialise notre modèle
    vae_model = VanillaVAE(in_channels=1, latent_dim=64, kld_weight=1e-3).to(device)
    checkpoint = torch.load(".\vae_checkpoint_epoch4_batch44345.pt", map_location=torch.device('cuda'))  # ou 'cuda' si tu es sur GPU
    vae_model.load_state_dict(checkpoint)
    vae_model.eval()
    
    with torch.no_grad :
#Dans la pipeline on imagine qu'on a pris pour une image là

        #Extraction des patchs de l'image
        image = cv2.imread(data_path)
        if image is None:
            return print(f"Could not read image: {data_path}") 

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        selected_patches = extract_patch(image)

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