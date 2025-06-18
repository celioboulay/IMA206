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

    #On initialise notre modèle
    device = torch.device('cuda')
    vae_model = VanillaVAE(in_channels=1, latent_dim=64, kld_weight=1e-3).to(device)
    checkpoint = torch.load("./vae_checkpoint_epoch4_batch44345.pt", map_location=torch.device('cuda'))
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    vae_model.eval()
    
    with torch.no_grad :
    #On prend toutes les images une par une pour en extraire les patchs
        
        image_extensions = (".jpg", ".jpeg", ".png")

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    print(f"Found image: {image_path}")

                    #Extraction des patchs de l'image
                    #Extraction des patchs de l'image
                    image = cv2.imread(data_path)
                    if image is None:
                        print(f"Could not read image: {data_path}") 

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    patch_scores, selected_patches = extract_patch(image)
                    print(selected_patches.shape)
                    ext.visualize_patches(image, patch_scores, patch_size=64, k = 5)

                    #Passage dans la VAE
                    embedding = []

                    for patch in selected_patches:
                        
                        patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                        z, _ = vae_model.encode(patch_tensor) 
                        z = z.squeeze(0)  

                        embedding.append(z)

                    # Concatène tous les embedding de patchs
                    merged_embedding = torch.cat(embedding, dim=0)  # shape: (5*d,)

                    embedding = []


                    # On sauvegarde l'embedding dans le dossier spécifié
                    embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(file)[0]}_embedding.pt")
                    torch.save(merged_embedding, embedding_path)
