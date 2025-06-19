import cv2
import patch_extraction as ext
import VAE
import os
import torch
import numpy as np
from VAE.models.vanilla_vae import VanillaVAE
from VAE.models.info_vae import InfoVAE

def imread_unicode(path):
    """Lit une image même si le chemin contient des caractères spéciaux."""
    try:
        with open(path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Failed to read {path} with error: {e}")
        return None


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


def compute(data_path, weight_path, embedding_dir, device, model_name="VanillaVAE"): 

    #On initialise notre modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == "InfoVAE":
        vae_model = VanillaVAE(in_channels=1, latent_dim=64, kld_weight=1e-3).to(device)
    else:
        vae_model = InfoVAE(in_channels=1, latent_dim=64, reg_weight=10, kld_weight=1e-3).to(device)

    checkpoint = torch.load(weight_path, map_location=torch.device('cuda'), weights_only=True)

    vae_model.load_state_dict(checkpoint['model_state_dict'])
    vae_model.eval()
    
    with torch.no_grad() :
    #On prend toutes les images une par une pour en extraire les patchs
        
        image_extensions = (".jpg", ".jpeg", ".png")
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    print(f"Found image: {image_path}")

                    #Extraction des patchs de l'image
                    image = imread_unicode(image_path)
                    if image is None:
                        print(f"Could not read image: {data_path}") 

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    patch_scores, selected_patches = extract_patch(image)

                    #Passage dans la VAE
                    embedding = []

                    for patch in selected_patches:
                        
                        patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                        z, _ = vae_model.encode(patch_tensor) 
                        z = z.squeeze(0)  

                        embedding.append(z)

                    # Concatène tous les embedding de patchs
                    merged_embedding = torch.cat(embedding, dim=0)  # shape: (5*d,)


                    # On sauvegarde l'embedding dans le dossier spécifié
                    embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(file)[0]}_embedding.pt")
                    torch.save(merged_embedding, embedding_path)



if __name__ == "__main__":
    data_path = "./Data"
    embedding_dir = "./embeddings/local/info_vae_dim64_reg10_kld1e-3"
    weight_path = "./local_features/VAE/info_vae_latent64_reg10_kld0.001_epoch4_batch5422.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    compute(data_path,weight_path, embedding_dir,  device)