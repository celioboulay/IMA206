import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import torch
from torchvision import transforms
from PIL import Image
import timm
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

from utils.transformations_init import *
from global_features.get_embeddings  import *
from global_features.simCLR_like import global_SSL


'''def load_model(device, model_path):
    model = timm.create_model('vit_large_patch16_224', pretrained=False)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model'''

from torchvision.models.feature_extraction import create_feature_extractor

def load_model(device, model_name):
    try:
        model = timm.create_model(model_name, pretrained=True)
        if hasattr(model, 'forward_features'):
            model.get_embedding = model.forward_features
        else:
            model.get_embedding = lambda x: model.forward(x)
    
    except Exception:
        if model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            return_nodes = {'avgpool': 'features'}
            extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
            extractor.get_embedding = lambda x: extractor(x)['features'].squeeze(-1).squeeze(-1)
            model = extractor

        # elif model_name == 'custom_vit':
        #     from my_models import CustomViT
        #     model = CustomViT()
        #     model.load_pretrained_weights()  
        #     model.get_embedding = model.forward

        else:
            raise ValueError(f"Modèle {model_name} non reconnu.")

    return model



def compute(data_path, embedding_dir, device):

    dataset = datasets.ImageFolder(data_path, transform=transform_local_center)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    f_theta_2 = timm.create_model('vit_large_patch16_224', pretrained=True) # modele temporaire pour simplifier le squelette
    # f_theta_2 = load_model
    f_theta_2.to(device)
    f_theta_2.eval()

    # global SLL args a passer dans compute
    f_theta_2 = global_SSL(data_path, device, f_theta_2, n_epochs=1, batch_size=8, lr=1e-3, weight_decay=1e-6, temperature=0.5)  # data_path parce quon va charger un dataset different
    features_tensor = get_embeddings(dataloader, device, model=f_theta_2)
    embeddings = process_features(features_tensor)
    torch.save(embeddings, embedding_dir+"/global_embeddings.pt")


if __name__ == "__main__":
    data_path = "./Data"
    embedding_dir = "./embeddings/global"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute(data_path, embedding_dir, device)