import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import torch
import torchvision.models as models
from PIL import Image
import utils.transformations_init


resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load('./Models/resnet50-0676ba61.pth'))
resnet50.fc = torch.nn.Identity()
resnet50.eval()



def extract_features(image_path): # renvoie une tenseur de taille 1,2048 avec les features d'entrée de la fc de resnet50
    img = Image.open(image_path).convert('RGB')
    img_t = utils.transformations_init.transform_pas_ouf(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(img_t)
    return features.squeeze().numpy()