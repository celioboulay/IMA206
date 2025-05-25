import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load('Models/resnet50-0676ba61.pth'))
resnet50.fc = torch.nn.Identity()
resnet50.eval()



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(image_path): # renvoie une tenseur de taille 1,2048 avec les features d'entrée de la fc de resnet50
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(img_t)
    return features.squeeze().numpy()