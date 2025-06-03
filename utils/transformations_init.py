from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

'''
On met ici les transformations qu'on utilisera ailleurs
'''

class SelectStrongestGradientPatch:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

    def __call__(self, img):
        gray = transforms.functional.to_grayscale(img, num_output_channels=1)
        tensor = transforms.functional.to_tensor(gray).unsqueeze(0)

        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0)
        
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)

        grad = torch.sqrt(grad_x**2 + grad_y**2)

        _, _, H, W = grad.shape
        stride = self.patch_size // 4  # overlap
        kernel_size = self.patch_size

        grad_sums = F.avg_pool2d(grad, kernel_size=kernel_size, stride=stride, padding=0) * (kernel_size**2)

        idx = torch.argmax(grad_sums)   # pooling pour trouver le max
        idx_h = (idx // grad_sums.shape[-1]) * stride
        idx_w = (idx % grad_sums.shape[-1]) * stride

        patch = transforms.functional.crop(img, top=idx_h.item(), left=idx_w.item(), height=self.patch_size, width=self.patch_size)

        return patch



transform_resize_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


transform_center_256 = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])



transform_local_center = transforms.Compose([
    SelectStrongestGradientPatch(patch_size=64),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_high_gradient = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    SelectStrongestGradientPatch(patch_size=256),   # on choisi le patch avec le gradient maximal (eq point d'interet)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


transform_trop_bien = None    # aled