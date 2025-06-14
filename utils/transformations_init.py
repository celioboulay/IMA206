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

        sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).unsqueeze(0)
        sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).unsqueeze(0)

        grad = torch.sqrt(F.conv2d(tensor, sobel_x, padding=1)**2 + F.conv2d(tensor, sobel_y, padding=1)**2)

        _, _, H, W = grad.shape
        crop_h, crop_w = min(self.patch_size, H), min(self.patch_size, W)

        if H < self.patch_size or W < self.patch_size:
            top, left = 0, 0
        else:
            stride = self.patch_size // 4
            grad_sums = F.avg_pool2d(grad, kernel_size=self.patch_size, stride=stride) * (self.patch_size**2)
            idx = torch.argmax(grad_sums)
            idx_h = (idx // grad_sums.shape[-1]) * stride
            idx_w = (idx % grad_sums.shape[-1]) * stride
            top, left = idx_h.item(), idx_w.item()

        return transforms.functional.crop(img, top=top, left=left, height=crop_h, width=crop_w)



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
    SelectStrongestGradientPatch(patch_size=512),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_high_gradient = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    SelectStrongestGradientPatch(patch_size=1000),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])




simclr_transform = transforms.Compose([  # randomized
    SelectStrongestGradientPatch(patch_size=256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
])
