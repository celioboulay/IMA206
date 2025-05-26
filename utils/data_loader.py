import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch


class Data_loader(object):

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.images = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def load_images(self):
        for file_path in self.file_paths:
            try:
                self.images.append(Image.open(file_path).convert('RGB'))
            except Exception as e:
                print(e)
        return self.images


    def load_folder(self, folder_path):
        self.file_paths = glob.glob(f'{folder_path}/*.jpg')
        return self.load_images()


    def display_image(self, im):
        if isinstance(im, torch.Tensor):
            im = transforms.ToPILImage()(im)
        im.show()


    def extract_patch(self, image, method=None):
        if method is None:
            method = self.transform
        patch = method(image)
        print(patch.shape)
        return patch




file_paths = glob.glob('/Users/celio/Documents/database/afro - afro-basaldella_1912/*.jpg')
data_loader = Data_loader(file_paths)
loaded_images = data_loader.load_images()
print(data_loader.images[0])


if loaded_images:
    im = data_loader.images[0]
    patch = data_loader.extract_patch(im)
    data_loader.display_image(patch)