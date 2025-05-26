import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch
from tqdm import tqdm 

class Data_loader(object):

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.images = []


        self.transform = transforms.Compose([ # a changer bien sur 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def load_images(self):
        for file_path in tqdm(self.file_paths, desc="Loading images"):
            try:
                self.images.append(Image.open(file_path).convert('RGB'))
            except Exception as e:
                print(e)
        #return self.images

    def display_image(self, im):
        Image._show(im)


    def extract_patch(self, image, method=None):
        if method is None:
            method = self.transform
        patch = method(image)
        print(patch.shape)
        return patch


all_files_paths = glob.glob('/Users/celio/Documents/database/**/*.jpg', recursive=True)

data_loader = Data_loader(all_files_paths[:600])
loaded_images = data_loader.load_images()
print(data_loader.images[0].size)


im = data_loader.images[0]
patch = data_loader.extract_patch(im)
data_loader.display_image(im)