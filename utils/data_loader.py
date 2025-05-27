import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch
from tqdm import tqdm 
import transformations_init
import os
import torchvision

class Data_loader(object):

    def __init__(self, file_paths, method):
        self.file_paths = file_paths
        self.images = []  # interet de pas direct charger les images ?
        self.dict = {}
        self.transform = method


    def load_images(self):
        for file_path in tqdm(self.file_paths, desc="Loading images"):
            try:
                self.images=[]  # reset pour eviter les problemes
                self.images.append(torchvision.transforms.functional.pil_to_tensor(Image.open(file_path).convert('RGB')))

                base_name = os.path.basename(file_path)
                image_keys=[]
                image_keys.append(os.path.splitext(base_name)[0])

            except Exception as e:
                print(e)

        self.dict = dict(zip(image_keys, self.images))
        #return self.images



    def display_image(self, im):
        #Image._show(im)
        pass


    def extract_patch(self, image, method=None):
        if method is None:
            method = self.transform
        patch = method(image)
        print(patch.shape)
        return patch





all_files_paths = glob.glob('/Users/celio/Documents/database/**/*.jpg', recursive=True) # extrait de la meme maniere, mais pas dans l'ordre naif

data_loader = Data_loader(all_files_paths[:20], method=transformations_init.transform_pas_ouf)
data_loader.load_images()

print(data_loader.dict['bouguereau_9'])


im = data_loader.images[0]