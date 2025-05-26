import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image



class Data_loader(object):

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.images = []

    def load_images(self):
        for file_path in self.file_paths:
            try: self.images.append(Image.open(file_path))
            except Exception as e: print(e)
        return self.images


    def load_folder(self, folder_path):
        self.file_paths = glob.glob(f'{folder_path}/*.jpg')
        return self.load_images()
    

    def display_image(self, n_image):
        Image._show(self.images[n_image])


    def extract_patch(self, image, size, method):
        patch = None

        return patch

    


file_paths = glob.glob('/Users/celio/Documents/database/afro - afro-basaldella_1912/*.jpg')
data_loader = Data_loader(file_paths)
loaded_images = data_loader.load_images()



if loaded_images:
    data_loader.display_image(2)