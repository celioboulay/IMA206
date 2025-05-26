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
    


file_paths = glob.glob('Data/database/afro - afro-basaldella_1912/*.jpg') # test
data_load = Data_loader(file_paths)
loaded_images = data_load.load_images()



if loaded_images:
    print("loaded")