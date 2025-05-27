import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)


import numpy as np
from utils.data_loader import *    # notamment Custom loader dataset
import pandas as pd



annotations_file = '/Users/celio/Documents/Classeur1.csv'
img_dir = '/Users/celio/Documents/smalldata'
annotations_df = pd.read_csv(annotations_file, sep=';')


train_df, test_df = train_test_split(annotations_df, test_size=0.2, 
    stratify=annotations_df.iloc[:, 1],  # colonne labels
    random_state=42) # random state pour les tests


train_dataset = CustomDataset(train_df, img_dir)
test_dataset = CustomDataset(test_df, img_dir)


im, label = test_dataset[20]
print(im.shape, ' ', label)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


