import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import resnet



import cv2
import numpy as np
import shutil
import random
from google import protobuf
from xml.dom import minidom
import matplotlib.pyplot as plt 
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import cdist



class TMM(object):
  
    def __init__(self, n_clusters=1, alpha=1):
        self.n_clusters = n_clusters
        self.tol = 1e-5
        self.alpha = float(alpha)

    def extract_features(image_path, model=None):
        return resnet.extract_features(image_path)   # modifier pour choisir le model si besoin

    def cluster_acc(Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w