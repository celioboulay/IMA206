import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import utils.resnet as resnet



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

    def extract_features(self, image_path, model=None):
        return resnet.extract_features(image_path)   # modifier pour choisir le model si besoin

    def cluster_acc(self, Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
    

    def KL(self, p,q): # KL divergence value between p and q
        S=0
        for i in range(len(p)):
            for j in range(len(p)):
                S+=p[i][j]*np.log(p[i][j]/q[i][j])
        return S
    

    def compute_target_distribution(self, q):
        f = np.sum(q, axis=0) # f_i = somme_i(q_ij)   
        result = (q**2 / f) / np.sum(q**2 / f, axis=1, keepdims=True)
        return(result)  # shape (n_i, n_j)
    

    def compute_soft_assignment(self, z, mu): # student distribution cf. 3.1.1 dec + van der MAtten and Hinton (2008)
        dist = np.sum((z[:, np.newaxis, :] - mu[np.newaxis, :, :])**2, axis=2)
        num = (1 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        return num / np.sum(num, axis=1, keepdims=True)