import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)




import cv2
import numpy as np
import shutil
import random
from xml.dom import minidom
import matplotlib.pyplot as plt 
from scipy.optimize import linear_sum_assignment
import torch

from scipy.spatial.distance import cdist



class TMM(object):
  
    def __init__(self, n_clusters=1, alpha=1):
        self.n_clusters = n_clusters
        self.tol = 1e-3
        self.alpha = float(alpha)


    def cluster_acc(self, Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
    

    def KL(self, p, q):
        mask = (p > 0)  # parce que log(0) c'est gênant
        S = torch.sum(p[mask] * torch.log(p[mask] / q[mask]))
        return S
    

    def compute_target_distribution(self, q):
        f = torch.sum(q, dim=0)  # f_i = somme_i(q_ij)
        result = (q**2 / f) / torch.sum(q**2 / f, dim=1, keepdim=True)
        return result  # shape (n_i, n_j)
    

    def compute_soft_assignment(self, z, mu, alpha): # student distribution cf. 3.1.1 DEC + van der Maaten and Hinton (2008)
        dist = torch.sum((z[:, None, :] - mu[None, :, :])**2, dim=2)
        num = (1 + dist / alpha) ** (-(alpha + 1) / 2)
        return num / torch.sum(num, dim=1, keepdim=True)
    

    def clusters_init(self, names=None):
        return 0
    
    def forward(self):
        return 0
    
    def compute_distributions():
        return 0
    