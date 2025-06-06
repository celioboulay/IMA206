import torch

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)  # [B, C, H*W]
    G = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    return G / (c * h * w)  # Normalisation
