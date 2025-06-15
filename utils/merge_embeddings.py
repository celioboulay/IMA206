import os
import torch

def merge(embeddings_dir, output_file, dim=0):

    tensors = [f for f in os.listdir(embeddings_dir) if f.endswith('.pt')]
    if len(tensors) != 2: raise ValueError('please compute embeddings before clustering')
    tensors.sort()

    tensor_list = [torch.load(os.path.join(embeddings_dir, f)) for f in tensors]



    '''
    f_alpha = load_model(...)

    boucle dec

    tant que pas implemente on utilisera
    embeddings = cat(z1, z2)
    '''

    merged_tensor = torch.cat(tensor_list, dim=dim)


    torch.save(merged_tensor, output_file)