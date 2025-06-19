import torch

def compute(data_path, embedding_dir, device) :
    # tant qu'on a rien de la team local........
    tensor = torch.load('./embeddings/global_embeddings.pt')
    torch.save(tensor,'./embeddings/local_embeddings.pt')
    