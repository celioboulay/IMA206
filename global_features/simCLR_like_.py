import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from simCLR_like import dataset_simCLR, nt_xent_loss, augment_transform




def global_SSL(data_path, device, f_theta, n_epochs=10, batch_size=32, lr=1e-3, weight_decay=1e-6, temperature=0.5):  # simCLR-like implementation for fine tunning f_theta_2
    '''
    in-place modification de f_theta_2, a voir si on met compute feature a la fin comme prevu ou si c'est plus simple de le mettre dans la boucle principale
    '''
    dataset = dataset_simCLR.SimCLRDataset(root_dir=data_path, transform=augment_transform.ssl_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # num workers=1 pour l'instant
    optimizer = optim.Adam(f_theta.parameters(), lr=lr, weight_decay=weight_decay)

    f_theta.train().to(device)

    for epoch in range(1, n_epochs+1):

        epoch_loss = 0

        for x1, x2 in tqdm(dataloader, desc=f"epoch [{epoch}/{n_epochs}]"):

            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = f_theta(x1), f_theta(x2)

            loss = nt_xent_loss.nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'epoch [{epoch}/{n_epochs}] - loss: {epoch_loss / len(dataloader):.4f}')                   

    return f_theta.eval().to(device)