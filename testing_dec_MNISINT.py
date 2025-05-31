import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm


############ J'ai repris le tp VAE et une autre implementation trouvée sur github

# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)


# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


n_epochs = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=30, device=device):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        self.mean_layer = nn.Linear(latent_dim, latent_dim)    # mean en 30D
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)  # logvar en 30D
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std).to(device)
        z = mean + epsilon * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    

model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD



def train(model, optimizer, epochs, device, x_dim=784):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss

train(model, optimizer, epochs= n_epochs, device=device)





def generate_digit_random():
    z_sample = torch.randn(1, 30).to(device)  # vecteur latent 30D aléatoire
    x_decoded = model.decode(z_sample)
    digit = x_decoded[0].detach().cpu().reshape(28, 28)
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

generate_digit_random()





from sklearn.decomposition import PCA

def plot_latent_space_2d(model, data_loader, device, n=1000):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.view(-1, 784).to(device)
            mean, _ = model.encode(x_batch)
            latents.append(mean.cpu().numpy())
            labels.extend(y_batch.numpy())
            if len(labels) >= n:
                break

    latents = np.concatenate(latents, axis=0)[:n]
    labels = np.array(labels)[:n]

    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', s=20)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f'2D PCA projection of latent space for {n} test images')
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.grid(True)
    plt.show()

plot_latent_space_2d(model, test_loader, device)


