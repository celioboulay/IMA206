import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim=10, multiplier=1):
        super(Encoder, self).__init__()

        # Layer parameters
        self.latent_dim = latent_dim
        self.multiplier = multiplier

        # Shape at the end of conv3
        self.reshape = nn.Flatten()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 0)
        self.conv3 = nn.Conv2d(32, 32, 4, 2, 0)

        # Fully connected layers
        self.lin1 = nn.Linear(2*2*32, 64)
        self.lin2 = nn.Linear(64, self.latent_dim * self.multiplier)

    def forward(self, x):
        '''
        Pass the input image mini-batch through conv, linear layers and
        non-linearities to output a (B,D,2) tensor where B is the mini-batch
        size and D the latent dimension.
        '''
        batch_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = self.reshape(x)

        # Fully connected layer with ReLu activation
        x = F.relu(self.lin1(x))

        # Fully connected layer for code z, or mean and log-variance
        x = self.lin2(x)

        # The shape of the output tensor should be (B,D) if multiplier=1,
        # where B is the batch size, and D the latent dimension.
        # Otherwise it should be (B,D,multiplier)
        if self.multiplier == 1:
            x = x.view(batch_size, self.latent_dim)
        else:
            x = x.view(batch_size, self.latent_dim, self.multiplier)

        return x

class Decoder(nn.Module):

    def __init__(self, latent_dim=10):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        # Shape required to start transpose convs (copy paste from the encoder)
        self.reshape = nn.Unflatten(1, (32, 2, 2))

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, 64)
        self.lin2 = nn.Linear(64, 2*2*32)

        # Convolutional layers
        self.convT1 = nn.ConvTranspose2d(32, 32, 4, 2, 0)
        self.convT2 = nn.ConvTranspose2d(32, 32, 4, 2, 0)
        self.convT3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        z = F.relu(self.lin1(z))
        z = F.relu(self.lin2(z))

        # Reshape
        z = self.reshape(z)

        # Convolutional layers with ReLu activations
        z = F.relu(self.convT1(z))
        z = F.relu(self.convT2(z))

        # Final conv layer with sigmoid activation
        z = torch.sigmoid(self.convT3(z))

        return z

def reconstruction_loss(reconstructions, data):
    """
    Calculates the reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, 1,
        height, width).

    reconstructions : torch.Tensor
        Reconstructed data. Shape : (batch_size, 1, height, width).

    Returns
    -------
    loss : torch.Tensor
        Binary cross entropy, AVERAGED over images in the batch but SUMMED over
        pixel and channel.
    """
    batch_size, n_chan, height, width = reconstructions.size()

    # The pixel-wise loss is the binary cross-entropy, computed from
    # reconstructions and data. It is summed over pixels and averaged across
    # samples in the batch.
    loss = F.binary_cross_entropy(reconstructions, data, reduction='none')
    loss = loss.sum(dim=(1,2,3))
    return loss.mean()

class VAEModel(nn.Module):
    def __init__(self, latent_dim):
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, 2)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, logvar, mode='sample'):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)

        mode : 'sample' or 'mean'
            Returns either a sample from qzx, or just the mean of qzx. The former
            is useful at training time. The latter is useful at inference time as
            the mean is usually used for reconstruction, rather than a sample.
        """
        if mode=='sample':
            # Implements the reparametrization trick (slide 43):
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        elif mode=='mean':
            return mean
        else:
            return ValueError("Unknown mode: {mode}".format(mode))

    def forward(self, x, mode='sample'):
        """
        Forward pass of model, used for training or reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)

        mode : 'sample' or 'mean'
            Reconstructs using either a sample from qzx or the mean of qzx
        """

        # stats_qzx is the output of the encoder
        stats_qzx = self.encoder(x)

        # Use the reparametrization trick to sample from q(z|x)
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1), mode=mode)

        # Decode the samples to image space
        reconstructions = self.decoder(samples_qzx)

        # Return everything:
        return {
            'reconstructions': reconstructions,
            'stats_qzx': stats_qzx,
            'samples_qzx': samples_qzx}

    def sample_qzx(self, x):
        """
        Returns a sample z from the latent distribution q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))
        return samples_qzx

    def sample_pz(self, N):
        samples_pz = torch.randn(N, self.latent_dim, device=self.encoder.conv1.weight.device)
        return samples_pz

    def generate_samples(self, samples_pz=None, N=None):
        if samples_pz is None:
            if N is None:
                return ValueError("samples_pz and N cannot be set to None at the same time. Specify one of the two.")

            # If samples z are not provided, we sample N samples from the prior
            # p(z)=N(0,Id), using sample_pz
            samples_pz = self.sample_pz(N) # FILL IN CODE

        # Decode the z's to obtain samples in image space (here, probability
        # maps which can later be sampled from or thresholded)
        generations = self.decoder(samples_pz) # FILL IN CODE
        return {'generations': generations}

def kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    """
    # To be consistent with the reconstruction loss, wetake the mean over the
    # minibatch (i.e., compute for each sample in the minibatch according to
    # the equation above, then take the mean).
    latent_kl = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).sum(dim=1)  # sum sur latent_dim
    return latent_kl.mean()

class BetaVAELoss(object):
    """
    Compute the Beta-VAE loss

    Parameters
    ----------
        beta: (scalar) the weight assigned to the regularization term
    """

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, reconstructions, data, stats_qzx):
        stats_qzx = stats_qzx.unbind(-1)

        # Reconstruction loss
        rec_loss = reconstruction_loss(reconstructions, data)

        # KL loss
        kl_loss = kl_normal_loss(*stats_qzx)

        # Total loss of beta-VAE
        loss = rec_loss + self.beta*kl_loss

        return loss


# Définir le prétraitement des images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Assurez-vous que les images sont en niveaux de gris
    transforms.Resize((32, 32)),  # Redimensionner les images
    transforms.ToTensor(),  # Convertir en tenseur
    transforms.Normalize((0.5,), (0.5,))  # Normalisation des pixels entre -1 et 1
])

# Charger les images depuis le dossier /data
dataset = datasets.ImageFolder(root='/data', transform=transform)

# Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Créer les DataLoader pour l'entraînement et le test
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définir le périphérique (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialiser le modèle et l'optimiseur
vae_model = VAEModel(latent_dim=10).to(device)
vae_loss = BetaVAELoss(beta=1.0)  # Remplacez 1.0 par la valeur de beta que vous souhaitez
optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

# Entraînement du modèle
n_epoch = 10  # Nombre d'époques

vae_model.train()

for epoch in range(n_epoch):
    train_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for data, _ in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # Mettre les données sur le bon périphérique
            data = data.to(device)

            # Passer les données dans le modèle
            predict = vae_model.forward(data)
            reconstructions = predict['reconstructions']
            stats_qzx = predict['stats_qzx']

            # Calculer la perte Beta-VAE
            loss = vae_loss(reconstructions, data, stats_qzx)

            # Backpropagation
            vae_model.zero_grad()
            loss.backward()
            optimizer.step()

            # Agréger la perte d'entraînement pour affichage à la fin de l'époque
            train_loss += loss.item()

            # Affichage de la barre tqdm avec la perte
            tepoch.set_postfix(loss=loss.item())

    # Affichage de la perte à la fin de l'époque
    print(f'Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}')
