import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class SAE(nn.Module): # pour l'instant jsp si faut vraiemnt appeler Stacked AE mais vsy
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 100, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, 5),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(200, 100, 5),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(200, 100, 5)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    

# Trying the just defined SAE 