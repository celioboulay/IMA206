import torch.nn as nn

class AE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*32*32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*32*32), nn.ReLU(),
            nn.Unflatten(1, (256, 32, 32)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


    def get_embedding(self, x):
        return self.encoder(x)
    

class Merge_NN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()