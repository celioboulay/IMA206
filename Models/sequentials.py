import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet18

class CNN_2(nn.Module):
    def __init__(self, nb_channels, nb_classes):
        super(CNN_2, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        
        self.features = nn.Sequential(
            conv_block(nb_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, nb_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    



class Autoencoder(nn.Module):   # for mnist
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (64,7,7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), # 14x14 -> 28x28
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def get_embedding(self, x):
        return self.encoder(x)
    




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
    




##### pour simCLR
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, projection_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, projection_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        base_encoder = resnet18(weights=None)
        num_features = base_encoder.fc.in_features
        base_encoder.fc = nn.Identity()  # remove classification head

        self.encoder = base_encoder
        self.projector = ProjectionHead(input_dim=num_features, projection_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
