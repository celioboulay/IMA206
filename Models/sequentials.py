import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

class SimpleCNN(nn.Module):
    def __init__(self, nb_channels, nb_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(nb_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (256 // 8) * (256 // 8), 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, nb_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x