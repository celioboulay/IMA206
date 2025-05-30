import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
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
    



resnet = models.resnet50(pretrained=True)

features_extractor = nn.Sequential(  # On récupère le début du ResNet jusqu'à layer3
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1,
    resnet.layer2,
    resnet.layer3
)
for param in features_extractor.parameters():
    param.requires_grad = False

class CNN3(nn.Module):
    def __init__(self, nb_classes, base_model=features_extractor):
        super().__init__()
        self.features = base_model
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),  # 1024 car layer3 sort du 1024 channels
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, nb_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x