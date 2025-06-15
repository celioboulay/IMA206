import torch.nn as nn


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
        base_encoder = None #resnet18(weights=None)
        num_features = base_encoder.fc.in_features
        base_encoder.fc = nn.Identity()  # remove classification head

        self.encoder = base_encoder
        self.projector = ProjectionHead(input_dim=num_features, projection_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
