import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    



def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_loop(model, optimizer, data, epochs=300, print_every=50):
    model.train()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, optimizer, data)
        if epoch % print_every == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
