import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sequentials import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=128, shuffle=False)


# cf github

latent_dim = 64
model = Autoencoder(latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



n_epochs = 6
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.size(0)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader.dataset):.6f}")

torch.save(model.state_dict(), "Models/autoencoder_mnist.pth")

model.eval()
data_iter = iter(test_loader)
images, _ = next(data_iter)
images = images.to(device)
with torch.no_grad():
    reconstructions = model(images)

fig, axs = plt.subplots(2, 5, figsize=(12,5))
for i in range(5):
    axs[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
    axs[0, i].set_title("Original")
    axs[0, i].axis('off')
    axs[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
    axs[1, i].set_title("Reconstruit")
    axs[1, i].axis('off')
plt.show()
