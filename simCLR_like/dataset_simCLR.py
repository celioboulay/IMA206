from torch.utils.data import DataLoader, Dataset
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.image_folder = ImageFolder(root=root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        x1 = self.transform(image)
        x2 = self.transform(image)
        return x1, x2
