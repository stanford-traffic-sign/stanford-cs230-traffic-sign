from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


BATCH_SIZE = 64
TRAIN_DATA_PATH = './data/train/'
VAL_DATA_PATH = './data/val/'

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_data = ImageFolder(root=VAL_DATA_PATH, transform=val_transform)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
