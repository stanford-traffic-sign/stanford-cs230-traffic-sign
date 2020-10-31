from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from config import data_path


BATCH_SIZE = 128
TRAIN_DATA_PATH = f'{data_path}/train'
VAL_DATA_PATH = f'{data_path}/val'

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.9, 1.1), ratio=(0.75, 1.33)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_nums,  std=std_nums),
])
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_nums,  std=std_nums),
])

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_data = ImageFolder(root=VAL_DATA_PATH, transform=val_transform)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
