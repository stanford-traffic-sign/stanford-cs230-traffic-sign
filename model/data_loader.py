import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

BATCH_SIZE = 10
TRAIN_DATA_PATH = './data/train/'
VAL_DATA_PATH = './data/val/'
# TRANSFORM_IMG = transforms.Compose([
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# TRANSFORM_IMG_TRAIN = transforms['train']
# TRANSFORM_IMG_VAL = transforms['val']

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_data = ImageFolder(root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
