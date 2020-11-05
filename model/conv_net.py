import torch.nn as nn
import torch.nn.functional as F

# from model.res_net import ResNet, BasicBlock
from utils.data_class import num_classes


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)  # in_channels, out_channels, kernel_size
        self.prelu = nn.PReLU(8)
        self.conv2 = nn.Conv2d(8, 8, 1)
        self.pool = nn.MaxPool2d(2, stride=2)  # kernel_size
        self.bn1 = nn.BatchNorm2d(8)
        self.conv_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(8, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # in_features, out_features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        x = self.conv_drop(x)
        x = self.bn1(self.pool(x))
        x = F.relu(self.conv3(x))
        x = self.conv_drop(x)
        x = self.bn2(self.pool(x))

        # Flatten the output for each image
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply log softmax on each image's output
        return F.log_softmax(x, dim=1)
