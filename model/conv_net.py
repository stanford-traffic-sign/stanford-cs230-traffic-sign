import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.res_net import ResNet, BasicBlock
from utils.data_class import num_classes


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.channel = 3

        self.conv1 = nn.Conv2d(self.channel, 8, kernel_size=5)  # in_channels, out_channels
        self.prelu = nn.PReLU(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=1)
        self.pool = nn.MaxPool2d(2, stride=2)  # kernel_size
        self.bn1 = nn.BatchNorm2d(8)
        self.conv_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # in_features, out_features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        # input units:  16 * (((((32 / 2) - 4) / 2) - 2)) ** 2 = 16 * 4 * 4
        #               /    \________________________/
        #           channels     output tensor size
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 84),
            nn.ReLU(True),
            nn.Linear(84, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 4 * 4)  # 16 * 4 * 4 refers to self.fc_loc
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        # Transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        x = self.conv_drop(x)
        x = self.bn1(self.pool(x))
        x = F.relu(self.conv3(x))
        x = self.conv_drop(x)
        x = self.bn2(self.pool(x))

        # Flatten the output for each image
        x = x.view(-1, 16 * 5 * 5)  # 16 * 5 * 5 refers to self.fc1
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply log softmax on each image's output
        return F.log_softmax(x, dim=1)
