import torch
import torch.nn as nn

from model.baseline_net import BaselineNet
from model.conv_net import ConvNet
from model.res_net import ResNet, BasicBlock


def Net():
    # return BaselineNet()
    return ConvNet()
    # return ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
    # return ResNet(BasicBlock, [3, 4, 6, 3])  # ResNet34
    # return ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
    # return ResNet(Bottleneck, [3, 4, 23, 3])  # ResNet101
    # return ResNet(Bottleneck, [3, 8, 36, 3])  # ResNet152
    # return torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    # return torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
    # return torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    # return torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    # return torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)


net = Net()
loss_fn = nn.CrossEntropyLoss()
