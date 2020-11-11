import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv_net import ConvNet
from model.res_net import ResNet, BasicBlock
from model.siamese_net import SiameseNet


def Net():
    return SiameseNet()
    # return ConvNet()
    # return ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
    # return ResNet(BasicBlock, [3, 4, 6, 3])  # ResNet34
    # return ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
    # return ResNet(Bottleneck, [3, 4, 23, 3])  # ResNet101
    # return ResNet(Bottleneck, [3, 8, 36, 3])  # ResNet152


loss_fn = nn.CrossEntropyLoss()


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
