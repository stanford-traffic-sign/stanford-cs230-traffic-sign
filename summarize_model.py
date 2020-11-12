from torchsummary import summary

from model.net import net


if __name__ == '__main__':
    summary(net, (3, 32, 32))
