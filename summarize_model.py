from torchsummary import summary

import model.net as net


if __name__ == '__main__':
    model = net.Net()
    summary(model, (3, 32, 32))
