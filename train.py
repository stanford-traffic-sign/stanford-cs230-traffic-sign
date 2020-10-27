import torch
from torch import nn, optim
from model.data_loader import train_data_loader
import model.net as net


def train(model, optimizer, loss_fn, data_loader, model_path):
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            steps = 100
            if i % steps == 0:  # print every mini-batch
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + steps, running_loss / steps))
                running_loss = 0.0

    print('Train finished')

    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')


if __name__ == '__main__':
    model = net.Net()
    loss_fn = net.loss_fn
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model_path = './my_model.pth'
    train(model, optimizer, loss_fn, train_data_loader, model_path)
