from torch import nn, optim
import torch

from model.data_loader import train_data_loader
import config
import model.net as net


def train(model, optimizer, loss_fn, data_loader, model_path):
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (batch_inputs, batch_labels) in enumerate(data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            steps = 200
            if i % steps == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {round(running_loss / steps, 3)}')
                running_loss = 0.0

    torch.save(model.state_dict(), model_path)
    print(f'Train finished. Model saved at {model_path}')


if __name__ == '__main__':
    model = net.Net()
    loss_fn = net.loss_fn
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    train(model, optimizer, loss_fn, train_data_loader, config.model_path)
