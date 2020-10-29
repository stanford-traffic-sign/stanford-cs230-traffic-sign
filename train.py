from torch import nn, optim
import torch

from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
import config
import model.net as net


def train(model, optimizer, loss_fn, train_data_loader, val_data_loader, model_path, total_epoch):
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        print(f'Epoch {epoch + 1}/{total_epoch}')
        print('----------')
        running_loss = 0.0
        for i, (batch_inputs, batch_labels) in enumerate(train_data_loader):
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
                print(f'- {i + 1} mini batches\tloss {round(running_loss / steps, 3)}')
                running_loss = 0.0

        train_accuracy, train_loss, train_total = evaluate(model, loss_fn, train_data_loader)
        val_accuracy, val_loss, val_total = evaluate(model, loss_fn, val_data_loader)
        print(f'Train\taccuracy {round(train_accuracy * 100, 2)}%\tloss {round(train_loss, 3)}\timages {train_total}')
        print(f'Val\taccuracy {round(val_accuracy * 100, 2)}%\tloss {round(val_loss, 3)}\timages {val_total}')
        print()

    torch.save(model.state_dict(), model_path)
    print(f'Train finished. Model saved at {model_path}')


if __name__ == '__main__':
    model = net.Net()
    loss_fn = net.loss_fn
    learning_rate = 0.001
    total_epoch = 4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer, loss_fn, train_data_loader, val_data_loader, config.model_path, total_epoch)
