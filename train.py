from torch import nn, optim
import json
import torch

from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
from utils.device import device
import config
import model.net as net


def train(
        model,
        optimizer,
        loss_fn,
        train_data_loader,
        val_data_loader,
        model_path,
        statistics_path,
        num_epochs,
        device):

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('----------')
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            steps = 100
            if i % steps == steps - 1:
                print(f'- {i + 1} mini batches\tloss {round(running_loss / steps, 3)}')
                running_loss = 0.0

        train_accuracy, train_loss, train_num_inputs = evaluate(model, loss_fn, train_data_loader, device)
        val_accuracy, val_loss, val_num_inputs = evaluate(model, loss_fn, val_data_loader, device)

        # Print statistics
        print(f'Train\taccuracy {round(train_accuracy * 100, 2)}%\tloss {round(train_loss, 3)}\timages {train_num_inputs}')
        print(f'Val\taccuracy {round(val_accuracy * 100, 2)}%\tloss {round(val_loss, 3)}\timages {val_num_inputs}')
        print()

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

    # Save model
    torch.save(model.state_dict(), model_path)

    # Save statistics
    with open(statistics_path, 'w') as json_file:
        statistics = {
            'Train Accuracy': train_accuracies,
            'Train Loss': train_losses,
            'Validation Accuracy': val_accuracies,
            'Validation Loss': val_losses,
        }
        json.dump(statistics, json_file)

    print(f'Train finished. Model saved at {model_path}. Statistics saved at {statistics_path}')


if __name__ == '__main__':
    learning_rate = 0.001
    num_epochs = 15

    model = net.Net()
    model.to(device)

    loss_fn = net.loss_fn
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(
        model,
        optimizer,
        loss_fn,
        train_data_loader,
        val_data_loader,
        config.model_path,
        config.statistics_path,
        num_epochs,
        device)
