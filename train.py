from torch import nn, optim
import torch

from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
import config
import model.net as net


def train(model, optimizer, loss_fn, train_data_loader, val_data_loader, model_path, num_epochs):
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('----------')
        running_loss = 0.0
        for i, (batch_inputs, batch_labels) in enumerate(train_data_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Print metrics
            running_loss += loss.item()
            steps = 100
            if i % steps == steps - 1:
                print(f'- {i + 1} mini batches\tloss {round(running_loss / steps, 3)}')
                running_loss = 0.0

        train_accuracy, train_loss, train_num_images = evaluate(model, loss_fn, train_data_loader)
        val_accuracy, val_loss, val_num_images = evaluate(model, loss_fn, val_data_loader)

        # Print metrics
        print(f'Train\taccuracy {round(train_accuracy * 100, 2)}%\tloss {round(train_loss, 3)}\timages {train_num_images}')
        print(f'Val\taccuracy {round(val_accuracy * 100, 2)}%\tloss {round(val_loss, 3)}\timages {val_num_images}')
        print()

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f'Train finished. Model saved at {model_path}')


if __name__ == '__main__':
    model = net.Net()
    loss_fn = net.loss_fn
    learning_rate = 0.001
    num_epochs = 15
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer, loss_fn, train_data_loader, val_data_loader, config.model_path, num_epochs)
