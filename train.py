from torch import nn, optim
import torch

from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
from model.net import net, loss_fn
from utils.device import device
from utils.tensorboard import plot_classes_preds
from utils.writer import writer
import config


def train(
        net,
        optimizer,
        loss_fn,
        train_data_loader,
        val_data_loader,
        model_path,
        num_epochs,
        device,
        writer):

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
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log statistics
            running_loss += loss.item()
            steps = 50
            if i % steps == steps - 1:
                # Log the running loss
                # print(f'- {i + 1} mini batches\tloss {round(running_loss / steps, 3)}')
                writer.add_scalar('training loss',
                                  running_loss / steps,
                                  epoch * len(train_data_loader) + i)

                # Log a Matplotlib Figure showing the model's predictions on a random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(net, inputs, labels),
                                  global_step=epoch * len(train_data_loader) + i)

                running_loss = 0.0

        train_accuracy, train_loss, train_num_inputs = evaluate(net, loss_fn, train_data_loader, device)
        val_accuracy, val_loss, val_num_inputs = evaluate(net, loss_fn, val_data_loader, device)

        # Log statistics
        print(f'Train\taccuracy {round(train_accuracy * 100, 2)}%\tloss {round(train_loss, 3)}\timages {train_num_inputs}')
        print(f'Val\taccuracy {round(val_accuracy * 100, 2)}%\tloss {round(val_loss, 3)}\timages {val_num_inputs}')
        print()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

    # Save model
    torch.save(net.state_dict(), model_path)

    print(f'Train finished. Model saved at {model_path}.')


if __name__ == '__main__':
    learning_rate = 0.001
    num_epochs = 50

    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train(
        net,
        optimizer,
        loss_fn,
        train_data_loader,
        val_data_loader,
        config.model_path,
        num_epochs,
        device,
        writer)

    writer.close()
