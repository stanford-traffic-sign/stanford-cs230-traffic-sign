import numpy as np
import torch

from model.data_loader import val_data_loader
from model.net import Net, loss_fn
from utils.data_class import class_map
from utils.device import device
import config


def evaluate(model, loss_fn, data_loader, device):
    correct = 0
    num_inputs = 0
    losses = []
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicts = torch.max(outputs.data, dim=1)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            num_inputs += labels.size(0)
            correct += torch.sum(predicts == labels).item()

    accuracy = correct / num_inputs
    return accuracy, np.mean(losses), num_inputs


def evaluate_by_class(model, data_loader, class_map, device):
    correct = list(0. for _ in range(len(class_map)))
    num_class = list(0. for _ in range(len(class_map)))
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = torch.squeeze(predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                correct[label] += c[i].item()
                num_class[label] += 1

    for i in range(len(class_map)):
        print(f'- {class_map[i]}\t{round(100 * correct[i] / num_class[i], 1)}%')


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load(config.model_path))
    val_accuracy, _, num_inputs = evaluate(model, loss_fn, val_data_loader)
    print(f'Accuracy {round(val_accuracy * 100, 2)}% ({num_inputs} images)')
    evaluate_by_class(model, val_data_loader, class_map, device)
