from model.net import Net, loss_fn
import numpy as np
import torch

from model.data_loader import val_data_loader
from utils.data_class import class_names
import config


def evaluate(model, loss_fn, data_loader):
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for (images, labels) in data_loader:
            outputs = model(images)
            _, predicts = torch.max(outputs.data, dim=1)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            total += labels.size(0)
            correct += torch.sum(predicts == labels).item()

    accuracy = correct / total
    return accuracy, np.mean(losses), total


def evaluate_by_class(model, data_loader, class_names):
    class_correct = list(0. for _ in range(len(class_names)))
    class_total = list(0. for _ in range(len(class_names)))
    with torch.no_grad():
        for (images, labels) in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = torch.squeeze(predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(class_names)):
        print(f'- {class_names[i]}\t{round(100 * class_correct[i] / class_total[i], 1)}%')


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load(config.model_path))
    val_accuracy, _, total = evaluate(model, loss_fn, val_data_loader)
    print(f'Accuracy {round(val_accuracy * 100, 2)}% ({total} images)')
    evaluate_by_class(model, val_data_loader, class_names)
