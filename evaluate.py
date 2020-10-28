from model.net import Net
import torch

from model.data_loader import val_data_loader
from utils.data_class import class_names
import config


def evaluate(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} val images: {round(100 * correct / total, 2)}%')


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
        print(f'Accuracy of {class_names[i]}, {round(100 * class_correct[i] / class_total[i], 1)}%')


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load(config.model_path))
    evaluate(model, val_data_loader)
    evaluate_by_class(model, val_data_loader, class_names)
