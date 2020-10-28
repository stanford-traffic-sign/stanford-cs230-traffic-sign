import torch
from model.data_loader import val_data_loader
from model.net import Net


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
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
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
        print('Accuracy of %5s : %2d %%' % (
            class_names[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    model = Net()
    model_path = './my_model.pth'
    class_names = ['speed_30', 'priority_road', 'give_way', 'stop', 'no_entry', 'left_straight', 'roundabout']
    model.load_state_dict(torch.load(model_path))
    evaluate(model, val_data_loader)
    evaluate_by_class(model, val_data_loader, class_names)
