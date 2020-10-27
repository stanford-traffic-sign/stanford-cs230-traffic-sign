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


if __name__ == '__main__':
    model = Net()
    model_path = './my_model.pth'
    model.load_state_dict(torch.load(model_path))
    evaluate(model, val_data_loader)
