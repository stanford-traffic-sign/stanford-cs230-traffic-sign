from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from model import net
from model.data_loader import val_data_loader
from utils.data_class import class_map
from utils.device import device
import config


def evaluate(model, loss_fn, data_loader, device):
    model.eval()

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


def get_predictions(model, data_loader):
    model = model.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds)
            real_values.extend(labels)
    predictions = torch.as_tensor(predictions).cpu()
    real_values = torch.as_tensor(real_values).cpu()
    return real_values, predictions


def show_confusion_matrix(confusion_matrix, class_names):
    cm = confusion_matrix.copy()

    cell_counts = cm.flatten()

    cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    row_percentages = ["{0:.2f}".format(value) for value in cm_row_norm.flatten()]

    cell_labels = [f"{cnt}\n{per}" for cnt, per in zip(cell_counts, row_percentages)]
    cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])

    df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)

    hmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Sign')
    plt.xlabel('Predicted Sign')


if __name__ == '__main__':
    model = net.Net()
    model.to(device)

    model.load_state_dict(torch.load(config.model_path))
    val_accuracy, _, num_inputs = evaluate(model, net.loss_fn, val_data_loader, device)

    print(f'Accuracy {round(val_accuracy * 100, 2)}% ({num_inputs} images)')

    y_true, y_pred = get_predictions(model, val_data_loader)
    print(classification_report(y_true, y_pred, target_names=class_map.values(), digits=4))

    # cm = confusion_matrix(y_true, y_pred)
    # show_confusion_matrix(cm, class_map.values())
    # plt.show()
