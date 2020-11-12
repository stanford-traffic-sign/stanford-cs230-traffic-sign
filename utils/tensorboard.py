import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from utils.data_class import class_map


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # Convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the 'images_to_probs' function.
    """
    preds, probs = images_to_probs(net, images)
    # Plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for i in np.arange(4):
        ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[i], one_channel=True)
        ax.set_title('{0}, {1:.1f}%\n(label: {2})'.format(
            class_map[preds[i]],
            probs[i] * 100.0,
            class_map[labels[i].item()]),
            color=('green' if preds[i] == labels[i].item() else 'red'))
    return fig
