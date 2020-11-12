from model.data_loader import val_data_loader
from model.net import net
import torch
import torch.nn.functional as F

from utils.data_class import num_classes, class_map
from utils.writer import writer


def add_pr_curve_tensorboard(class_map, class_index, test_probs, test_preds, global_step=0):
    """
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(class_map[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


if __name__ == '__main__':
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in val_data_loader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # plot all the pr curves
    for i in range(num_classes):
        add_pr_curve_tensorboard(class_map, i, test_probs, test_preds)
