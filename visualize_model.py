import torchvision

from model.net import net
from model.data_loader import train_data_loader
from utils.writer import writer


def show_image_grid(writer, images):
    img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid, one_channel=False)
    writer.add_image('gtsrb_images', img_grid)


def visualize_model(writer):
    writer.add_graph(net, images)


if __name__ == '__main__':
    data_iter = iter(train_data_loader)
    images, labels = data_iter.next()

    show_image_grid(writer, images)
    visualize_model(writer)

    writer.close()
