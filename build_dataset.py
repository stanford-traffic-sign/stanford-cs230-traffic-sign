from glob import glob
from pathlib import Path
import numpy as np
import shutil

from utils.data_class import class_map


def build_dataset(class_map):
    train_folders = sorted(glob('GTSRB/Final_Training/Images/*'))

    DATA_DIR = Path('data')
    DATASETS = ['train', 'val']

    for dataset in DATASETS:
        for class_name in class_map.values():
            (DATA_DIR / dataset / class_name).mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in class_map.items():
        image_paths = np.array(glob(f'{train_folders[class_idx]}/*.ppm'))
        print(f'{class_name}: {len(image_paths)}')
        np.random.seed(1)
        np.random.shuffle(image_paths)

        dataset_split = np.split(
            image_paths,
            # 80% of the images for training, 20% for validation
            indices_or_sections=[int(0.8 * len(image_paths))])

        dataset_data = zip(DATASETS, dataset_split)

        for dataset, images in dataset_data:
            for img_path in images:
                shutil.copy(img_path, f'{DATA_DIR}/{dataset}/{class_name}/')


if __name__ == '__main__':
    build_dataset(class_map)
