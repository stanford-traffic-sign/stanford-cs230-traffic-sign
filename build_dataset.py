from pathlib import Path
import numpy as np
from glob import glob
import shutil

if __name__ == '__main__':
    train_folders = sorted(glob('GTSRB/Final_Training/Images/*'))
    class_names = ['speed_30', 'priority_road', 'give_way', 'stop', 'no_entry', 'left_straight', 'roundabout']
    class_indices = [1, 12, 13, 14, 17, 37, 40]

    DATA_DIR = Path('data')
    DATASETS = ['train', 'val']

    for dataset in DATASETS:
        for class_name in class_names:
            (DATA_DIR / dataset / class_name).mkdir(parents=True, exist_ok=True)

    for i, cls_index in enumerate(class_indices):
        image_paths = np.array(glob(f'{train_folders[cls_index]}/*.ppm'))
        class_name = class_names[i]
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
