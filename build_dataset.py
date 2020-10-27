from pathlib import Path
import numpy as np
from glob import glob
import shutil

if __name__ == '__main__':
    train_folders = sorted(glob('GTSRB/Final_Training/Images/*'))
    class_names = ['priority_road', 'give_way', 'stop', 'no_entry']
    class_indices = [12, 13, 14, 17]

    DATA_DIR = Path('data')
    DATASETS = ['train', 'val', 'test']

    for dataset in DATASETS:
        for class_name in class_names:
            (DATA_DIR / dataset / class_name).mkdir(parents=True, exist_ok=True)

    for i, cls_index in enumerate(class_indices):
        image_paths = np.array(glob(f'{train_folders[cls_index]}/*.ppm'))
        class_name = class_names[i]
        print(f'{class_name}: {len(image_paths)}')
        np.random.shuffle(image_paths)

        dataset_split = np.split(
            image_paths,
            # 80% of the images for training, 10% for validation, and 10% test for each class
            indices_or_sections=[int(0.8 * len(image_paths)), int(0.9 * len(image_paths))])

        dataset_data = zip(DATASETS, dataset_split)

        for dataset, images in dataset_data:
            for img_path in images:
                shutil.copy(img_path, f'{DATA_DIR}/{dataset}/{class_name}/')
