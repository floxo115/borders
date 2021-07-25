from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize

def create_array_from_data(a_path, a_folders, a_name):
    print(f'starting to create npy file for {a_name}')
    image_paths = []
    for folder_id in a_folders:
        for image_path in a_path.joinpath(Path(folder_id)).glob('*.jpg'):
            image_paths.append(image_path)

    images = []
    for to_fetch_path in image_paths:
        image = Image.open(to_fetch_path)
        image_resized = resize(image, [90, 90])
        image_array = np.array(image_resized)
        image_array.reshape(90, 90)
        images.append(image_array)

    print(f'saving array')
    images = np.array(images)
    with open(a_path.parent.joinpath(f'{a_name}.npy'), 'wb') as f:
        np.save(f, images)

if __name__ == '__main__':
    DATASET_LOCATION = r'dataset/dataset_files'
    DATASET_PATH = Path(DATASET_LOCATION)

    folders = [f"{i:03}" for i in range(400)]
    folders = np.random.permutation(folders)
    len_folders = len(folders)
    train_folders = folders[:int(len_folders * 0.70)]
    validation_folders = folders[int(len_folders * 0.7):int(len_folders * 0.75)]
    test_folders = folders[int(len_folders * 0.75):]

    DATASET_PATH = Path(DATASET_LOCATION)
    create_array_from_data(DATASET_PATH, train_folders, 'training_data')
    create_array_from_data(DATASET_PATH, validation_folders, 'validation_data')
    create_array_from_data(DATASET_PATH, test_folders, 'testing_data')