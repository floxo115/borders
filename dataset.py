import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ex4 import ex4

np.random.seed(0)


class dataset(Dataset):
    def __init__(self, a_path: Path, a_name: str):
        with open(a_path.joinpath(a_name), 'rb') as f:
            self.array = np.load(f)

    def __getitem__(self, index):
        image_array = self.array[index]
        # image = Image.open(to_fetch_path)
        # image_resized = resize(image, [90, 90])
        # image_array = np.array(image_resized)
        image_array = image_array / 255
        border_x = (random.randint(5, 15), random.randint(5, 15))
        border_y = (random.randint(5, 15), random.randint(5, 15))
        return *ex4(image_array, border_x, border_y), image_array

    def __len__(self):
        return len(self.array)


# folders = [f"{i:03}" for i in range(3)]
# folders = np.random.permutation(folders)
# len_folders = len(folders)
# train_folders = folders[:int(len_folders * 0.5)]
# validation_folders = folders[int(len_folders * 0.5):int(len_folders * 0.75)]
# test_folders = folders[int(len_folders * 0.75):]

def collate_fn(data):
    input_arrays = []
    known_arrays = []
    # borders = []
    targets = []
    for el in data:
        input_arrays.append(el[0])
        known_arrays.append(el[1])
        # borders.append(torch.tensor(el[2], dtype=torch.float64))
        targets.append(el[3])

    input_arrays = torch.tensor(input_arrays, dtype=torch.float64).reshape(-1, 1, 90, 90)
    known_arrays = torch.tensor(known_arrays, dtype=torch.float64).reshape(-1, 1, 90, 90)
    # borders = torch.tensor(borders, dtype=torch.float64)
    targets = torch.tensor(targets, dtype=torch.float64).reshape(-1, 1, 90, 90)

    return input_arrays, known_arrays, targets
