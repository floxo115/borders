import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import dataset, collate_fn
from model import CNN

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DATASET_LOCATION = r'dataset/dataset_files'
DATASET_PATH = Path(DATASET_LOCATION)

train_set = dataset(DATASET_PATH.parent, 'training_data.npy')
validation_set = dataset(DATASET_PATH.parent, 'validation_data.npy')
test_set = dataset(DATASET_PATH.parent, 'testing_data.npy')

train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=12, shuffle=True)
validation_loader = DataLoader(validation_set, collate_fn=collate_fn, batch_size=12, shuffle=True)
test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=12, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

cnn = CNN()

cnn.to(device=device)
cnn.double()

optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)
mse = torch.nn.MSELoss()

EPOCHS = 1000
for update in range(EPOCHS):

    if update != 0 and update % 10 == 0:
        torch.save(cnn, r'/content/drive/MyDrive/border_data/models/' + f'model-{update}')

    losses_sum = 0
    n_batches = 0
    for input_arrays, known_arrays, targets in validation_loader:
        input_arrays = input_arrays.to(device)
        known_arrays = known_arrays.to(dtype=torch.bool).to(device)
        targets = targets.to(device)

        # Compute the output
        output = cnn(input_arrays)

        output = output.masked_fill(known_arrays, 0)
        targets = targets.masked_fill(known_arrays, 0)

        loss = mse(output, targets)  # .sum() to create a scalar
        # # Compute the gradients
        loss.backward()
        # # Preform the update

        optimizer.step()
        # Reset the accumulated gradients
        optimizer.zero_grad()

        if True:
            print(f"Update cnn {update}/{EPOCHS}:")
            print(f"loss: {loss}")

        losses_sum += loss
        n_batches += 1

    if update != 0 and update % 10 == 0:
        with open(r'/content/drive/MyDrive/border_data/models/trainloss.csv', 'a') as f:
            f.write(f'{update}, {losses_sum / n_batches}\n')
