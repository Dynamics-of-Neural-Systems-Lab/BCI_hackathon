"""
Training with VRNN
This file was tangled by /home/moritz/wiki/roam/adopt_vrnn_architecture_for_unsupervised_learning_of_sequential_data_issue_1_dynamics_of_neural_systems_lab_bci_hackathon.org

DONT modify here!
"""

import os
import sys

sys.path.insert(1, "/home/moritz/Projects/BCI_hackathon/")  # TODO
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from VariationalRecurrentNeuralNetwork.model import VRNN
from utils.train import TrainConfig

from tqdm import tqdm

from utils.augmentations import get_default_transform
from utils import creating_dataset

# hyperparameters
n_layers = 1
n_epochs = 4
clip = 10
learning_rate = 1e-3  # TODO try
batch_size = 256  # 128
seed = 128

train_config = TrainConfig(
    exp_name="test_2_run_fedya",
    p_augs=0.3,
    batch_size=batch_size,
    eval_interval=150,
    num_workers=0,
)


def get_dataset(train_config):
    ## Data preparation
    transform = get_default_transform(train_config.p_augs)
    data_paths = dict(
        datasets=[DATA_PATH],
        hand_type=["left", "right"],  # [left, 'right']
        human_type=["health", "amputant"],  # [amputant, 'health']
        test_dataset_list=["fedya_tropin_standart_elbow_left"],
    )
    data_config = creating_dataset.DataConfig(**data_paths)
    train_dataset, test_dataset = creating_dataset.get_datasets(
        data_config, transform=transform
    )
    return train_dataset, test_dataset


def preprocess_batch(data):
    # restructure `data` such that always 8 consecutive time steps are given to the model at once
    data = data.view(data.shape[0], 8, -1, 8)  # Shape: [B, C, T/8, 8]
    data = data.permute(0, 1, 3, 2)  # Shape: [B, C, 8, T/8]
    data = data.contiguous().view(data.shape[0], 8 * 8, -1)  # Shape: [B, C*8, T/8]
    data = data.permute(2, 0, 1)  # [B, C, T] -> [T, B, C]
    data = (data - data.min()) / (data.max() - data.min())
    data = data.to(torch.float32)

    return data


def train(train_loader, epoch):
    train_loss = 0
    tqdm_obj = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, labels) in tqdm_obj:

        # transforming data
        data = data.to(device)
        labels = labels.to(device)
        data = preprocess_batch(data)

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, y_loss, _, _, _ = model(data, labels)
        loss = kld_loss + nll_loss + y_loss * 10
        loss.backward()
        optimizer.step()

        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # printing
        tqdm_obj.set_description(
            "Train Epoch: {} [{}/{} ({:.0f}%)] KLD Loss: {:.6f} NLL Loss: {:.6f} Target Loss: {:.3f}".format(
                epoch,
                batch_idx * batch_size,
                batch_size * (len(train_loader.dataset) // batch_size),
                100.0 * batch_idx / len(train_loader),
                kld_loss / batch_size,
                nll_loss / batch_size,
                y_loss / batch_size,
            )
        )

        # sample = model.sample(32)
        # plt.imshow(sample.to(torch.device("cpu")).numpy())
        # plt.pause(1e-6)

        train_loss += loss.item()

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(test_loader, epoch):
    """uses test data to evaluate
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss, mean_y_loss = 0, 0, 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):

            data = data.to(device)
            labels = labels.to(device)
            data = preprocess_batch(data)

            kld_loss, nll_loss, y_loss, _, _, _ = model(data, labels)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()
            mean_y_loss += y_loss.item()

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)
    mean_y_loss /= len(test_loader.dataset)

    print(
        "====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f}, target loss = {:.4f} ".format(
            mean_kld_loss, mean_nll_loss, mean_y_loss
        )
    )


# changing device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

# manual seed
torch.manual_seed(seed)
plt.ion()

# get dataset
train_dataset, test_dataset = get_dataset(train_config)


# init model + optimizer + datasets  NOTE: could use `prepare_data_loaders`
train_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
    num_workers=train_config.num_workers,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
    num_workers=train_config.num_workers,
)

model = VRNN(
    x_dim=8 * 8,  # chunking 8 time steps so this becomes 64
    h_dim=64,  # hidden recurrent state # conisder increasing to 128
    z_dim=16,  # latent space (could become 16)
    y_dim=20,
    n_layers=1,
)  # how many recurrent layers
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):
    # training + testing
    train(train_loader, epoch)
    test(test_loader, epoch)

    # saving model
    # if epoch % save_every == 1:
    #     fn = "saves/vrnn_state_dict_" + str(epoch) + ".pth"
    #     torch.save(model.state_dict(), fn)
    #     print("Saved model to " + fn)
