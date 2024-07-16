import numpy as np
import pandas as pd
import torch
import os
import time

from hci_challenge import timeautoencoder as tae
from hci_challenge import preprocessing as dp
from hci_challenge import statistics
from hci_challenge import metrics as mt
from hci_challenge.hci_utils.train import TrainConfig, run_train_model


from hci_challenge.hci_utils.augmentations import get_default_transform
from hci_challenge.hci_utils import creating_dataset

# this is the implementation of the custom baseline model
from hci_challenge.hci_utils import hvatnet

from hci_challenge.hci_utils.augmentations import get_default_transform
from hci_challenge.hci_utils import creating_dataset
from hci_challenge.hci_utils.hand_visualize import Hand, save_animation
from hci_challenge.hci_utils.quats_and_angles import get_quats


DATA_PATH = r"./dataset_v2_blocks/"

def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")
    return n_total, n_trainable


data_paths = dict(
    datasets=[DATA_PATH],
    hand_type = ['left', 'right'], # [left, 'right']
    human_type = ['health', 'amputant'], # [amputant, 'health']
    test_dataset_list = ['fedya_tropin_standart_elbow_left']  # don't change this !
)

train_config = TrainConfig(exp_name='test_2_run_fedya', p_augs=0.3, batch_size=64, eval_interval=5,
                           num_workers=0,)

# define a config object to keep track of data variables
data_config = creating_dataset.DataConfig(**data_paths)

# get transforms
p_transform = 0.1  # probability of applying the transform
transform = get_default_transform(p_transform)

# load the data
train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)


print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

X, Y = train_dataset[0]
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

model_config = hvatnet.Config(n_electrodes=8, n_channels_out=20,
                            n_res_blocks=3, n_blocks_per_layer=3,
                             kernel_size=3, n_filters=64,#128,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
model = hvatnet.HVATNetv3(model_config)
count_parameters(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_train_model(model, (train_dataset, test_dataset), train_config, device)



batches = [train_dataset[i] for i in range(10)]
Y = np.concatenate([b[1] for b in batches], axis=1)
quats = get_quats(Y)

hand_gt = Hand(quats)
ani = hand_gt.visualize_all_frames()
save_animation(ani, 'test_vis.gif', fps=25,)   # this will save a .gif file