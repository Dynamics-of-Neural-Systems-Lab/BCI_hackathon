import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import wandb

from utils.train import TrainConfig, run_train_model
from utils.augmentations import get_default_transform
from utils import creating_dataset

from Model.mamba_net import MambaModel, Config
from utils.hvatnet import HVATNetv3, Config
import Model.mamba_net
import matplotlib.pyplot as plt
from safetensors.torch import load_file, load_model

import matplotlib
matplotlib.use('Agg')
 


 
#Data charging

DATA_PATH = "/msc/home/alopez22/BCI_hackathon/dataset_v2_blocks"

data_paths = dict(
    datasets=[DATA_PATH],
    hand_type = ['left', 'right'], # [left, 'right']
    human_type = ['health', 'amputant'], # [amputant, 'health']
    test_dataset_list = ['fedya_tropin_standart_elbow_left']  # don't change this !
)

data_config = creating_dataset.DataConfig(**data_paths)
train_dataset, test_dataset = creating_dataset.get_datasets(data_config,)


import random

indices = [random.randint(0, len(train_dataset)-1) for j in range(20)]

for index in indices:

    X, Y = train_dataset[index]

    

    f, axes = plt.subplots(8, 1, figsize=(10, 10), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(X[i], label="True", lw=2, color='k')
        #ax.plot(Y_hat[i], label="Predicted", lw=1, color='r')
    plt.title("EMG of train dataset {}".format(index))

    plt.savefig('/msc/home/alopez22/BCI_hackathon/plots/train_EMG_{}.png'.format(index), dpi=300, bbox_inches='tight')
    plt.close(f)