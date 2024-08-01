import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import wandb

from utils.train import TrainConfig, run_train_model
from utils.augmentations import get_default_transform
from utils import creating_dataset

from Model.mamba_net import MambaModel, Config, RawMambaModel, EncodedMamba
import Model.mamba_net

#from utils.hvatnet import HVATNetv3, Config
#import utils.hvatnet


train_config = TrainConfig(exp_name='Embedded_MAMBA', p_augs=0.3, batch_size=64, eval_interval=150, num_workers=0)


DATA_PATH = "/msc/home/alopez22/BCI_hackathon/dataset_v2_blocks"

def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")
    return n_total, n_trainable
    
## Data preparation
transform = get_default_transform(train_config.p_augs)
data_paths = dict(datasets=[DATA_PATH],
                    hand_type = ['left', 'right'], # [left, 'right']
                    human_type = ['health', 'amputant'], # [amputant, 'health']
                    test_dataset_list = ['fedya_tropin_standart_elbow_left'])
data_config = creating_dataset.DataConfig(**data_paths)
train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)



model_config = Config(n_electrodes=8, n_channels_out=20,
                            n_res_blocks=3, n_blocks_per_layer=3,
                            n_filters=128, kernel_size=3,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
#model = MambaModel(model_config)
#model = HVATNetv3(model_config)
model = EncodedMamba(model_config)
count_parameters(model)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_train_model(model, (train_dataset, test_dataset), train_config, device)