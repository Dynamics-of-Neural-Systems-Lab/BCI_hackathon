import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, load_model


import wandb

from utils.train import TrainConfig, run_train_model
from utils.augmentations import get_default_transform
from utils import creating_dataset

from Model.mamba_net import MambaModel, Config
from utils.hvatnet import HVATNetv3
import Model.mamba_net
import matplotlib.pyplot as plt

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
test_dataset = creating_dataset.get_datasets(data_config, only_test = True)


#Define model

model_config = Config(n_electrodes=8, n_channels_out=20,
                            n_res_blocks=3, n_blocks_per_layer=3,
                            n_filters=128, kernel_size=3,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
model = HVATNetv3(model_config)
#model = MambaModel(model_config)

#Test


# Load the model parameters from a file
model_path = "/msc/home/alopez22/BCI_hackathon/logs/baseline/step_4050_loss_0.2724.safetensors"  # replace with your file path
load_model(model, model_path)

# Ensure the model is in evaluation mode
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device =  torch.device("cpu")
model.to(device)

for trial in test_dataset:
    X, Y = test_dataset[0]

    # Convert the input data to a tensor and move it to the appropriate device
    X_tensor = torch.tensor(X.T).to(device).permute(1,0)

    print(X_tensor.shape)

    # Run the inference
    with torch.no_grad():  # Disable gradient calculation for inference
        Y_hat = model(X_tensor.unsqueeze(0)).squeeze().detach().cpu().numpy().T


    print(F.mse_loss(pred, targets))

"""
f, axes = plt.subplots(20, 1, figsize=(10, 10), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(Y[i], label="True", lw=2, color='k')
    ax.plot(Y_hat[i], label="Predicted", lw=1, color='r')
    ax.legend()

plt.savefig('/msc/home/alopez22/BCI_hackathon/mamba_embedded.png', dpi=300, bbox_inches='tight')
plt.close(f)
"""