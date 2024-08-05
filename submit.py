import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

from pathlib import Path
from natsort import natsorted
import numpy as np
from safetensors.torch import load_model
import torch 
import pandas as pd

from utils.creating_dataset import LEFT_TO_RIGHT_HAND, init_dataset, DataConfig
from Model.mamba_net import MambaModel, RawMambaModel, EncodedMamba, Config 

from utils.data_utils import VRHandMYODataset

LEFT_TO_RIGHT_HAND = [6, 5, 4, 3, 2, 1, 0, 7]


#Load the model


#Define model

model_config = Config(n_electrodes=8, n_channels_out=20,
                            n_res_blocks=3, n_blocks_per_layer=3,
                            n_filters=128, kernel_size=3,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
model = MambaModel(model_config)


# Load the model parameters from a file
model_path = "/msc/home/alopez22/BCI_hackathon/logs/mamba_in_standard_loop/step_19200_loss_0.2811.safetensors"  # replace with your file path
load_model(model, model_path)

# Ensure the model is in evaluation mode
model.eval()

#Load the model to the gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



#Load data
data_folder =Path("/msc/home/alopez22/BCI_hackathon/dataset_v2_blocks/amputant/left/fedya_tropin_standart_elbow_left/preproc_angles/submit")
all_paths = natsorted(data_folder.glob('*.npz'))

#Prepare data path

pred_list = []

# loop over each trial
for i, p in enumerate(all_paths):
    # get EMG data 
    sample = np.load(p)
    myo = sample['data_myo']
    myo = myo[:, LEFT_TO_RIGHT_HAND]

    # predictions will have to be downsampled
    gt_len = myo[::8].shape[0]

    # padding
    target_length = (myo.shape[0] + 255) // 256 * 256
    padded_myo = np.pad(myo, ((0, target_length - myo.shape[0]), (0, 0)), mode='constant', constant_values=0)

    # some prediction. might be slididng window.
    preds = model.inference(padded_myo)
    preds_downsampled = preds[:gt_len]
    print(f"Completed {i+1}/{len(all_paths)}. Loaded data: {myo.shape} - padded to: {padded_myo.shape} - predictions {preds.shape} - downsampled to: {preds_downsampled.shape}")
    pred_list.append(preds_downsampled)

pred_cat = np.concatenate(pred_list, axis=0)
df = pd.DataFrame(pred_cat)
print(df.head())

df.insert(0, "sample_id", range(1, 1 + len(df)))

df.to_csv('submit_file_mamba_in_baseline_skeleton.csv', index=False)



