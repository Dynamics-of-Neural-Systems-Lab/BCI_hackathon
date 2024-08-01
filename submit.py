import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

from pathlib import Path
from natsort import natsorted
import numpy as np
from safetensors.torch import load_model
import torch 
import pandas as pd

from utils import hvatnet
from utils.creating_dataset import LEFT_TO_RIGHT_HAND


device = 'cuda:0'
dtype = torch.float32

weights = "/msc/home/alopez22/BCI_hackathon/logs/baseline/step_4050_loss_0.2724.safetensors"
"""
MODEL_TYPE = 'hvatnet'
model_config = hvatnet.Config(n_electrodes=8, n_channels_out=20,
                            n_res_blocks=3, n_blocks_per_layer=3,
                            n_filters=128, kernel_size=3,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
model = hvatnet.HVATNetv3(model_config)

load_model(model, weights)

model = model.to(device).to(dtype)
"""


DATA_PATH = Path("/msc/home/alopez22/BCI_hackathon/dataset_v2_blocks")
test_data_name = 'fedya_tropin_standart_elbow_left'  # shoould match `test_dataset_list` used to train the model


data_folder = DATA_PATH / "amputant" / "left" / test_data_name / "preproc_angles" / "train"
all_paths = natsorted(data_folder.glob('*.npz'))
print(f'Found {len(all_paths)} samples in {data_folder}')



#Prediction


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

    print(padded_myo.shape)

    """# some prediction. might be slididng window.
    preds = model.inference(padded_myo)
    preds_downsampled = preds[:gt_len]
    print(f"Completed {i+1}/{len(all_paths)}. Loaded data: {myo.shape} - padded to: {padded_myo.shape} - predictions {preds.shape} - downsampled to: {preds_downsampled.shape}")
    pred_list.append(preds_downsampled)

pred_cat = np.concatenate(pred_list, axis=0)
df = pd.DataFrame(pred_cat)
df.head()"""
