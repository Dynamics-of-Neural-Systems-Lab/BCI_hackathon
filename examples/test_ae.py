import numpy as np
import pandas as pd
import torch
import os
import time

from hci_challenge import timeautoencoder as tae
from hci_challenge import preprocessing as dp
from hci_challenge import statistics
from hci_challenge import metrics as mt


#import timediffusion as tdf

#import process_edited as pce



data = 'hurricane'
filename = f'./test_data/{data}.csv'

# Read dataframe
print(filename)
real_df = pd.read_csv(filename)
real_df1 = real_df.drop('date', axis=1).iloc[0:2000,:]
real_df2= real_df.iloc[0:2000,:]

# Pre-processing Data
threshold = 1; column_to_partition = 'Symbol'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#processed_data, time_info = dp.partition_multi_seq(real_df, threshold, column_to_partition);

processed_data = dp.splitData(real_df1, 24, threshold);
time_info = dp.splitTimeData(real_df2, processed_data.shape[1]).to(device)

##############################################################################################################################
# Auto-encoder Training
n_epochs = 50000; eps = 1e-5
weight_decay = 1e-6 ; lr = 2e-4; hidden_size = 200; num_layers = 1; batch_size = 50
channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = 8
lat_dim = 7; seq_col = 'Symbol'
#real_df1 = real_df1.drop(column_to_partition, axis=1)

ds = tae.train_autoencoder(real_df1, processed_data, time_info.to(device), channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
                           batch_size, threshold,  min_beta, max_beta, emb_dim, time_dim, lat_dim, device)

