import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from simple_parsing import Serializable
from dataclasses import dataclass


from tqdm import tqdm
#import tqdm.notebook
import random
import numpy as np 

import math

import hci_challenge.preprocessing as dp
import hci_challenge.process_edited as pce

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## randomly assigned anatomicallimits for now....
anatomical_limits = {
    'index_mcp_theta': (-20, 90),
    'index_mcp_fi': (0, 45),
    'index_pip_alpha': (0, 100),
    'index_dip_alpha': (0, 80),
    'middle_mcp_theta': (-20, 90),
    'middle_mcp_fi': (0, 45),
    'middle_pip_alpha': (0, 100),
    'middle_dip_alpha': (0, 80),
    'ring_mcp_theta': (-20, 90),
    'ring_mcp_fi': (0, 45),
    'ring_pip_alpha': (0, 100),
    'ring_dip_alpha': (0, 80),
    'pinky_mcp_theta': (-20, 90),
    'pinky_mcp_fi': (0, 45),
    'pinky_pip_alpha': (0, 100),
    'pinky_dip_alpha': (0, 80),
    'thumb_mcp_theta': (-20, 90),
    'thumb_mcp_fi': (0, 45),
    'thumb_pip_alpha': (0, 80),
    'thumb_dip_alpha': (0, 80)
}


@dataclass
class Config(Serializable):
    n_electrodes: int
    
    n_filters: int
    kernel_size: int
    n_res_blocks: int
    
    n_time: int
    
    hidden_size: int
    num_layers: int
    
    n_transformer_channels: int
    
    latent_dim: int
    
    n_channels_out: int    
    
    emb_dim: int


class TuneModule(nn.Module):
    def __init__(self, n_electrodes=8, temperature=5):
        super(TuneModule, self).__init__()
        """
        - interpolate signal spatially 
        - change amplitude of the signal

        n_electrodes: number of electrodes (default: 8)
        temperture: temperature for softmax of weights (default: 5)
        """
        # spatial rotation.
        self.spatial_weights = torch.nn.Parameter(torch.eye(n_electrodes, n_electrodes), requires_grad=True)
        self.temp = torch.tensor(temperature, requires_grad=False)

        # normalization + amplitude scaling
        self.layer_norm = nn.LayerNorm(n_electrodes, elementwise_affine=True, eps=1e-5)
        
    def forward(self, x):
        """
        x: batch, channel, time
        """

        #x = x.permute(0, 2, 1) # batch, time, channel

        # spatial rotation
        weights = torch.softmax(self.spatial_weights*self.temp , dim=0)
        
        x = torch.matmul(x, weights) # batch, time, channel

        # normalization + amplitude scaling
        x = self.layer_norm(x)

        #x = x.permute(0, 2, 1) # batch, channel, time

        return x

class SimpleResBlock(nn.Module):
    """
    Input is [batch, emb, time]
    Res block.
    In features input and output the same.
    So we can apply this block several times.
    """
    def __init__(self, in_channels, kernel_size):
        super(SimpleResBlock, self).__init__()


        self.conv1 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=kernel_size,
                               bias=True,
                               padding='same')

        self.activation = nn.GELU()

        self.conv2 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=kernel_size,
                               bias=True,
                               padding='same')


    def forward(self, x_input):

        x = self.conv1(x_input)
        x = self.activation(x)
        x = self.conv2(x)

        res = x + x_input

        return res
    
################################################################################################################
def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)

    # Compute the angles for all terms
    #angles = torch.tensor(2**torch.arange(num_terms).float().to(device) * torch.tensor(math.pi).to(device) * v.unsqueeze(-1)).to(device)
    angles = torch.tensor(2**torch.arange(num_terms).float().to(device) * torch.tensor(math.pi).to(device) * v.unsqueeze(-1)).to(device)

    # Compute sine and cosine values for all angles
    sine_values = torch.sin(angles)
    cosine_values = torch.cos(angles)

    # Reshape sine and cosine values for concatenation
    sine_values = sine_values.reshape(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.reshape(*cosine_values.shape[:-2], -1)

    # Concatenate sine and cosine values along the last dimension
    result = torch.cat((sine_values, cosine_values), dim=-1)

    return result

################################################################################################################
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat

################################################################################################################
class Embedding_data(nn.Module):
    def __init__(self, input_size, emb_dim):
        super().__init__()         
       
        self.mlp_nums = nn.Sequential(nn.Linear(16 * input_size, 16 * input_size),  # this should be 16 * n_nums, 16 * n_nums
                                      nn.SiLU(),
                                      nn.Linear(16 * input_size, 16 * input_size))
            
        self.mlp_output = nn.Sequential(nn.Linear(16 * input_size, emb_dim), # this should be 16 * n_nums, 16 * n_nums
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, input_size))
        
    def forward(self, x):        
       
        x_emb = torch.Tensor().to(device)   
        
        x_encoding = compute_sine_cosine(x, num_terms=8)
        
        x_emb = self.mlp_nums(x_encoding)   
        
        final_emb = self.mlp_output(x_emb)
        
        return final_emb

################################################################################################################
def get_torch_trans(heads = 8, layers = 1, channels = 64):
    encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers = layers)

class Transformer_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
        self.feature_layer = get_torch_trans(heads = 8, layers = 1, channels = self.channels)
        self.conv_layer2 = nn.Conv1d(self.channels, 1, 1)
    
    def forward_feature(self, y, base_shape):
        B, channels, L, K = base_shape
        if K == 1:
            return y.squeeze(1)
        y = y.reshape(B, channels, L, K).permute(0, 2, 1, 3).reshape(B*L, channels, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channels, K).permute(0, 2, 1, 3)
        return y
    
    def forward(self, x):
        x = x.unsqueeze(1)
        B, input_channel, K, L = x.shape
        base_shape = x.shape

        x = x.reshape(B, input_channel, K*L)       
        
        conv_x = self.conv_layer1(x).reshape(B, self.channels, K, L)
        x = self.forward_feature(conv_x, conv_x.shape)
        x = self.conv_layer2(x.reshape(B, self.channels, K*L)).squeeze(1).reshape(B, K, L)
        
        return x
           
################################################################################################################
class DeapStack(nn.Module):
    def __init__(self, config): 
        super().__init__()
        
        self.tune_module = TuneModule(n_electrodes=config.n_electrodes, temperature=5.0)
        
        # Change number of features to custom one
        self.spatial_reduce = nn.Conv1d(config.n_electrodes, config.n_filters, kernel_size=1, padding='same')
        self.denoiser = nn.Sequential(*[SimpleResBlock(config.n_filters, config.kernel_size) for _ in range(config.n_res_blocks)])

        
        self.sinusoidal_embedding = Embedding_data(config.n_filters, config.emb_dim)        
        
        self.encoder_transformer = Transformer_Block(config.n_transformer_channels)
        
        self.encoder_mu = nn.GRU(config.n_filters, config.hidden_size, config.num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(config.n_filters, config.hidden_size, config.num_layers, batch_first=True)
        
        self.fc_mu = nn.Linear(config.hidden_size, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_size, config.latent_dim)


        self.decoder_mlp = nn.Sequential(nn.Linear(config.latent_dim, config.hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(config.hidden_size, config.hidden_size))
        
        self.channels = config.n_electrodes

        self.sigmoid = torch.nn.Sigmoid()
        
        self.nums_linear = nn.Linear(config.hidden_size, config.n_channels_out) 
        
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encoder(self, x):
        
        # create sinusoidal embedding of data
        x = self.sinusoidal_embedding(x)
        
        # encode via transformer
        x = self.encoder_transformer(x)
        
        # get mean and variance of variational encoding
        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)
        
        
        mu_z = self.fc_mu(mu_z);
        logvar_z = self.fc_logvar(logvar_z)        
        
        
        emb = self.reparametrize(mu_z, logvar_z)
        
        return emb, mu_z, logvar_z

    def decoder(self, latent_feature):
        decoded_outputs = dict()
        
        latent_feature = self.decoder_mlp(latent_feature)  

        decoded_outputs = self.sigmoid(self.nums_linear(latent_feature))

        return decoded_outputs

    def forward(self, x, targets=None, target_covariance=None):
        
        # smooths data across channels
        x = self.tune_module(x)
        
        # denoising part + expands features
        x = self.spatial_reduce(x.permute(0,2,1)).permute(0,2,1)
        x = self.denoiser(x.permute(0,2,1)).permute(0,2,1)
        
        # encoder
        emb, mu_z, logvar_z = self.encoder(x)
        
        # decoder
        pred = self.decoder(emb)
        
        # downsampling
        pred = pred[:,::8,:]
        
        if targets is None:
            return pred, emb, mu_z, logvar_z
        
        # MAE loss
        loss, l1_loss = loss_function(pred, targets, mu_z, logvar_z, target_covariance)       

        
        return pred, emb, mu_z, logvar_z, loss, l1_loss
        
        #return outputs, emb, mu_z, logvar_z
    

def loss_function(pred, targets, mu_z, logvar_z, target_covariance):

    l1_loss = F.l1_loss(pred, targets)   
    
    loss = l1_loss.clone()
    
    # vae loss
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    
    # combined loss
    beta = 0.1
    loss = loss + beta * loss_kld
    
    #loss = loss + anatomical_loss(pred, anatomical_limits)   
    
    #if target_covariance is not None:
    #    loss = loss + correlation_penalty(pred.permute(2,0,1).reshape(20,-1), target_covariance)
        
    return loss, l1_loss


def anatomical_loss(predicted_angles, anatomical_limits, penalty_weight=1.0):
    #loss = torch.nn.MSELoss()(predicted_angles, target_angles)
    penalty = 0
    for i, (angle_name, (min_val, max_val)) in enumerate(anatomical_limits.items()):
        penalty += torch.sum(torch.clamp(predicted_angles[:, i] - max_val, min=0) ** 2)
        penalty += torch.sum(torch.clamp(min_val - predicted_angles[:, i], min=0) ** 2)
        
    loss = penalty_weight * penalty
    return loss


def correlation_penalty(predicted_angles, target_covariance, penalty_weight=1.0):
    predicted_covariance = torch.cov(predicted_angles)
    penalty = torch.nn.MSELoss()(predicted_covariance, target_covariance)
    return penalty_weight * penalty


