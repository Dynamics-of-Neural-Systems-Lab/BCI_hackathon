
import sys

from hci_challenge.hci_utils.train import TrainConfig
from hci_challenge.hci_utils.augmentations import get_default_transform
from hci_challenge.hci_utils import creating_dataset

from hci_challenge.train import run_train_model
from hci_challenge.autoencoder import DeapStack, Config

import torch

def main():
    
    train_config = TrainConfig(exp_name='test_1', p_augs=0.3, batch_size=64, eval_interval=5, num_workers=0)
    
    def count_parameters(model): 
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")
        return n_total, n_trainable
    
    DATA_PATH = r"./dataset_v2_blocks/"
    
    
    data_paths = dict(
        datasets=[DATA_PATH],
        hand_type = ['left', 'right'], # [left, 'right']
        human_type = ['health', 'amputant'], # [amputant, 'health']
        test_dataset_list = ['fedya_tropin_standart_elbow_left']  # don't change this !
    )
    
    # define a config object to keep track of data variables
    data_config = creating_dataset.DataConfig(**data_paths)
    
    # get transforms
    transform = get_default_transform(train_config.p_augs)
    
    
    # load the data
    train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)
    
    
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    
    X, Y = train_dataset[0]
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

   
    model_config = Config(n_electrodes=8,
                          n_channels_out=20,
                          n_res_blocks=3, 
                          n_filters=64, 
                          kernel_size=3,
                          n_time=256,
                          hidden_size=100,
                          num_layers=1,
                          n_transformer_channels=64,
                          latent_dim=7,
                          emb_dim=64,
                          )
    
    
    model = DeapStack(model_config)
    count_parameters(model)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_train_model(model,  (train_dataset, test_dataset), train_config, device=device)
    


if __name__ == "__main__":
    sys.exit(main())






# optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

# #inputs = processed_data.to(device)
    
# losses = []
# lambd = 0.7
# beta = max_beta
# best_train_loss = float('inf')
# all_indices = list(range(len(train_dataset)))

# for epoch in tqdm(range(n_epochs)):
#     ######################### Train Auto-Encoder #########################
#     batch_indices = random.sample(all_indices, batch_size)

#     inputs = [train_dataset[u][0].T for u in batch_indices]
#     expected_outputs = [train_dataset[u][1].T for u in batch_indices] #[train_dataset[u][1].T for u in batch_indices]

#     inputs = torch.Tensor(inputs).to(device)
#     expected_outputs = torch.Tensor(expected_outputs).to(device)

#     optimizer_ae.zero_grad()
#     outputs, _, mu_z, logvar_z = model(inputs)   
    
#     outputs = outputs[:,::data_config.down_sample_target,:]
    
#     num_loss = F.l1_loss(outputs, expected_outputs)
#     #num_loss = F.mse_loss(outputs, expected_outputs)
    
#     temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
#     loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

#     num_loss_Auto = num_loss + beta * loss_kld
#     num_loss_Auto.backward()
#     optimizer_ae.step()
    
    
    
#     print('Train Loss: {:5f}'.format(num_loss.item()))
#     print('Val Loss: {:5f}'.format(num_loss.item()))



#%%


# batches = [train_dataset[i] for i in range(10)]
# Y = np.concatenate([b[1] for b in batches], axis=1)
# quats = get_quats(Y)

# hand_gt = Hand(quats)
# ani = hand_gt.visualize_all_frames()
# save_animation(ani, 'test_vis.gif', fps=25,)   # this will save a .gif file
