"""
Training with VRNN
This file was tangled by /home/moritz/wiki/roam/adopt_vrnn_architecture_for_unsupervised_learning_of_sequential_data_issue_1_dynamics_of_neural_systems_lab_bci_hackathon.org
"""

# common stuff

import os
import sys
sys.path.insert(1, "/home/moritz/Projects/BCI_hackathon/")  # TODO

import torch
import wandb
from tqdm.auto import tqdm

from utils.train import TrainConfig, run_train_model
from utils.augmentations import get_default_transform
from utils import creating_dataset
from pathlib import Path

DATA_PATH = Path("/home/moritz/Projects/BCI_hackathon/data/dataset_v2_blocks")

%load_ext autoreload

%autoreload 2

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
import numpy as np

def emg_fourier_embedding(emg_signal, n_points = 128, sample_rate = 200, min_freq=20, max_freq=150, skip_freqs=4):
    # Assuming `emg_signal` is your EMG signal array with 1000 time points
    n_windows = emg_signal.shape[1] // n_points

    # Calculate frequency resolution
    freq_resolution = sample_rate / n_points

    # Determine the indices of the bins corresponding to 20-150 Hz
    min_freq = 20
    max_freq = 150
    min_bin = int(np.ceil(min_freq / freq_resolution))  # 7
    max_bin = int(np.floor(max_freq / freq_resolution))  # 41

    # Chunk the signal and perform FFT
    fft_results = []
    for i in range(0, n_windows*n_points, n_points):
        window = emg_signal[:, i:i+n_points]
        fft_result = np.fft.fft(window)  # computes over the last axis
        fft_magnitudes = np.abs(fft_result)
        # Select relevant bins
        selected_magnitudes = fft_magnitudes[:, min_bin:max_bin+1:4]
        fft_results.append(selected_magnitudes)

    return torch.tensor(fft_results).view(n_windows, -1).float()
    # Now `fft_results` contains the selected magnitudes for the frequency range of interest for each window

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from dataclasses import dataclass

window_size = 1664
@dataclass
class TrainingConfig:
    image_size = (20, window_size // 8)
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 30
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 2
    save_model_epochs = 3
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = f"{ATT_DIR}/kinematics"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    # hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    # hub_private_repo = False
    # overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from diffusers import UNet2DModel, UNet2DConditionModel

unet = UNet2DConditionModel(  # UNet2DModel  
    sample_size=(20, window_size // 8),
    in_channels=1,
    out_channels=1,
    block_out_channels=(128, 256, 512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
    ),
    # conditional variables
    cross_attention_dim=168,
    # class_embed_type="projection",
    # projection_class_embeddings_input_dim=2184,  # dimensionality of the condition (NOTE depends on window_size
    # class_embeddings_concat=True,  # default
)

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
train_config = TrainConfig(
    exp_name="test_2_run_fedya",
    p_augs=0.3,
    batch_size=16,  # unused
    eval_interval=150,
    num_workers=0,
)
transform = get_default_transform(train_config.p_augs)
data_paths = dict(datasets=[DATA_PATH],
                    hand_type = ['left', 'right'], # [left, 'right']
                    human_type = ['health', 'amputant'], # [amputant, 'health']
                    test_dataset_list = ['fedya_tropin_standart_elbow_left'])
data_config = creating_dataset.DataConfig(**data_paths, window_size=window_size, samples_per_epoch=1000)
train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
num_images = 5
fig, ax = plt.subplots(num_images, 1, figsize=(5, 5))

for i in range(num_images):
    ax[i].imshow(train_dataset[i][1])
    ax[i].set_axis_off()

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from torchvision import transforms
import math

preprocess = transforms.Compose(
    [
        # transforms.Resize(config.image_size),  # should just assert
        transforms.ToTensor(),
        transforms.Normalize([math.pi/4], [math.pi/2]),  # Normalize is important to rescale the pixel values into a [-1, 1] range, which is what the model expects.
    ]
)

preprocessed_train_data = [{"images": preprocess(train_dataset[i][1]),
                      "conditions": emg_fourier_embedding(train_dataset[i][0])}
                      for i in range(len(train_dataset))]

preprocessed_test_data = [{"images": preprocess(test_dataset[i][1]),
                      "conditions": emg_fourier_embedding(test_dataset[i][0])}
                      for i in range(len(test_dataset))]

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
train_dataloader = torch.utils.data.DataLoader(preprocessed_train_data, batch_size=config.train_batch_size, shuffle=True, num_workers=4)  # TODO attention with debugging
test_dataloader = torch.utils.data.DataLoader(preprocessed_test_data, batch_size=config.eval_batch_size, shuffle=False, num_workers=4)

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
sample_image = preprocessed_train_data[0]["images"].unsqueeze(0)
unet(sample_image, encoder_hidden_states=preprocessed_train_data[0]["conditions"].unsqueeze(0), timestep=0).sample.shape

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
import torch
from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=500)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

num_images = 2
fig, ax = plt.subplots(num_images, 1, figsize=(4, 1))

ax[0].imshow(sample_image[0, 0])
ax[0].set_axis_off()

ax[1].imshow(noisy_image[0, 0])
ax[1].set_axis_off()
Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0, :, :, 0])

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"

import torch.nn.functional as F

noise_pred = unet(noisy_image, timesteps, encoder_hidden_states=preprocessed_train_data[0]["conditions"].unsqueeze(0)).sample
loss = F.mse_loss(noise_pred, noise)

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from diffusers.utils import make_image_grid
import os

def evaluate(config, epoch, unet, scheduler, batch_size=16, guidance_scale=1.0):
    test_samples = next(iter(test_dataloader))
    conditions = test_samples["conditions"].to(device)
    sample = torch.randn(
        (batch_size, unet.config.in_channels, unet.config.sample_size[0], unet.config.sample_size[1]),
        device=device
    )
    # sample = sample * scheduler.init_noise_sigma
    for t in tqdm(scheduler.timesteps):
        # 1. predict noise residual
        with torch.no_grad():
            residual = unet(sample, t, encoder_hidden_states=conditions).sample

        # 2. compute less noisy image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

    print("Evaluation MSE loss: ", F.mse_loss(sample, test_samples["images"].to(device)).item())

    # Make a grid out of the images
    # image_grid = make_image_grid(sample, rows=4, cols=batch_size//4)

    # # Save the images
    # test_dir = os.path.join(config.output_dir, "samples")
    # os.makedirs(test_dir, exist_ok=True)
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from pathlib import Path
import os

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            conditions = batch["conditions"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=conditions, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    model.save_pretrained(config.output_dir + f"/epoch_{epoch}")

ATT_DIR="/home/moritz/wiki/data/10/7a0ce5-2537-42da-907e-fb785c61a62f"
from accelerate import notebook_launcher

args = (config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
