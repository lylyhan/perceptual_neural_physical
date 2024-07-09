from diffusers import DDPMScheduler, UNet1DModel, DDPMPipeline
import torch
import h5py
import mersenne24
from dataclasses import dataclass
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torch.nn.functional as F
import IPython.display as ipd
import os
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path



@dataclass
class TrainingConfig:
    audio_len = 2**14  # the generated audio length
    train_batch_size = 1
    eval_batch_size = 2  # how many images to sample during evaluation
    num_epochs = 10
    sr = 22050
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    #output_dir = "/home/han/data/mersenne24_data/f_W/ddpm_noise-init0/"  # the model name locally and on the HF Hub
    output_dir = "/gpfswork/rech/aej/ufg99no/data/mersenne24_data/f_W/ddpm_noise-init0/"
    seed = 0

class NoiseData(Dataset):
    def __init__(self,
                 noise_dir, #path to audio hdf files
                 audio_len, 
                 mode, 
                 sr
                 ): #path to noise audio hdf files
        super().__init__()

        self.noise_dir = noise_dir #path to hdf5 file
        self.audio_len = audio_len
        self.mode = mode
        self.noise_inventory()
        #make list of noise pitches
        self.sr = sr
   
    def __getitem__(self, idx): 
        id = self.ids[idx]
        y = self.y_from_id(id, idx)
        
        return {'noise': y}
    
    def noise_inventory(self):
        with h5py.File(self.noise_dir, "r") as f: # these are synth sounds
            self.N_noise1 =len(f['noise'].keys())
            self.ids1 = list(f['noise'].keys())
            
        with h5py.File(self.noise_dir[:-3]+"_nonval.h5", "r") as f:
            self.N_noise2 = len(f['noise'].keys())
            self.ids2 = list(f['noise'].keys())
        self.ids = self.ids1
        self.ids.extend(self.ids2)
        N_all = len(self.ids)
        if self.mode == "train":
            self.ids = self.ids[:round(N_all*0.8)]
        elif self.mode == "eval":
            self.ids = self.ids[round(N_all*0.8):]


    def __len__(self):
        return len(self.ids)

    def y_from_id(self, id, idx):
        
        if int(idx) < self.N_noise1:
            noise_dir = self.noise_dir
        else:
            noise_dir = self.noise_dir[:-3]+"_nonval.h5"
        
        with h5py.File(noise_dir, "r") as f: # these are synth sounds
            x = np.array(f['noise'][str(id)])
            sr = np.array(f['sr'][str(id)])
        
        if sr != self.sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.sr)
        if len(x) < self.audio_len:
            x = np.concatenate([x, np.zeros(self.audio_len-len(x))])
        else:
            x = x[:self.audio_len]
       
        x = torch.tensor(x, dtype=torch.float32).cuda().unsqueeze(0)
        return x


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
        print("training epoch, ", epoch)
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            target_audio = batch["noise"]
            # Sample noise to add to the images
            noise = torch.randn(target_audio.shape, device=target_audio.device)
            bs = target_audio.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=target_audio.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_audio = noise_scheduler.add_noise(target_audio, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_audio, timesteps, return_dict=False)[0]
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
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)




if __name__ == "__main__":
   

    config = TrainingConfig()

    model = UNet1DModel(
        sample_size=config.audio_len,  # the target audio length
        sample_rate=config.sr,
        in_channels=1,  # the number of input channels, 
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock1D",  # a regular ResNet downsampling block
            "DownBlock1D",
            "DownBlock1D",
            "DownBlock1D",
            "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock1D",
        ),
        up_block_types=(
            "UpBlock1D",  # a regular ResNet upsampling block
            "AttnUpBlock1D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock1D",
            "UpBlock1D",
            "UpBlock1D",
            "UpBlock1D",
        ),
    )


    #noise_dir = "/home/han/data/mersenne24_data/x/mersenne24_realaudio.h5"
    noise_dir = "/gpfswork/rech/aej/ufg99no/data/mersenne24_data/x/mersenne24_realaudio.h5"
    dataset = NoiseData(noise_dir, mode="train", audio_len=2**17, sr=22050)
    train_dataloader = DataLoader(dataset)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=config.lr_warmup_steps,
                    num_training_steps=(len(train_dataloader) * config.num_epochs),
                )
    
    # try adding noise to the audio according to the noise schedule
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)








