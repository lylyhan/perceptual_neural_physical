from diffusers import DanceDiffusionPipeline
import torch
import os
import numpy as np
import soundfile as sf

pretrain_dir = "/home/han/data/mersenne24_data/f_W/ddpm_noise-init0"
out_dir = "./samples"

seed = 0 
eval_batch_size = 10

# load pipeline
pipe = DanceDiffusionPipeline.from_pretrained(pretrain_dir, torch_dtype=torch.float16).to("cuda")

audios = pipe(
        batch_size=eval_batch_size,
        generator=torch.Generator(device=pipe.device).manual_seed(seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).audios

test_dir = os.path.join(out_dir)
os.makedirs(test_dir, exist_ok=True)
for i, audio in enumerate(audios):
    sf.write(os.path.join(test_dir, f"test_{i}.wav"), audio.T, pipe.unet.config.sample_rate,)
