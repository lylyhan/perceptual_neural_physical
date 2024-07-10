from diffusers import DiffusionPipeline
import torch
import os
import numpy as np

pretrain_dir = "/home/han/data/mersenne24_data/f_W/ddpm_noise-init0"
out_dir = "./samples"

# load pipeline
pipe = DiffusionPipeline.from_pretrained(pretrain_dir, torch_dtype=torch.float16).to("cuda")

noises = pipe(batch_size=2,
        generator=torch.Generator(device='cpu').manual_seed(100),
    ).images

noises = np.array(noises)

test_dir = os.path.join(out_dir, "samples")
os.makedirs(test_dir, exist_ok=True)
np.save("test_samples.npy", noises)
