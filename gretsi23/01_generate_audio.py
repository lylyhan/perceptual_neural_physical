"""
This script computes 27k am chirp sounds with the pnp_synth.amchirp submodule.
"""
import datetime
import h5py
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.physical import amchirp
import setups
import random
import sys
import soundfile as sf
import time
import torch

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [h5py, np, pd]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Create directory for audio files.
audio_dir = os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)

for fold in setups.FOLDS:
    # Define path to HDF5 file
    fold_df = setups.load_fold(fold)
    h5_name = "amchirp_{}_audio.h5".format(fold)
    h5_path = os.path.join(audio_dir, h5_name)

    # Create HDF5 file
    with h5py.File(h5_path, "w") as h5_file:
        audio_group = h5_file.create_group("x")
        shape_group = h5_file.create_group("theta")

    # Define row iterator
    row_iter = fold_df.iterrows()

    # Loop over batches.
    batch_size = len(fold_df)
    n_batches = 1 + len(fold_df) // batch_size

    for i, row in row_iter:
        #i, row = irow

        # Physical audio synthesis (g). theta -> x
        theta = torch.tensor([row[column] for column in setups.THETA_COLUMNS]).to("cuda")
        x = amchirp.generate_am_chirp(theta)
        key = str(row["ID"])

        # Append to HDF5 file
        with h5py.File(h5_path, "a") as h5_file:
            # Store shape annd waveform into HDF5 container.
            h5_file["x"][key] = x.cpu()
            h5_file["theta"][key] = theta.cpu()

    # Print
    now = str(datetime.datetime.now())
    #batch_str = str(batch_id).zfill(len(str(n_batches)))
    #print(now + " Exported: {}, batch {}".format(fold, batch_str))
    sys.stdout.flush()

    # Empty line between folds
    print("")


# Print elapsed time.
print(str(datetime.datetime.now()) + " Success.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.0
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
    elapsed_hours, elapsed_minutes, elapsed_seconds
)
print("Total elapsed time: " + elapsed_str + ".")
