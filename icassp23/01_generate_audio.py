"""
This script computes 100k drum sounds by solving a 4th-order partial
differential equation with the pnp_synth.ftm submodule.
FTM stands for Functional Transformation Method.
"""
import datetime
import h5py
import icassp23
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.physical import ftm
import random
import sys
import soundfile as sf
import time

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

for fold in icassp23.FOLDS:
    # Define path to HDF5 file
    fold_df = icassp23.load_fold(fold)
    h5_name = "icassp23_{}_audio.h5".format(fold)
    h5_path = os.path.join(audio_dir, h5_name)

    # Create HDF5 file
    with h5py.File(h5_path, "w") as h5_file:
        audio_group = h5_file.create_group("x")
        shape_group = h5_file.create_group("theta")

        # Loop over shapes
        for i, row in fold_df.iterrows():
            # Physical audio synthesis (g). theta -> x
            theta = np.array([row[column] for column in icassp23.THETA_COLUMNS])
            x = ftm.rectangular_drum(theta, **ftm.constants)

            # Store shape annd waveform into HDF5 container.
            key = str(row["ID"])
            audio_group[key] = x
            theta_group[key] = theta
