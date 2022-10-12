"""
This script computes the Riemannian metric M(theta) = (J(theta)^T J(theta))
associated to the Jacobian of (Phi o g) at theta, and so for every theta_n
in the dataset.
"""
import os
import time
import sys
import datetime
import numpy as np
import pandas as pd
import icassp23
import h5py

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
id_start = int(sys.argv[2])
id_end = int(sys.argv[3])
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [np, pd]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Create directory for audio files.
audio_dir = os.path.join(save_dir, "M")
os.makedirs(audio_dir, exist_ok=True)

# Load DataFrame
full_df = icassp23.load_fold("full")
params = full_df.values
n_samples = params.shape[0]

# Make h5 files for M
for fold in icassp23.FOLDS:
    fold_df = icassp23.load_fold(fold)
    h5_name = "icassp23_{}_M.h5".format(fold)
    h5_path = os.path.join(audio_dir, h5_name)
    with h5py.File(h5_path, "w") as h5_file:
        M_group = h5_file.create_group("M")
        evals_group = h5_file.create_group("sigma")


for i in range(id_start, id_end):
     # Convert to NumPy array and save to disk
    fold = full_df["fold"].iloc[i]
    assert i == full_df["ID"].iloc[i]
    i_prefix = "icassp23_" + str(i).zfill(len(str(n_samples)))
    J_path = os.path.join(save_dir, "J", i_prefix + "_grad_jtfs.npy")
    J = np.load(J_path)
    M = np.matmul(J.T, J)
    assert M.shape[0] == 5 and M.shape[1] == 5
    h5_path = os.path.join(audio_dir, "icassp23_{}_M.h5".format(fold))
    with h5py.File(h5_path, "a") as h5_file:
        h5_file['M'][str(i)] = M
        h5_file['sigma'][str(i)] = np.linalg.eigvals(M)


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
