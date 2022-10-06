"""
This script calculates the JTFS coefficients and the associated Riemannian
metric with respect to each normalized parameter.
"""
import datetime
import functorch
import functools
import icassp23
import kymatio
import numpy as np
import os
import pandas as pd
import pnp_synth
import sklearn
import sys
import time
import torch

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
id_start = int(sys.argv[2])
id_end = int(sys.argv[3])
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Create folders
for fold in icassp23.folds:
    os.makedirs(os.path.join(save_dir, "S", fold), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "J", fold), exist_ok=True)

# Load DataFrame
full_df = icassp23.load_fold()
params = full_df.values
n_samples = params.shape[0]
assert n_samples > id_end > id_start >= 0  # id is between 0 and (100k-1)

# Rescale shape parameters ("theta") to the interval [0, 1].
nus, scaler = icassp23.scale_theta()

# Define the forward PNP operator.
S_from_nu = icassp23.pnp_forward_factory(scaler)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
dS_over_dnu = functorch.jacfwd(S_from_nu)

# Loop over examples.
torch.autograd.set_detect_anomaly(True)
for i in range(id_start, id_end):
    # Compute forward transformation: nu -> theta -> x -> S
    nu = torch.tensor(nus[i, :], requires_grad=True)
    fold = full_df["set"].iloc[i]
    assert i == full_df["ID"].iloc[i]
    S = S_from_nu(nu)

    # Compute Jacobian: d(S) / d(nu)
    J = dS_over_dnu(nu)

    # Convert to NumPy array and save to disk
    i_prefix = "icassp23_" + str(i).zfill(len(str(n_samples)))
    S_path = os.path.join(save_dir, "S", fold, i_prefix + "_jtfs.npy")
    np.save(S_path, S.detach().numpy())
    J_path = os.path.join(save_dir, "J", fold, i_prefix + "_grad_jtfs.npy")
    np.save(J_path, J.detach().numpy())

    # Print
    print(str(datetime.datetime.now()) + " Exported: {}/{}".format(fold, i_str))
    sys.stdout.flush()
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
