"""
This script calculates the JTFS coefficients and the associated Jacobian
with respect to each normalized parameter.
"""
import datetime
import functorch
import functools
import setups
import kymatio
import numpy as np
from pnp_synth.physical import amchirp
import os
import pandas as pd
import pnp_synth
import h5py
import sklearn
import sys
import time
import torch

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

dir_name = "M_nominmax"
if "nominmax" in dir_name:
    scaler = None
    logscale = False
    full_df = setups.load_fold()
    nus = np.stack([
        full_df[column].values if logscale else 10**full_df[column].values for column in setups.THETA_COLUMNS
    ], axis=1)
else:
    # Rescale shape parameters ("theta") to the interval [0, 1].
    nus, scaler = setups.scale_theta() #sorted in terms of id

# Define the forward PNP operator.
S_from_nu = setups.pnp_forward_factory(scaler)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
dS_over_dnu = functorch.jacfwd(S_from_nu)


os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)
torch.autograd.set_detect_anomaly(True)
# Make h5 files for M
for fold in setups.FOLDS:
    fold_df = setups.load_fold(fold)
    h5_name = "amchirp_{}_M.h5".format(fold)
    h5_path = os.path.join(save_dir, dir_name, h5_name)
    with h5py.File(h5_path, "w") as h5_file:
        M_group = h5_file.create_group("M")
        evals_group = h5_file.create_group("sigma")

    # Define row iterator
    row_iter = fold_df.iterrows()
    # Loop over batches.
    batch_size = len(fold_df)
    n_batches = 1 + len(fold_df) // batch_size

    for i, row in row_iter: # sorted in terms of each fold
        #theta = torch.tensor([row[column] for column in setups.THETA_COLUMNS])
        key = int(row["ID"]) # index in the full dataframe
        nu = torch.tensor(nus[key, :], requires_grad=True).to("cuda")
        #print(key, i, (nu.detach().numpy() + 1) / 2 * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_, theta)
        #assert nu.detach().numpy() * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_ == theta

        # Compute Jacobian: d(S) / d(nu)
        J = dS_over_dnu(nu).detach()
        M = torch.matmul(J.T, J)
        assert M.shape[0] == 3 and M.shape[1] == 3

        # Append to HDF5 file
        with h5py.File(h5_path, "a") as h5_file:
            h5_file['M'][str(i)] = M.cpu()
            h5_file['sigma'][str(i)] = torch.linalg.eigvals(M).cpu()

    # Print
    now = str(datetime.datetime.now())
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
