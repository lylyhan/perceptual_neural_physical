"""
This script calculates the JTFS coefficients and the associated LMA steps
with respect to each normalized parameter.
"""
import datetime
import functorch
import functools
import icassp23
import kymatio
import numpy as np
from pnp_synth.physical import ftm
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
id_start = int(sys.argv[2])
id_end = int(sys.argv[3])
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

dir_name = "J_modal_nominmax"

if "nominmax" in dir_name:
    scaler = None
    if "log" in dir_name:
        logscale = True
    else:
        logscale = False
        
    full_df = icassp23.load_fold()
    nus = []
    for column in icassp23.THETA_COLUMNS:
        if not logscale and column in ["omega", "p", "D"]:
            nus.append(10 ** full_df[column].values)
        else:
            nus.append(full_df[column].values)

    nus = np.stack(nus, axis=1)        

else:
    # Rescale shape parameters ("theta") to the interval [0, 1].
    nus, scaler = icassp23.scale_theta() #sorted in terms of id

# Define the forward PNP operator.
S_from_nu = icassp23.pnp_forward_factory(scaler)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
dS_over_dnu = functorch.jacfwd(S_from_nu)

os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)
torch.autograd.set_detect_anomaly(True)
# Make h5 files for M
for fold in icassp23.FOLDS:
    fold_df = icassp23.load_fold(fold)
    h5_name = "ftm_{}_J_{}.h5".format(fold, id_start)
    h5_path = os.path.join(save_dir, dir_name, h5_name)
    if not os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as h5_file:
            J_group = h5_file.create_group("J")
            JdagJ_group = h5_file.create_group("JdagJ")
            M_group = h5_file.create_group("M")
            evals_group = h5_file.create_group("sigma")

fold_df = icassp23.load_fold()
for i in range(id_start, id_end):
    row = fold_df.iloc[i]
    #theta = torch.tensor([row[column] for column in setups.THETA_COLUMNS])
    key = int(row["ID"]) # index in the full dataframe
    fold = row['fold']
    h5_name = "ftm_{}_J_{}.h5".format(fold, id_start)
    h5_path = os.path.join(save_dir, dir_name, h5_name)
    nu = torch.tensor(nus[key, :], requires_grad=True).to("cuda")
    #print(key, i, (nu.detach().numpy() + 1) / 2 * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_, theta)
    #assert nu.detach().numpy() * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_ == theta
    with h5py.File(h5_path, "r") as h5_file:
        if str(i) not in h5_file['sigma'].keys():
            # Compute Jacobian: d(S) / d(nu)
            ismake = True
        else:
            ismake = False
    
    if ismake:
        J = dS_over_dnu(nu).detach()
        M = torch.matmul(J.T, J)
        JdagJ = torch.matmul(torch.inverse(M),J.T)
        assert M.shape[0] == 5 and M.shape[1] == 5

        # Append to HDF5 file
        with h5py.File(h5_path, "a") as h5_file:
            h5_file['J'][str(i)] = J.cpu()
            h5_file['JdagJ'][str(i)] = JdagJ.cpu()
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
