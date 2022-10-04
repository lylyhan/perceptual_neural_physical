"""
This script calculates the JTFS coefficients and the associated Riemannian
metric with respect to each normalized parameter.
"""
import datetime
import functorch
import functools
import icassp23
import kymatio
from kymatio.torch import TimeFrequencyScattering1D
import numpy as np
import os
import pandas as pd
import pnp_synth
import sklearn
import sys
import time
import torch

save_dir = "/scratch/vl1019/icassp23_data"
folds = ["train", "test", "val"]

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
out_path_jtfs = sys.argv[1]
out_path_grad = sys.argv[2]
id_start = int(sys.argv[3])
id_end = int(sys.argv[4])
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")

# Create folders
for fold in folds:
    os.makedirs(os.path.join(out_path_jtfs, fold), exist_ok=True)
    os.makedirs(os.path.join(out_path_grad, fold), exist_ok=True)

# Load DataFrame
full_df = icassp23.load_dataframe(csv_path, folds)
params = full_df.values
n_samples = params.shape[0]
assert n_samples > id_end > id_start + 1 > 0  # id is between 0 and (100k-1)

# Rescale shape parameters ("theta") to the interval [0, 1].
nus, scaler = scale_theta(full_df)

# Define the forward PNP operator.
S_from_nu = icassp23.pnp_forward_factory(scaler, jtfs_params)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
nabla = functorch.jacfwd(pnp_forward)


torch.autograd.set_detect_anomaly(True)
for i in range(id_start, id_end):
    # Compute forward transformation: nu -> theta -> x -> S
    nu = torch.tensor(nus[i, :], requires_grad=True)
    fold = full_df["fold"].iloc[i]
    id = full_df["ID"].iloc[i]
    S = S_from_nu(nu)

    # Compute Jacobian: d(S) / d(nu)
    jacobian = icassp23_jacobian(nu)

    #torch.cuda.empty_cache()
    np.save(
        os.path.join(out_path_jtfs, fold, id + "_jtfs.npy"),
        S.cpu().detach().numpy(),
    )
    np.save(
        os.path.join(out_path_grad, fold, id + "_grad_jtfs.npy"),
        jacobian.cpu().detach().numpy(),
    )
    print(datetime.datetime.now() + " Exported: {}/{}".format(fold, id))
print("")

# Print elapsed time.
print(str(datetime.datetime.now()) + " Success.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(elapsed_hours,
                                               elapsed_minutes,
                                               elapsed_seconds)
print("Total elapsed time: " + elapsed_str + ".")
