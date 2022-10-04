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

jtfs_params = dict(
    J = 14, # scattering scale ~ 1000 ms
    shape = (2**17,), # 2**16 of zero padding plus 2**16 of signal
    Q = 12, # number of filters per octave
    T = 2**13, # local temporal averaging
    F = 2, # local frequential averaging
    max_pad_factor=1, # temporal padding cannot be greater than 1x support
    max_pad_factor_fr=1, # frequential padding cannot be greater than 1x support
    average = True, # average in time
    average_fr = True, # average in frequency
)


def icassp23_synth(theta):
    """Drum synthesizer, based on the Functional Transformation Method (FTM).
    We apply 2**16 samples of zero padding (~3 seconds) on the left."""
    x = pnp_synth.ftm.rectangular_drum(theta, **pnp_synth.ftm.constants)
    padding = (pnp_synth.ftm.constants["dur"], 0)
    x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
    return x_padded


def icassp23_perceptual(Phi, x):
    # Sx is a tensor with shape (1, n_paths, n_time_frames)
    Sx = Phi(x)

    # remove leading singleton dimension and unpad
    Sx_unpadded = Sx[0, :, Sx.shape[-1]:]

    # flatten to shape (n_paths * n_time_frames,)
    Sx_flattened = Sx_unpadded.flatten()

    # apply "stable" log transformation
    log_Sx = log1p(Sx)

    return log_Sx


def icassp23_pnp_forward_factory(scaler, jtfs_params):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """
    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params)

    perceptual_operator = functools.partial(
        icassp23_perceptual, Phi=jtfs_operator)

    return functools.partial(
        pnp_synth.pnp_forward,
        Phi=jtfs_operator,
        g=icassp23_synth,
        scaler=scaler)


# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
out_path_jtfs = sys.argv[1]
out_path_grad = sys.argv[2]
csv_path = sys.argv[3]
id_start = int(sys.argv[4])
id_end = int(sys.argv[5])
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
icassp23_pnp_forward = icassp23_pnp_forward_factory(scaler, jtfs_params)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
icassp23_jacobian = functorch.jacfwd(icassp23_pnp_forward)


torch.autograd.set_detect_anomaly(True)
for i in range(id_start, id_end):
    # Compute forward transformation: nu -> theta -> x -> S
    nu = torch.tensor(nus[i, :], requires_grad=True)
    fold = full_df["fold"].iloc[i]
    id = full_df["ID"].iloc[i]
    S = icassp23_pnp_forward(nu)

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
