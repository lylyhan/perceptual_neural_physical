"""
This script calculates the JTFS coefficients and the associated Riemannian metric with respect to each normalized parameter.
There will be a vector of 5 associated to each sample.
"""
import datetime
import functorch
import kymatio
from kymatio.torch import TimeFrequencyScattering1D
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.perceptual import jtfs
from pnp_synth.neural import forward, inverse_scale
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

def load_dataframe(folds):
    "Load DataFrame corresponding to the entire dataset (100k drum sounds)."
    fold_dfs = {}
    for fold in folds:
        csv_name = fold + "_param_log_v2.csv"
        csv_path = os.path.join("data", csv_name)
        fold_df = pd.read_csv(csv_path)
        fold_dfs[fold] = fold_df

    full_df = pd.concat(fold_dfs.values())
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    return full_df


def icassp23_synth(theta):
    """Drum synthesizer, based on the Functional Transformation Method (FTM).
    We apply 2**16 samples of zero padding (~3 seconds) on the left."""
    x = pnp_synth.ftm.rectangular_drum(theta, **pnp_synth.ftm.constants)
    padding = (2**16, 0)
    x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
    return x_padded


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
full_df = load_dataframe(csv_path, folds)
params = full_df.values
n_samples = params.shape[0]
assert n_samples > id_end > id_start + 1 > 0  # id is between 0 and (100k-1)

# Rescale shape parameters ("theta") to the interval [0, 1].
nus, scaler = scale_theta(full_df)

# Instantiate Joint-Time Frequency Scattering (JTFS) operator
jtfs_operator = TimeFrequencyScattering1D(**jtfs_params).cuda()

# Define the PNP forward operator, as a composition between
def icassp23_pnp_forward(rescaled_param):
    return forward.pnp_forward(
        Phi=jtfs_operator,
        g=icassp23_synth,
        scaler=scaler,
        rescaled_param=rescaled_param
        )

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
icassp23_jacobian = functorch.jacfwd(icassp23_pnp_forward)

torch.autograd.set_detect_anomaly(True)
for i in range(id_start, id_end):
    param_n = torch.tensor(
        full_df_norm[i, :], requires_grad=True
    )  # where the gradient starts taping
    fold = full_df.values[i, -1]
    id = full_df.values[i, 2]
    raw_jtfs = icassp23_pnp_forward(param_n)
    # cal grad
    grads = icassp23_jacobian(param_n)
    JTJ = torch.matmul(grads.T, grads)
    torch.cuda.empty_cache()
    np.save(
        os.path.join(out_path_jtfs, fold, id + "_jtfs.npy"),
        raw_jtfs.cpu().detach().numpy(),
    )
    np.save(
        os.path.join(out_path_grad, fold, id + "_grad_jtfs.npy"),
        JTJ.cpu().detach().numpy(),
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
