import functools
from kymatio.torch import TimeFrequencyScattering1D
import os
import pandas as pd
import pnp_synth
from pnp_synth.neural import forward
from pnp_synth.physical import ftm
import sklearn.preprocessing
import torch

folds = ["train", "test", "val"]

jtfs_params = dict(
    J=13,  # scattering scale ~ 1000 ms
    shape=(2**16,), # input duration ~ 3 seconds
    Q=(12, 1),  # number of filters per octave in time at 1st, 2nd order
    Q_fr=1, # number of fiters per octave in frequency
    F=2,  # local frequential averaging
    max_pad_factor=1,  # temporal padding cannot be greater than 1x support
    max_pad_factor_fr=1,  # frequential padding cannot be greater than 1x support
    pad_mode='zero',
    pad_mode_fr='zero'
)


def load_fold(fold="full"):
    """Load DataFrame."""
    fold_dfs = {}
    csv_folder = os.path.join(os.path.dirname(__file__), "data")
    csv_name = "full_param_log_v2.csv"
    csv_path = os.path.join(csv_folder, csv_name)
    full_df = pd.read_csv(csv_path)
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    if fold == "full":
        return full_df
    else:
        return full_df[full_df["fold"]==fold]


def pnp_forward_factory(scaler):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """
    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list")
    jtfs_operator.average_global = True

    Phi = functools.partial(S_from_x, jtfs_operator=jtfs_operator)

    return functools.partial(
        forward.pnp_forward, Phi=Phi, g=x_from_theta, scaler=scaler
    )


def scale_theta(full_df):
    """
    Take DataFrame, scale training set to [0, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """
    # Fit scaler according to training set only
    train_df = full_df.loc[full_df["set"] == "train"]
    scaler = sklearn.preprocessing.MinMaxScaler()
    train_theta = train_df.values[:, 3:-1]
    scaler.fit(train_theta)

    # Transform whole dataset with scaler
    theta = full_df.values[:, 3:-1]
    nu = scaler.transform(theta)
    return nu, scaler


def S_from_x(x, jtfs_operator):
    "Computes log-compressed Joint-Time Frequency Scattering."
    # Sx is a list of dictionaries
    Sx_list = jtfs_operator(x)

    # Convert to array
    Sx_array = torch.cat([path['coef'].flatten() for path in Sx_list])

    # apply "stable" log transformation
    # the number 1e3 is ad hoc and of the order of 1/mu where mu=1e-3 is the
    # median value of Sx across all paths
    log1p_Sx = torch.log1p(Sx_array*1e3)

    return log1p_Sx


def x_from_theta(theta):
    """Drum synthesizer, based on the Functional Transformation Method (FTM).
    We apply 2**15 samples of zero padding (~1.5 seconds) on the left."""
    x = ftm.rectangular_drum(theta, **ftm.constants)
    x_trimmed = x[:ftm.constants["dur"]//2]
    padding = (ftm.constants["dur"]//2, 0)
    x_padded = torch.nn.functional.pad(
        x_trimmed, padding, mode="constant", value=0)
    return x_padded
