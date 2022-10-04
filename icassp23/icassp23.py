from kymatio.torch import TimeFrequencyScattering1D
import os
import pandas as pd
import pnp_synth
from pnp_synth.physical import ftm
import torch

folds = ["train", "test", "val"]

jtfs_params = dict(
    J=14,  # scattering scale ~ 1000 ms
    shape=(2**17,),  # 2**16 of zero padding plus 2**16 of signal
    Q=12,  # number of filters per octave
    T=2**13,  # local temporal averaging
    F=2,  # local frequential averaging
    max_pad_factor=1,  # temporal padding cannot be greater than 1x support
    max_pad_factor_fr=1,  # frequential padding cannot be greater than 1x support
    average=True,  # average in time
    average_fr=True,  # average in frequency
)


def load_dataframe():
    "Load DataFrame corresponding to the entire dataset (100k drum sounds)."
    fold_dfs = {}
    csv_folder = os.path.join(os.path.dirname(__file__), "data")
    for fold in folds:
        csv_name = fold + "_param_log_v2.csv"
        csv_path = os.path.join(csv_folder, csv_name)
        fold_df = pd.read_csv(csv_path)
        fold_dfs[fold] = fold_df

    full_df = pd.concat(fold_dfs.values())
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    return full_df


def pnp_forward_factory(scaler):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """
    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params)

    Phi = functools.partial(S_from_theta, jtfs_operator=jtfs_operator)

    return functools.partial(
        pnp_synth.pnp_forward, Phi=S_from_x, g=x_from_theta, scaler=scaler
    )


def scale_theta(full_df):
    """
    Take DataFrame, scale training set to [0, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """
    # Fit scaler according to training set only
    train_df = full_df.loc[full_df["set"] == "train"]
    scaler = MinMaxScaler()
    train_theta = train_df.values[:, 3:-1]
    scaler.fit(train_theta)

    # Transform whole dataset with scaler
    theta = full_df.values[:, 3:-1]
    nu = scaler.transform(theta)
    return nu, scaler


def S_from_x(jtfs_operator, x):
    """
    Computes log-compressed Joint-Time Frequency Scattering.
    """
    # Sx is a tensor with shape (1, n_paths, n_time_frames)
    Sx = jtfs_operator(x)

    # remove leading singleton dimension and unpad
    Sx_unpadded = Sx[0, :, Sx.shape[-1] :]

    # flatten to shape (n_paths * n_time_frames,)
    Sx_flattened = Sx_unpadded.flatten()

    # apply "stable" log transformation
    log1p_Sx = log1p(Sx)

    return log1p_Sx


def x_from_theta(theta):
    """Drum synthesizer, based on the Functional Transformation Method (FTM).
    We apply 2**16 samples of zero padding (~3 seconds) on the left."""
    x = ftm.rectangular_drum(theta, **ftm.constants)
    padding = (ftm.constants["dur"], 0)
    x_padded = torch.nn.functional.pad(x, padding, mode="constant", value=0)
    return x_padded
