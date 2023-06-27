import functools
from kymatio.torch import TimeFrequencyScattering1D
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.neural import forward, loss
import torch.nn as nn
from pnp_synth.physical import ftm
import sklearn.preprocessing
import torch
import numpy as np

FOLDS = ["train", "test", "val"]
THETA_COLUMNS = ["omega", "tau", "p", "D", "alpha"]
SAMPLES_PER_EPOCH = 512*50

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

mss_param = dict(
    max_n_fft=2048,
    num_scales=6,
    hop_lengths=None,
    mag_w=1.0,
    logmag_w=0.0,
    p=1.0,
)

def load_fold(fold="full"):
    """Load DataFrame."""
    fold_dfs = {}
    csv_folder = os.path.join(os.path.dirname(__file__), "data")
    csv_name = "full_param_log.csv"
    csv_path = os.path.join(csv_folder, csv_name)
    full_df = pd.read_csv(csv_path)
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    if fold == "full":
        return full_df
    else:
        return full_df[full_df["fold"]==fold]


def pnp_forward_factory(scaler, logscale, synth_type):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """

    if synth_type == "ftm":
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
    elif synth_type == "amchirp":
        jtfs_params = dict(
            J=13,  # scattering scale ~ 1000 ms
            shape=(2**13*4,), # input duration ~ 3 seconds
            Q=(12, 1),  # number of filters per octave in time at 1st, 2nd order
            Q_fr=1, # number of fiters per octave in frequency
            F=2,  # local frequential averaging
            max_pad_factor=1,  # temporal padding cannot be greater than 1x support
            max_pad_factor_fr=1,  # frequential padding cannot be greater than 1x support
            pad_mode='zero',
            pad_mode_fr='zero'
        )
    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list")
    jtfs_operator.average_global = True

    Phi = functools.partial(S_from_x, jtfs_operator=jtfs_operator)

    g = functools.partial(x_from_theta, logscale=logscale)
    return functools.partial(
        forward.pnp_forward, Phi=Phi, g=g, scaler=scaler
    )


def scale_theta(logscale):
    """
    Scale training set to [-1, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """
    # Load training set
    train_df = load_fold(fold="train")

    # Fit scaler according to training set only
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    train_theta = []
    for column in THETA_COLUMNS:
        if not logscale and column in ["omega", "p", "D"]:
            train_theta.append(10 ** train_df[column].values)
        else:
            train_theta.append(train_df[column].values)
    train_theta = np.stack(train_theta, axis=1)
    #train_theta = np.stack([
    #    10 ** train_df[column].values for column in THETA_COLUMNS if not logscale and column in ["omega", "p", "D"] else train_df[column].values
    #    ], axis=1)
    scaler.fit(train_theta)

    # Load whole dataset
    full_df = load_fold(fold="full")

    # Transform whole dataset with scaler
    theta = []
    for column in THETA_COLUMNS:
        if not logscale and column in ["omega", "p", "D"]:
            theta.append(10 ** full_df[column].values)
        else:
            theta.append(full_df[column].values)
    theta = np.stack(theta, axis=1)
    #theta = np.stack([
    #    full_df[column].values for column in THETA_COLUMNS
    #], axis=1)
    nus = scaler.transform(theta)
    return nus, scaler


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


def x_from_theta(theta, logscale):
    """Drum synthesizer, based on the Functional Transformation Method (FTM)."""
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants)
    return x

def pnp_forward_factory_mss(scaler, logscale):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """

    g = functools.partial(x_from_theta, logscale=logscale)
    return functools.partial(
        forward.pnp_forward, Phi=MultiScaleSpectralLoss(), g=g, scaler=scaler
    )

class MultiScaleSpectralLoss(nn.Module):
    def __init__(
        self,
        max_n_fft=2048,
        num_scales=6,
        hop_lengths=None,
        mag_w=1.0,
        logmag_w=0.0,
    ):
        super().__init__()
        assert max_n_fft // 2 ** (num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = (
            [n // 4 for n in self.n_ffts] if not hop_lengths else hop_lengths
        )
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        self.ops = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i])
            for i, n_fft in enumerate(self.n_ffts)
        ]

    def forward(self, x):
        S = []
        for op in self.ops:
            S.append(op(x).flatten())
            print(op(x).shape)
        S = torch.cat(S)
        print("you passeed here?", S.shape)
        return S

class MagnitudeSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).type_as(x),
            return_complex=True,
        ).abs()
