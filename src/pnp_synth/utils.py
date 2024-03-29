import functools
from kymatio.torch import TimeFrequencyScattering1D #kymatio 0.3.0
import pandas as pd
from pnp_synth.neural import forward
from pnp_synth.physical import ftm, amchirp
import sklearn.preprocessing
import torch
import numpy as np

#kymatio 0.4.0
#from kymatio.scattering1d.frontend.torch_frontend import TimeFrequencyScatteringTorch as TimeFrequencyScattering1D
#from kymatio.torch import TimeFrequencyScattering as TimeFrequencyScattering1D

folds = ["train", "test", "val"]

def jtfsparam(synth_type):
    if synth_type == "ftm":
        return dict(
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
        return dict(
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

def load_fold(full_df, fold="full"): 
    """Load DataFrame."""
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    if fold == "full":
        return full_df
    else:
        return full_df[full_df["fold"]==fold]


def scale_theta(full_df, out_fold, scaler, logscale, synth_type):
    """
    Scale training set to [-1, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """

    if synth_type == "ftm":
        THETA_COLUMNS = ["omega", "tau", "p", "D", "alpha"]
    elif synth_type == "amchirp":
        THETA_COLUMNS = ["f0", "fm", "gamma"]

    # Load partial dataset
    out_df = load_fold(full_df, out_fold)

    # Transform partial dataset with scaler
    if synth_type == "ftm":
        theta = []
        for column in THETA_COLUMNS:
            if not logscale and column in ["omega", "p", "D"]:
                theta.append(10 ** out_df[column].values)
            else:
                theta.append(out_df[column].values)
        theta = np.stack(theta, axis=1)
    elif synth_type == "amchirp":
        theta = np.stack([
            out_df[column].values if logscale else 10**out_df[column].values for column in THETA_COLUMNS
        ], axis=1) 

    if scaler:
        nus = scaler.transform(theta)
        return nus 
    else:
        return theta

def pnp_forward_factory(scaler, synth_type, logscale):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        #kymatio 0.4.0
        #jtfs_params = dict(
        #    J=13,
        #    shape=(2**13*4,),
        #    Q=(12,1),
        #    Q_fr=1,
        #    F=2,
        #    J_fr=3,
        #    format='time',
        #)


    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list").to(device)
    jtfs_operator.average_global = True

    Phi = functools.partial(S_from_x, jtfs_operator=jtfs_operator)
    g = functools.partial(x_from_theta, synth_type=synth_type, logscale=logscale)
    return functools.partial(
        forward.pnp_forward, Phi=Phi, g=g, scaler=scaler
    )


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


def x_from_theta(theta, synth_type, logscale):
    """Drum synthesizer, based on the Functional Transformation Method (FTM)."""
    if synth_type == "ftm":
        x = ftm.rectangular_drum(theta, logscale=logscale, **ftm.constants)
    elif synth_type == "amchirp":
        x = amchirp.generate_am_chirp(theta, logscale=logscale)
    return x
