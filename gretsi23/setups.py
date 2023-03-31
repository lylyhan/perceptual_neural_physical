import functools
#from kymatio.scattering1d.frontend.torch_frontend import TimeFrequencyScatteringTorch as TimeFrequencyScattering1D
#from kymatio.torch import TimeFrequencyScattering as TimeFrequencyScattering1D
from kymatio.torch import TimeFrequencyScattering1D
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.neural import forward
from pnp_synth.physical import amchirp
import sklearn.preprocessing
import torch
import numpy as np
import random 

FOLDS = ["train", "test", "val"]
THETA_COLUMNS = ["f0", "fm", "gamma"]
SAMPLES_PER_EPOCH = 512*50
random.seed(960602)  
logscale = True
synth_type = "amchirp"

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
"""
#kymatio 0.4.0
jtfs_params = dict(
    J=13,
    shape=(2**13*4,),
    Q=(12,1),
    Q_fr=1,
    F=2,
    J_fr=3,
    format='time',
)
"""

def make_dataframe():
    n_steps = 30 #in total half of FTM dataset in seconds, 27k samples
    #middle 14 steps constitute 2744 samples, ~10% of total dataset
    f0_min = 512
    f0_max = 1024
    fm_min = 4 
    fm_max = 16
    gamma_min = 0.5
    gamma_max = 4
    max_test = int(n_steps ** 3 / 10)

    f0s = np.logspace(np.log10(f0_min), np.log10(f0_max), n_steps+1)
    fms = np.logspace(np.log10(fm_min), np.log10(fm_max), n_steps+1)
    gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_steps+1)
    df = {}
    count = 0
    #validation set should be in the middle of sampling space. or centered around in a few scattered spots.
    range_f0 = [np.log10(f0s[8]), np.log10(f0s[-9])]
    range_fm = [np.log10(fms[8]), np.log10(fms[-9])]
    range_gamma = [np.log10(gammas[8]), np.log10(gammas[-9])]

    df['f0'] = {}
    df['fm'] = {}
    df['gamma'] = {}
    df['fold'] = {}
    train_ids =[]
    for n_f0 in range(n_steps):
        for n_fm in range(n_steps):
            for n_gamma in range(n_steps):
                f0 = f0s[n_f0] + random.random() * (f0s[n_f0+1] - f0s[n_f0])
                fm = fms[n_fm] + random.random() * (fms[n_fm+1] - fms[n_fm])
                gamma = gammas[n_gamma] + random.random() * (gammas[n_gamma+1] - gammas[n_gamma])
                df['f0'][count] = np.log10(f0)
                df['fm'][count] = np.log10(fm)
                df['gamma'][count]  = np.log10(gamma)
                if np.log10(f0) <= range_f0[-1] and np.log10(f0) >= range_f0[0]:
                    if np.log10(fm) <= range_fm[-1] and np.log10(fm) >= range_fm[0]:
                        if np.log10(gamma) <= range_gamma[-1] and np.log10(gamma) >= range_gamma[0]:
                            df['fold'][count] = 'val'
                            
                if count not in df['fold'].keys():
                    df['fold'][count] = 'train'
                    train_ids.append(count)
                    #count_train += 1
                count += 1
    #assign parts of train to test
    random.shuffle(train_ids)
    test_ids = train_ids[:max_test]
    df['fold'].update(df['fold'].fromkeys(test_ids,"test"))

    df = pd.DataFrame.from_dict(df)
    df['ID'] = np.arange(0,df.shape[0],1)
   
    print(df[df["fold"]=="train"].shape[0], df[df["fold"]=="test"].shape[0], df[df["fold"]=="val"].shape[0])
    csv_folder = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(csv_folder,exist_ok=True)
    csv_name = "full_param_log.csv"
    
    df.to_csv(os.path.join(csv_folder, csv_name))


     
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


def pnp_forward_factory(scaler):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. a synthesizer g
    3. a JTFS representation Phi
    """
    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list").to("cuda")
    jtfs_operator.average_global = True

    Phi = functools.partial(S_from_x, jtfs_operator=jtfs_operator)

    return functools.partial(
        forward.pnp_forward, Phi=Phi, g=x_from_theta, scaler=scaler
    )


def scale_theta():
    """
    Scale training set to [0, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """
    # Load training set
    train_df = load_fold(fold="train")

    # Fit scaler according to training set only
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    train_theta = np.stack([
        train_df[column].values if logscale else 10**train_df[column].values for column in THETA_COLUMNS
    ], axis=1)
    scaler.fit(train_theta)

    # Load whole dataset
    full_df = load_fold(fold="full")

    # Transform whole dataset with scaler
    theta = np.stack([
        full_df[column].values if logscale else 10**full_df[column].values for column in THETA_COLUMNS
    ], axis=1)
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


def x_from_theta(theta):
    """AM chirp synthesizer."""
    x = amchirp.generate_am_chirp(theta, logscale)
    return x 
