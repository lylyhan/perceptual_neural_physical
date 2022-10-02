"""
This script calculates the JTFS coefficients and the associated Riemannian metric with respect to each normalized parameter.
There will be a vector of 5 associated to each sample.
"""
from kymatio.torch import TimeFrequencyScattering1D
import numpy as np
import os
import pandas as pd
from pnp_synth.physical import ftm
from pnp_synth.perceptual import jtfs
from sklearn.preprocessing import MinMaxScaler
import sys
import torch


csv_path = os.path.expanduser("~/perceptual_neural_physical/data")
save_dir = "/scratch/vl1019/icassp23_data"


def preprocess_gt(full_df):
    # takes dataframe, scale values in dataframe, output dataframe and scaler
    train_df = full_df.loc[full_df["set"] == "train"]
    # normalize
    scaler = MinMaxScaler()
    scaler.fit(train_df.values[:, 3:-1])
    full_df_norm = scaler.transform(
        full_df.values[:, 3:-1]
    )  # just a tensor, not dataframe
    return full_df_norm, scaler


def inverse_scale(y_norm, scaler):
    sc_max = torch.tensor(scaler.data_max_)
    sc_min = torch.tensor(scaler.data_min_)
    y_norm_o = y_norm * (sc_max - sc_min) + sc_min
    return y_norm_o


if __name__ == "__main__":
    out_path_jtfs = sys.argv[1]
    out_path_grad = sys.argv[2]
    id_start = int(sys.argv[3])
    id_end = int(sys.argv[4])

    folds = ["train", "test", "val"]
    fold_dfs = {}
    for fold in folds:
        csv_name = fold + "_param_log_v2.csv"
        csv_path = os.path.join("..", "data", csv_name)
        fold_df = pd.read_csv(csv_path)
        fold_dfs[fold] = fold_df
        # make outpath dirs
        if not os.path.exists(os.path.join(out_path_jtfs, fold)):
            os.makedirs(os.path.join(out_path_jtfs, fold))
        if not os.path.exists(os.path.join(out_path_grad, fold)):
            os.makedirs(os.path.join(out_path_grad, fold))

    full_df = pd.concat(fold_dfs.values()).sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)

    params = full_df.values
    n_samp = params.shape[0]
    assert n_samp > id_end > id_start + 1 > 0  # nsamp is from 0 to 100k-1?

    full_df_norm, scaler = preprocess_gt(full_df)

    jtfs_operator = TimeFrequencyScattering1D(**jtfs.jtfs_params).cuda()

    def cal_jtfs(param_n):
        param_o = inverse_scale(param_n, scaler)
        wav1 = ftm.rectangular_drum(param_o, **ftm.constants)

    torch.autograd.set_detect_anomaly(True)
    for i in range(id_start, id_end):
        param_n = torch.tensor(
            full_df_norm[i, :], requires_grad=True
        )  # where the gradient starts taping
        fold = full_df.values[i, -1]
        id = full_df.values[i, 2]
        raw_jtfs = cal_jtfs(param_n)  # .cpu().detach().numpy()
        # cal grad
        grads = functorch.jacfwd(cal_jtfs)(param_n)  # (639,5)
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
