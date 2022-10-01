import numpy as np
import pandas as pd
import random
import sys
from pnp_synth.physical import ftm 
import soundfile as sf
import os


out_path = sys.argv[1]
csv_folder = sys.argv[2]
id_start = int(sys.argv[3])
id_end = int(sys.argv[4])

folds = ["test", "train", "val"]
fold_dfs = {}

for fold in folds:
    csv_name = fold + "_param_log_v2.csv"
    csv_path = os.path.join(csv_folder, csv_name)
    fold_df = pd.read_csv(csv_path)
    fold_dfs[fold] = fold_df
    #make outpath dirs
    path_out = os.path.join(out_path, fold)
    if not os.path.exists(path_out):
        os.makedirs(path_out)       


full_df = pd.concat(fold_dfs.values()).sort_values(
    by="ID", ignore_index=False
)
assert len(set(full_df["ID"])) == len(full_df)

params = full_df.values
n_samp = params.shape[0]
assert n_samp >= id_end > id_start >= 0 #nsamp is from 0 to 100k-1?


for i in range(id_start, id_end):
    theta = params[i,3:-1]
    fold = params[i,-1]
    id = params[i,2]
    y = ftm.rectangular_drum(theta, **ftm.constants)
    filename = os.path.join(out_path, fold, str(id)+"_sound.wav")
    sf.write(filename, y, ftm.constants["sr"])
