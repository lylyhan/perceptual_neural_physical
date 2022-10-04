import numpy as np
import pandas as pd
import pnp_synth
import random
import sys
import soundfile as sf
import os

save_dir = sys.argv[1]
id_start = int(sys.argv[2])
id_end = int(sys.argv[3])

csv_folder = os.path.join(os.path.dirname(__file__), "data")
folds = ["test", "train", "val"]
full_df = load_dataframe(folds)

params = full_df.values
n_samp = params.shape[0]
assert n_samp >= id_end > id_start >= 0  # nsamp is from 0 to 100k-1?


for fold in folds:
    os.path.makedirs(os.path.join(save_dir, "x", fold), exist_ok=True)


for i in range(id_start, id_end):
    theta = params[i, 3:-1]
    fold = full_df["set"].iloc[i]
    id = full_df["ID"].iloc[i]
    y = pnp_synth.ftm.rectangular_drum(theta, **ftm.constants)
    filename = os.path.join(out_path, fold, str(id) + "_sound.wav")
    sf.write(filename, y, ftm.constants["sr"])
