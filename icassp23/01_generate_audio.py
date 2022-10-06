import icassp23
import h5py
import numpy as np
import pandas as pd
import pnp_synth
import sys
import os

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [h5py, np, pd]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

csv_folder = os.path.join(os.path.dirname(__file__), "data")
folds = ["test", "train", "val"]
full_df = icassp23.load_dataframe(folds)

params = full_df.values
n_samp = params.shape[0]
assert n_samp >= id_end > id_start >= 0  # nsamp is from 0 to 100k-1?


for fold in folds:
    os.path.makedirs(os.path.join(save_dir, "x", fold), exist_ok=True)

    fold_df = icassp23.load_dataframe([fold])

    for row in fold_df.iterrows():
        theta = row.values[3:-1]
        id = row["ID"]
        id_str = str(id).zfill(len(str(99999)))
        y = pnp_synth.ftm.rectangular_drum(theta, **ftm.constants)
        filename = os.path.join(out_path, fold, str(id) + "_sound.wav")
        sf.write(filename, y, ftm.constants["sr"])
