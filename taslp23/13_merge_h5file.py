import h5py
import os
import numpy as np

h5_path = "/gpfswork/rech/aej/ufg99no/data/ftm_jtfs/M_log"

for fold in ["train", "test", "val"]:
    h5_fold = os.path.join(h5_path, "ftm_{}_J.sh5".format(fold))
    print(h5_fold)
    with h5py.File(h5_fold, "w") as f:
        M_group = f.create_group("M")
        evals_group = f.create_group("sigma")

    with h5py.File(h5_fold, "a") as f_glob:
        for h5_file in os.listdir(h5_path):
            if fold in h5_file:  
                with h5py.File(os.path.join(h5_path, h5_file), "r") as f:
                    for group in ['M', 'sigma']:
                        for key in f[group].keys():
                           #print("what's the problem", group, key)
                           if key not in f_glob[group].keys():
                               f_glob[group][key] = np.array(f[group][key])
                        