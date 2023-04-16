import h5py
import os

h5_path = "/gpfswork/rech/rwb/ufg99no/data/icassp23_data/J_modal_nominmax"

for fold in ["train", "test", "val"]:
    h5_fold = os.path.join(h5_path, "ftm_{}_J.h5".format(fold))
    if not os.path.exists(h5_fold):
        with h5py.File(h5_fold, "w") as f:
            J_group = f.create_group("J")
            JdagJ_group = f.create_group("JdagJ")
            M_group = f.create_group("M")
            evals_group = f.create_group("sigma")

    with h5py.File(h5_fold, "a") as f_glob:
        for h5_file in os.listdir(h5_path):
            if fold in h5_file:  
                with h5py.File(h5_file, "r") as f:
                    for group in ['J', 'JdagJ', 'M', 'sigma']:
                        for key in f[group].keys():
                            f_glob[group][key] = f[group][key]
                        