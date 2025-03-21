import os
import gc
import h5py
import glob
import hashlib
import numpy as np
import sys


data_path = "/scratch/vl1019/icassp23_data"

def md5checksum(filename):
    with open(filename,"rb") as f:
        bytes = f.read() # read file as bytes
        readable_hash = hashlib.md5(bytes).hexdigest()
    return readable_hash


folds = ["train", "test", "val"]

for fold in folds:
    x_h5 = os.path.join(data_path, "x", "_".join(["icassp23", fold, "audio.h5"]))
    S_h5 = os.path.join(data_path, "S", "_".join(["icassp23", fold, "jtfs.h5"]))
    J_h5 = os.path.join(data_path, "J", "_".join(["icassp23", fold, "grad.h5"]))

    #extract ids from x_h5
    ids = None
    with h5py.File(x_h5, "r") as f:
        ids = list(f["x"].keys())
        theta_dict = dict(f["theta"])
    n_samples = len(ids)
    print("There are " + str(n_samples) + " sounds in " + fold)
    sys.stdout.flush()

    h5_files = [J_h5, S_h5]
    #open S_h5 and J_h5 to write
    for j, content in enumerate(["J", "S"]):
        failed_ids = []

        #initilize h5 files for S and J
        if not os.path.exists(h5_files[j]):
            with h5py.File(h5_files[j], "w") as h5_file:
                audio_group = h5_file.create_group(content)
                shape_group = h5_file.create_group("theta")
            missing_ids = ids
            print("Created " + h5_files[j])
        else:
            with h5py.File(h5_files[j], "r") as h5_file:
                missing_ids = [
                    id for id in ids if id not in h5_file[content].keys()]
                print("Opened " + h5_files[j])
                print("Missing files: {}".format(len(missing_ids)))
        sys.stdout.flush()

        for i in missing_ids:
            if content == "S":
                filename = "_".join(["icassp23", str(i).zfill(6), "jtfs.npy"])
            else:
                filename = "_".join(["icassp23", str(i).zfill(6), "grad", "jtfs.npy"])

            glob_regexp = os.path.join(data_path, content, "*", filename)
            c_files = list(glob.glob(glob_regexp)) #search filename under S folder

            assert len(c_files) <= 2
            assert len(c_files) > 0, "can't find id " + glob_regexp

            if len(c_files) == 2:
                ch0 = md5checksum(c_files[0])
                ch1 = md5checksum(c_files[1])
                if not (ch0 == ch1):
                    print("Mismatch at ID={}".format(i))
                    print("{}, md5={}".format(c_files[0], ch0))
                    print("{}, md5={}".format(c_files[1], ch1))
                    print("")
                    failed_ids.append(i)
                    continue

            c_file = c_files[0]

            #read the file and write it in h5 file
            with h5py.File(h5_files[j], "a") as h5_file:
                c = np.load(c_file) #load content
                h5_file[content][str(i)] = c

        #check if h5 files are maximally filled
        written_ids = None
        with h5py.File(h5_files[j], "r") as f:
            written_ids = list(f[content].keys())
        print("There are ", str(len(written_ids)), " ids in " + content)
        sys.stdout.flush()

        np.save("failed_ids_{}_{}.npy".format(fold, content), np.array(failed_ids))
