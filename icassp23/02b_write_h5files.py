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

            print("i={}, filename={}".format(i, filename))

            glob_regexp = os.path.join(data_path, content, "*", filename)
            c_files = list(glob.glob(glob_regexp)) #search filename under S folder

            print("c_files={}".format(c_files))

            assert len(c_files) <= 2
            assert len(c_files) > 0, "can't find id " + glob_regexp

            if len(c_files) == 2:
                assert md5checksum(c_files[0]) == md5checksum(c_files[1])
                f1 = np.load(c_files[0])
                f2 = np.load(c_files[1])
                assert np.allclose(f1, f2) #check if contents are the same
                c_file = c_files[0]
            else:
                c_file = c_files[0]

            #read the file and write it in h5 file
            print("will open h5_files[j] ")
            with h5py.File(h5_files[j], "a") as h5_file:
                print("file opened")
                c = np.load(c_file) #load content
                print("numpy loaded")
                h5_file[content][str(i)] = c
                print("content written")
                #h5_file["theta"][str(i)] = theta_dict[str(i)]
                #print("shape written")

            print("Done with {}, {}".format(c_file, glob_regexp))

        #check if h5 files are maximally filled
        written_ids = None
        with h5py.File(h5_files[j], "r") as f:
            written_ids = list(f[content].keys())
        print("There are ", str(len(written_ids)), " ids in " + content)
        sys.stdout.flush()
