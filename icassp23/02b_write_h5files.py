import os
import h5py
import glob
import hashlib
import numpy as np


data_path = "/scratch/vl1019/icassp23_data"

def md5checksum(filename):
    with open(filename,"rb") as f:
        bytes = f.read() # read file as bytes
        readable_hash = hashlib.md5(bytes).hexdigest()
    return readable_hash


folds = ["train", "test", "val"]

for fold in folds:
    x_h5 = os.path.join(data_path, "x", "_".join(["icassp23", fold, "audio.h5"]))
    S_h5 = os.path.join(data_path, "S", "_".join(["icassp23", fold, "audio.h5"]))
    J_h5 = os.path.join(data_path, "J", "_".join(["icassp23", fold, "audio.h5"]))

    #extract ids from x_h5
    ids = None
    with h5py.File(x_h5,"r") as f:
        ids = list(f['x'].keys())
    n_samples = len(ids)
    print("sanity check, there are " + str(n_samples) + " sounds in " + fold)
    h5_files = [J_h5, S_h5]
    #open S_h5 and J_h5 to write
    for j, content in enumerate(["J","S"]):
        for i in ids:
            if content == "S":
                filename = "_".join(["icassp23", str(i).zfill(len(str(n_samples))), "jtfs.npy"])
            else:
                filename = "_".join(["icassp23", str(i).zfill(len(str(n_samples))), "grad", "jtfs.npy"])

            c_files = glob.glob(os.path.join(data_path, content) + '*/' + filename) #search filename under S folder
            assert len(c_files) <= 2 #we are in trouble if duplicates are more than 2
            assert len(c_files) > 0 #we are in trouble if can't find this file
            if len(c_files) == 2:
                assert md5checksum(c_files[0]) == md5checksum(c_files[1]) #check md5checksum
                f1 = np.load(c_files[0])
                f2 = np.load(c_files[1])
                assert np.allclose(f1,f2) #check if contents are the same
                c_file = c_files[0]
            else:
                c_file = c_files[0]

            #initilize h5 files for S and J
            with h5py.File(h5_files[j], "w") as h5_file:
                audio_group = h5_file.create_group(content)

            #read the file and write it in h5 file
            with h5py.File(h5_files[j], "a") as h5_file:
                c = np.load(c_file) #load content
                h5_file[content][str(i)] = c
    
        #check if h5 files are maximally filled
        written_ids = None
        with h5py.File(h5_files[j],"r") as f:
            written_ids = list(f[content].keys())
        print("after writing" + content + ", sanity check, there are " , str(len(written_ids)), " ids in h5 file") 
