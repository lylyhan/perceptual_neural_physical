"""
This script calculates the JTFS coefficients, and gradient(averaged over path) of JTFS coefficient with respect to each normalized parameter.
There will be a vector of 5 associated to each sample.
"""
import numpy as np
import os
import soundfile as sf
from ..pnp.physical import ftm
import pandas as pd
import librosa

from kymatio.torch import TimeFrequencyScattering1D,Scattering1D
import torch
from sklearn.preprocessing import MinMaxScaler
import copy
import math
import functorch

csv_path = "../data"
save_dir = "/home/han/data/"

def preprocess_gt(y_train, y_test, y_val):
    
    param_idx = [0,2,3]
    y_train_cp = copy.deepcopy(y_train)
    y_test_cp = copy.deepcopy(y_test)
    y_val_cp = copy.deepcopy(y_val)
    
    #logscale
    for idx in param_idx:
        y_train_cp[:,idx] = [math.log10(i) for i in y_train_cp[:,idx]]
        y_test_cp[:,idx] = [math.log10(i) for i in y_test_cp[:,idx]]
        y_val_cp[:,idx] = [math.log10(i) for i in y_val_cp[:,idx]]
        
    #normalize
    scaler = MinMaxScaler()
    scaler.fit(y_train_cp)
    y_train_normalized = scaler.transform(y_train_cp)
    y_val_normalized = scaler.transform(y_val_cp)
    y_test_normalized = scaler.transform(y_test_cp)

    return y_train_normalized, y_test_normalized, y_val_normalized, scaler

def inverse_scale(y_norm,scaler):
    sc_max = torch.tensor(scaler.data_max_)
    sc_min = torch.tensor(scaler.data_min_)
    
    param_idx = [0,2,3]
    y_norm_o = y_norm * (sc_max - sc_min) + sc_min
    helper = torch.ones(y_norm_o.shape)
    #inverse logscale
    for idx in param_idx:
        helper[idx] = torch.pow(10,y_norm_o[idx]) / y_norm_o[idx]
    y_norm_o = y_norm_o * helper
    return y_norm_o


if __name__ == "__main__":
    
    #load original parameters and normalize
    df_train = pd.read_csv(os.path.join(csv_path, "train_param_v2.csv"))
    df_test = pd.read_csv(os.path.join(csv_path, "test_param_v2.csv"))
    df_val = pd.read_csv(os.path.join(csv_path, "val_param_v2.csv"))
    y_train = df_train.values[:,1:-1].astype(np.float64)
    y_test = df_test.values[:,1:-1].astype(np.float64)
    y_val = df_val.values[:,1:-1].astype(np.float64)
    y_train_norm, y_test_norm, y_val_norm, scaler = preprocess_gt(y_train, y_test, y_val)
    
    
    jtfs = TimeFrequencyScattering1D(
            J = 14, #scale
            shape = (2**16, ), 
            Q = 1, #filters per octave, frequency resolution
            T = 2**16, 
            F = 2,
            max_pad_factor=1,
            max_pad_factor_fr=1,
            average = True,
            average_fr = True,
        ).cuda()
    
    def cal_jtfs(param_n):
        param_o = inverse_scale(param_n, scaler) 
        wav1 = ftm.getsounds_imp_linear_nonorm_torch(m1,m2,x1,x2,h,param_o[None,:],l0)
        jwav = jtfs(wav1).squeeze()
        return jwav

    m1 = m2 = 10
    x1 = x2 = 0.4
    h = 0.03
    l0 = np.pi
    batchsize = 10
    sets = ["train", "test", "val"]
    for j, param_norm in enumerate([y_train_norm, y_test_norm, y_val_norm]):
        print("making gradients for set ", sets[j])
        set_grad = []
        set_jtfs = []
        for i in range(param_norm.shape[0]): #iterate over each sample in the dataset
            if i%1000 == 0:
                print(i)
            #scale normalized param back to original ranges
            torch.autograd.set_detect_anomaly(True)
            param_n = torch.tensor(param_norm[i,:], requires_grad=True) #where the gradient starts taping
            set_jtfs.append(cal_jtfs(param_n).cpu().detach().numpy())
            grads = functorch.jacfwd(cal_jtfs)(param_n) #(639,5)

            JTJ = torch.matmul(grads.T, grads)
            set_grad.append(JTJ.cpu().detach().numpy())
            torch.cuda.empty_cache()
        set_grad = np.stack(set_grad, axis=0)    
        set_jtfs = np.stack(set_jtfs, axis=0)

        np.save(os.path.join(save_dir,sets[j] + "_grad_jtfs.npy"),set_grad)
        np.save(os.path.join(save_dir,sets[j] + "_jtfs.npy"),set_grad)


