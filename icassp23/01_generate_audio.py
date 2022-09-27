import numpy as np
import pandas
import random
import sys

from ..pnp.physical import ftm  #not sure if this will work
import soundfile as sf
import os

#define dataset path
out_path = "/home/han/data/drum_data/"


df_train = pandas.read_csv("../data/train_param_v2.csv") 
df_val = pandas.read_csv("../data/val_param_v2.csv") 
df_test = pandas.read_csv("../data/test_param_v2.csv") 

print("finish loading annotations!")

val_params = df_val.values
train_params = df_train.values
test_params = df_test.values

n_samp_val,n_param_val = val_params.shape
n_samp_test,n_param_val = test_params.shape
n_samp_train,n_param_val = train_params.shape

x1 = 0.4
x2 = 0.4
h = 0.03
l0 = np.pi

mode = 10
sr = 22050

print("begin making dataset!")

path_out = os.path.join(out_path,"val")

if not os.path.exists(path_out):
    os.makedirs(path_out)       
for i in range(n_samp_val):
    omega,tau,p,D,alpha = val_params[i,1:-1]
    y = ftm.getsounds_imp_linear_nonorm(mode,mode,x1,x2,h,tau,omega,p,D,l0,alpha,sr)
    y = y/ max(y)
    filename = os.path.join(path_out,str(val_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)


print("finished validation set!!")


path_out = os.path.join(out_path,"test")
if not os.path.exists(path_out):
    os.makedirs(path_out)
for i in range(n_samp_test):
    omega,tau,p,D,alpha = test_params[i,1:-1]
    y = ftm.getsounds_imp_linear_nonorm(mode,mode,x1,x2,h,tau,omega,p,D,l0,alpha,sr)
    y = y/ max(y)
    filename = os.path.join(path_out,str(test_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)
    
    
print("finished test set!!")

path_out = os.path.join(out_path,"train")
if not os.path.exists(path_out):
    os.makedirs(path_out)
for i in range(n_samp_train):
    omega,tau,p,D,alpha = train_params[i,1:-1]
    y = ftm.getsounds_imp_linear_nonorm(mode,mode,x1,x2,h,tau,omega,p,D,l0,alpha,sr)
    y = y/ max(y)
    filename = os.path.join(path_out,str(train_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)
    

print("finished train set!!")






