"""
This script trains an EfficientNet for sound matching of drum sounds
with Parametric Loss (P-Loss) as its objective.
"""
from ast import Mod
import datetime
import joblib
import nnAudio
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import sklearn
import sys
import time
import torch
from pytorch_lightning import loggers as pl_loggers
from datetime import timedelta
import pandas as pd

import icassp25
from pnp_synth.neural import cnn
import doce


save_dir = "/gpfswork/rech/aej/ufg99no/data/ftm_jtfs"
loss_type = "p"
minmax = 1
logscale_theta = 1
finetune = False
epoch_max = 70
save_freq = 10

eff_type = sys.argv[0]
opt = sys.argv[1]
init_id = int(sys.argv[2])

if eff_type == "b0":
        batch_size = 256
else:
        batch_size = 128

doce.eval(save_dir, init_id, batch_size, 
        loss_type, eff_type, minmax, logscale_theta, finetune,
        opt, save_freq, epoch_max)