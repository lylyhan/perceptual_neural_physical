"""
This script trains an EfficientNet for sound matching of drum sounds
with PNP loss without Riemmanian volume weights as its objective.
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
from pytorch_lightning.tuner.tuning import Tuner

import icassp25
from pnp_synth.neural import cnn
from pnp_synth import utils

start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]  # /home/han/data/
init_id = sys.argv[2]
minmax = int(sys.argv[3])
logscale_theta = int(sys.argv[4])
opt = sys.argv[5]
eff_type = sys.argv[6]

batch_size = 256
is_train = True
save_freq = 10

print("Command-line arguments:\n" + "\n".join(sys.argv[1:]))
print(f"Batch size: {batch_size}\n")

for module in [joblib, nnAudio, np, pl, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()


names = ["M"]
    
if minmax == False:
    names.append("nominmax")
if logscale_theta == True:
    names.append("log")
J_foldname = "_".join(names)


data_dir = os.path.join(save_dir, "x")
weight_dir = os.path.join(save_dir, J_foldname)
model_dir = os.path.join(save_dir, "f_W")
cqt_dir = data_dir

epoch_max = 70
steps_per_epoch = icassp25.SAMPLES_PER_EPOCH / batch_size
max_steps = steps_per_epoch * epoch_max
# feature parameters
Q = 12
J = 10
outdim = 5
sr = 22050

bn_var = 0.5
cnn_type = "efficientnet"  # efficientnet / cnn.wav2shape
loss_type = "weighted_p"  # spec / weighted_p / ploss
weight_type = "novol"  # novol / pnp / None
if loss_type == "weighted_p":
    LMA = {
        'mode': "adaptive", #scheduled / constant
        'accelerator': 0.05,
        'brake': 1,
        'damping': "id"
    }
else:
    LMA = None

lr = 1e-3
finetune = False
mu = 1e-10 # the scaling factor of M

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0" #restrict machine
    print("Current device: ", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method('spawn')
    model_save_path = os.path.join(
        model_dir,
        "_".join(
            [
                eff_type, 
                loss_type,
                "finetune" + str(finetune),
                "log-" + str(logscale_theta),
                "minmax-" + str(minmax),
                "opt-" + opt,
                "batch_size" + str(batch_size),
                "lr-"+ str(lr),
                "mu-"+str(mu),
                "init-" + str(init_id),
            ]
        ),
    )
    os.makedirs(model_save_path, exist_ok=True)
    pred_path = os.path.join(model_save_path, "test_predictions.npy")

    if minmax:
        nus, scaler = icassp25.scale_theta(logscale_theta)
    else:
        scaler = None
    #no min max scaling
    full_df = icassp25.load_fold(fold="full")

    # initialize dataset
    dataset = cnn.DrumDataModule(
        batch_size=batch_size,
        data_dir=data_dir,  # path to hdf5 files
        cqt_dir=cqt_dir,
        df=full_df,
        weight_dir=weight_dir,  # path to gradient folders
        weight_type=weight_type,  # novol, pnp
        feature="cqt",
        logscale=logscale_theta,
        J=J,
        Q=Q,
        sr=sr,
        scaler=scaler,
        num_workers=0
    )

    print(str(datetime.datetime.now()) + " Finished initializing dataset")
    # initialize model, designate loss function

    model = cnn.EffNet(in_channels=1, outdim=outdim, loss=loss_type, eff_type=eff_type, 
                       scaler=scaler, LMA=LMA, steps_per_epoch=steps_per_epoch,
                         var=bn_var, save_path=pred_path, lr=lr, minmax=minmax, 
                         logtheta=logscale_theta, opt=opt, mu=mu)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # initialize checkpoint methods
    # save best checkpoint 
    if loss_type == "ploss":
        abbr_loss = "p"
    elif loss_type == "weighted_p":
        abbr_loss = "pnp"
    prefix = 'step=learn+optimizer={}+model={}+loss={}+log={}_trial_{}'.format(
        opt, eff_type[1], abbr_loss, logscale_theta, init_id)
    checkpoint_cb_best = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_loss",
        save_top_k=1,
        filename= prefix + "bestckpt-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=1,
        save_weights_only=False,
    )
    # save checkpoint every save_freq epochs
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="epoch ploss metrics",
        save_last=True,
        save_top_k=-1,
        filename= prefix + "ckpt-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=save_freq,
        save_weights_only=False,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(model_save_path,"logs"))
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # initialize trainer, declare training parameters, possiibly in neural/cnn.py
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=epoch_max,
        max_steps=max_steps,
        limit_train_batches=steps_per_epoch,  # if integer than it's #steps per epoch, if float then it's percentage
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        callbacks=[checkpoint_cb, checkpoint_cb_best, lr_monitor],
        logger=tb_logger,
        enable_progress_bar=True,
        max_time=None, #timedelta(hours=12)
    )
    
    # train 
    print("Training ...")
    trainer.fit(model, dataset)

    # Print elapsed time.
    print(str(datetime.datetime.now()) + " Success.")
    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.0
    elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
        elapsed_hours, elapsed_minutes, elapsed_seconds
    )
    print("Total elapsed time: " + elapsed_str + ".")
