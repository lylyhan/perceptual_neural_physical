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

import taslp23
from pnp_synth.neural import cnn
from pnp_synth import utils

start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]  # /home/han/data/
init_id = sys.argv[2]
minmax = sys.argv[3]
logscale_theta = sys.argv[4]
opt = sys.argv[5]
synth_type = sys.argv[6]

batch_size = 256
is_train = True

print("Command-line arguments:\n" + "\n".join(sys.argv[1:]))
print(f"Batch size: {batch_size}\n")

for module in [joblib, nnAudio, np, pl, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

data_dir = os.path.join(save_dir, "x")
weight_dir = os.path.join(save_dir, "M")
model_dir = os.path.join(save_dir, "f_W")
cqt_dir = data_dir


epoch_max = 70
steps_per_epoch = taslp23.SAMPLES_PER_EPOCH / batch_size
max_steps = steps_per_epoch * epoch_max
# feature parameters
Q = 12
J = 10
outdim = 5
bn_var = 0.5
sr = 22050
cnn_type = "efficientnet"  # efficientnet / cnn.wav2shape
loss_type = "ploss"  # spec / weighted_p / ploss
weight_type = "None"  # novol / pnp / None
LMA = {
    'mode': "constant", #scheduled / constant
    'lambda': 1,
    'threshold': 1e+8,
    'accelerator': 0.5,
    'brake': 1,
    'damping': "mean"
}
lr = 1e-3
finetune = False


if __name__ == "__main__":
    print("Current device: ", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method('spawn')
    model_save_path = os.path.join(
        model_dir,
        "_".join(
            [
                loss_type,
                "finetune" + str(finetune),
                "log-" + str(logscale_theta),
                "minmax-" + str(minmax),
                "opt-" + opt,
                "batch_size" + str(batch_size),
                "lr-"+ str(lr),
                "init-" + str(init_id),
            ]
        ),
    )
    os.makedirs(model_save_path, exist_ok=True)
    pred_path = os.path.join(model_save_path, "test_predictions.npy")

    if minmax:
        nus, scaler = taslp23.scale_theta(logscale_theta)
    else:
        scaler = None
    #no min max scaling
    full_df = taslp23.load_fold(fold="full")

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
    if cnn_type == "cnn.wav2shape":
        model = cnn.wav2shape(
            in_channels=1, bin_per_oct=Q, outdim=outdim, loss=loss_type, scaler=scaler
        )
    elif cnn_type == "efficientnet":
        model = cnn.EffNet(in_channels=1, outdim=outdim, loss=loss_type, scaler=scaler, LMA=LMA, steps_per_epoch=steps_per_epoch, var=bn_var, save_path=pred_path, lr=lr, minmax=minmax, opt=opt)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # initialize checkpoint methods
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_loss",
        save_last=True,
        filename= "best", #"ckpt-{epoch:02d}-{val_loss:.2f}",
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
        callbacks=[checkpoint_cb, lr_monitor],
        enable_progress_bar=True,
        logger=tb_logger,
        max_time=timedelta(hours=12)
    )
    # train
    print("Training ...")
    trainer.fit(model, dataset)

    test_loss = trainer.test(model, dataset, verbose=False)
    print("Model saved at: {}".format(model_save_path))
    print("Average test loss: {}".format(test_loss))
    print("\n")

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
