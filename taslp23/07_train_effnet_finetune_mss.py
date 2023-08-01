"""
This script trains an EfficientNet for sound matching of drum sounds
with Ploss pretraining and MSS loss finetuning as its objective.
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

import taslp23
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
weight_dir = os.path.join(save_dir, "J")
model_dir = os.path.join(save_dir, "f_W")
cqt_dir = data_dir

epoch_max = 70
steps_per_epoch = taslp23.SAMPLES_PER_EPOCH / batch_size
max_steps = steps_per_epoch * epoch_max
# feature parameters
Q = 12
if synth_type == "ftm":
    J = 10
    outdim = 5
    sr = 22050
elif synth_type == "amchirp":
    J = 6
    outdim = 3
    sr = 2 ** 13
bn_var = 0.5
cnn_type = "efficientnet"  # efficientnet / cnn.wav2shape
loss_type = "spec"  # spec / weighted_p / ploss
weight_type = "novol"  # novol / pnp / None
LMA = {
    'mode': "adaptive", #scheduled / constant
    'lambda': 1e+20,
    'threshold': 1e+20,
    'accelerator': 0.2,
    'brake': 1,
    'damping': "id"
}
lr = 1e-3
finetune = False


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0" #restrict machine
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
        nus, scaler = taslp23.scale_theta(logscale_theta, synth_type)
    else:
        scaler = None
    #no min max scaling
    full_df = taslp23.load_fold(synth_type, fold="full")

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
    dataset_mss = cnn.DrumDataModule(
        batch_size=64,
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
        model = cnn.EffNet(in_channels=1, outdim=outdim, loss="ploss", scaler=scaler, LMA=LMA, steps_per_epoch=steps_per_epoch, var=bn_var, save_path=pred_path, lr=lr, minmax=minmax, logtheta=logscale_theta, opt=opt)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # initialize checkpoint methods
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_loss",
        save_last=True,
        filename= "best",#"ckpt-{epoch:02d}-{val_loss:.2f}",
        save_weights_only=False,
    )

    finetune_dir = os.path.join(model_save_path, "finetune")
    os.makedirs(finetune_dir, exist_ok=True)
    checkpoint_cb_finetune = ModelCheckpoint(
        dirpath=finetune_dir,
        monitor="val_loss",
        save_last=True,
        filename= "best",#"ckpt-{epoch:02d}-{val_loss:.2f}",
        save_weights_only=False,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(model_save_path,"logs"))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # initialize trainer, declare training parameters, possiibly in neural/cnn.py
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=epoch_max // 2,
        max_steps=max_steps // 2,
        limit_train_batches=steps_per_epoch,  # if integer than it's #steps per epoch, if float then it's percentage
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=tb_logger,
        enable_progress_bar=True,
        max_time=timedelta(hours=10)
    )
    trainer_ft = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=epoch_max // 2,
        max_steps=max_steps // 2,
        limit_train_batches=steps_per_epoch,  # if integer than it's #steps per epoch, if float then it's percentage
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        callbacks=[checkpoint_cb_finetune, lr_monitor],
        logger=tb_logger,
        enable_progress_bar=True,
        max_time=timedelta(hours=10)
    )

    # train
    if is_train:
        print("Pre-training ...")
        trainer.fit(model, dataset)
        ckpt_path = os.path.join(model_save_path, "best.ckpt")
        print("Load Pretrained model")
        model = model.load_from_checkpoint(
            os.path.join(model_save_path, ckpt_path), 
            in_channels=1, outdim=outdim, loss=loss_type, scaler=scaler,
            var=bn_var, save_path=pred_path, steps_per_epoch=steps_per_epoch, lr=lr, LMA=LMA, minmax=minmax,logtheta=logscale_theta, opt=opt)
        print("Fine-tuning model")
        trainer_ft.fit(model, dataset_mss) #change batch size to fit gpu
    else:
        model = model.load_from_checkpoint(
            os.path.join(model_save_path, "finetune", "best.ckpt"), 
            in_channels=1, outdim=outdim, loss=loss_type, scaler=scaler,
            var=bn_var, save_path=pred_path, steps_per_epoch=steps_per_epoch, lr=lr, LMA=LMA, minmax=minmax,logtheta=logscale_theta, opt=opt)
        


    test_loss = trainer_ft.test(model, dataset, verbose=False)
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
