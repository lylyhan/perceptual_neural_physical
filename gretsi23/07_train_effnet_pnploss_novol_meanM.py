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
from pytorch_lightning.callbacks import ModelCheckpoint
import sklearn
import sys
import time
import torch
from pytorch_lightning import loggers as pl_loggers
from datetime import timedelta

import setups
from pnp_synth.neural import cnn
from pnp_synth import utils

start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]  # /home/han/data/
init_id = sys.argv[2]
batch_size = int(sys.argv[3])
if len(sys.argv) < 5:
    is_train = True
else:
    is_train = False
    ckpt_path = sys.argv[4]

print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [joblib, nnAudio, np, pl, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

data_dir = os.path.join(save_dir, "x")
weight_dir = os.path.join(save_dir, "J")
model_dir = os.path.join(save_dir, "f_W")
cqt_dir = data_dir

epoch_max = 70
steps_per_epoch = setups.SAMPLES_PER_EPOCH / batch_size
max_steps = steps_per_epoch * epoch_max
# feature parameters
Q = 12
J = 6
outdim = 3
bn_var = 0.5
sr = 2 ** 13
cnn_type = "efficientnet"  # efficientnet / cnn.wav2shape
loss_type = "weighted_p"  # spec / weighted_p / ploss
weight_type = "novol"  # novol / pnp / None
LMA = {
    'mode': "constant", #scheduled / constant
    'lambda': 1,
    'threshold': 1e+8,
    'accelerator': 0.5,
    'brake': 1,
    'damping': "mean"
}
logscale_theta = True
assert utils.logscale == logscale_theta
utils.logscale = logscale_theta 
setups.logscale = logscale_theta


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Current device: ", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method('spawn')
    model_save_path = os.path.join(
        model_dir,
        "_".join(
            [
                cnn_type,
                loss_type,
                weight_type,
                str(J),
                str(Q),
                "batch_size" + str(batch_size),
                "bn_var" + str(bn_var),
                "init-" + str(init_id),
                "LMA_" + str(np.log10(LMA['lambda'])) + "_" + LMA['mode'],
                "brake_"+"{:0.2f}".format(LMA['brake']),
                "damping_"+str(LMA['damping']),
                #"outdim-" + str(outdim),
                 "log-" + str(logscale_theta),
            ]
        ),
    )
    os.makedirs(model_save_path, exist_ok=True)
    pred_path = os.path.join(model_save_path, "test_predictions.npy")
    y_norms, scaler = setups.scale_theta()
    full_df = setups.load_fold(fold="full")
    # initialize dataset
    dataset = cnn.DrumDataModule(
        batch_size=batch_size,
        data_dir=data_dir,  # path to hdf5 files
        cqt_dir=cqt_dir,
        df=full_df,
        weight_dir=weight_dir,  # path to gradient folders
        weight_type=weight_type,  # novol, pnp
        feature="cqt",
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
        model = cnn.EffNet(in_channels=1, outdim=outdim, loss=loss_type, scaler=scaler, var=bn_var, LMA=LMA, save_path=pred_path)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # initialize checkpoint methods
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_loss",
        save_last=True,
        filename= "best",#"ckpt-{epoch:02d}-{val_loss:.2f}",
        save_weights_only=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(model_save_path,"logs"))
    # initialize trainer, declare training parameters, possiibly in neural/cnn.py
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        auto_select_gpus=True,
        max_epochs=epoch_max,
        max_steps=max_steps,
        limit_train_batches=steps_per_epoch,  # if integer than it's #steps per epoch, if float then it's percentage
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        callbacks=[checkpoint_cb],
        logger=tb_logger,
        max_time=timedelta(hours=12)
    )
    # train
    if is_train:
        print("Training ...")
        trainer.fit(model, dataset)
    else:
        print("Skipped Training, loading model")
        model = model.load_from_checkpoint(os.path.join(model_save_path, ckpt_path),in_channels=1, outdim=outdim, loss=loss_type, scaler=scaler, var=bn_var, LMA=LMA, save_path=pred_path)


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
