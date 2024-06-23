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


#this function takes in 

def train(save_dir, init_id, batch_size, 
          loss_type, eff_type, minmax, logscale_theta, finetune,
          opt, save_freq, epoch_max):
    """
    This function will train a specified model, and save one model checkpoint every save_freq epochs.
    it should also output 2 vectors of average train loss and validation loss per save_freq epoch? (this is available in tensorboard!!)

    save_dir: directory to where the home folder where data, experimental results are saved
    init_id: trial number 0-4
    batch_size: 256
    loss_type: ploss, weighted_p (PNP), specl2 (L2 MSS), spec(L1 + logL1 MSS)
    eff_type: b0, b1, ... b7
    minmax: 0, 1 True
    logscale_theta: 0, 1 True
    finetune: 0, 1 False
    opt: sophia, adam
    save_freq: frequency (in epoch) to save model checkpoints
    epoch_max: maximum training epochs 70
    """

    start_time = int(time.time())
    print(str(datetime.datetime.now()) + " Start.")
    print(__doc__ + "\n")

    data_dir = os.path.join(save_dir, "x")
    weight_dir = os.path.join(save_dir, "M_log")
    model_dir = os.path.join(save_dir, "f_W")
    cqt_dir = data_dir

    steps_per_epoch = icassp25.SAMPLES_PER_EPOCH / batch_size
    max_steps = steps_per_epoch * epoch_max
    
    Q = 12
    J = 10
    outdim = 5
    sr = 22050
    bn_var = 0.5

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

    print("Current device: ", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method('spawn')
    model_save_path = os.path.join(
        model_dir,
        "_".join([
            eff_type, 
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
        nus, scaler = icassp25.scale_theta(logscale_theta)
    else:
        scaler = None

    #no min max scaling
    full_df = icassp25.load_fold(fold="full")
    #print("sanity check", full_df)

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
                       logtheta=logscale_theta, opt=opt)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # initialize checkpoint methods
    # save checkpoint every save_freq epochs
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_loss",
        save_last=True,
        filename= "ckpt-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=save_freq,
        save_weights_only=False,
    )
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
        filename= prefix + "bestckpt-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=1,
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
        enable_progress_bar=True,
        logger=tb_logger,
        max_time=None#timedelta(hours=12)
    )

    # train
    print("Training ...")
    trainer.fit(model, dataset) # whatever loss is used for training

    #extract tensorboard logs
    


    print(str(datetime.datetime.now()) + " Success.")
    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.0
    elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
        elapsed_hours, elapsed_minutes, elapsed_seconds
    )
    print("Total elapsed time: " + elapsed_str + ".")
    
    



def eval(save_dir, init_id, batch_size, 
          loss_type, eff_type, minmax, logscale_theta, finetune,
          opt, save_freq, epoch_max):
    """
     This function will load a series of model checkpoint and output its test metrics. 
     its best if all metrics type are computed simultaneously during test run,
     instead of calling to be computed individually in this function

     save_freq is an argument to the logger, data will be pulled from the tensorboard logs

     """
    
    start_time = int(time.time())
    print(str(datetime.datetime.now()) + " Start.")
    print(__doc__ + "\n")

    data_dir = os.path.join(save_dir, "x")
    weight_dir = os.path.join(save_dir, "M_log")
    model_dir = os.path.join(save_dir, "f_W")
    cqt_dir = data_dir

    steps_per_epoch = icassp25.SAMPLES_PER_EPOCH / batch_size
    max_steps = steps_per_epoch * epoch_max
    
    Q = 12
    J = 10
    outdim = 5
    sr = 22050
    bn_var = 0.5

    weight_type = "None"  # novol / pnp / None

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
    mu = 1e-10
    
    print("Current device: ", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method('spawn')
    if loss_type == "weighted_p":
        name_list = [
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
    else:
        name_list = [
            eff_type, 
            loss_type,
            "finetune" + str(finetune),
            "log-" + str(logscale_theta),
            "minmax-" + str(minmax),
            "opt-" + opt,
            "batch_size" + str(batch_size),
            "lr-"+ str(lr),
            "init-" + str(init_id),
            ]
        
    model_save_path = os.path.join(model_dir, "_".join(name_list))
    pred_path = os.path.join(model_save_path, "test_predictions.npy")
    os.makedirs(model_save_path, exist_ok=True)

    if minmax: 
        nus, scaler = icassp25.scale_theta(logscale_theta)
    else:
        scaler = None

    #no min max scaling
    full_df = icassp25.load_fold(fold="full")
    #print("sanity check", full_df)

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
    if loss_type == "weighted_p":
        model = cnn.EffNet(in_channels=1, outdim=outdim, loss=loss_type, eff_type=eff_type,
                       scaler=scaler, LMA=LMA, steps_per_epoch=steps_per_epoch, 
                       var=bn_var, save_path=pred_path, lr=lr, minmax=minmax, 
                       logtheta=logscale_theta, opt=opt, mu=mu)
    else:
        model = cnn.EffNet(in_channels=1, outdim=outdim, loss=loss_type, eff_type=eff_type,
                       scaler=scaler, LMA=LMA, steps_per_epoch=steps_per_epoch, 
                       var=bn_var, save_path=pred_path, lr=lr, minmax=minmax, 
                       logtheta=logscale_theta, opt=opt)
    print(str(datetime.datetime.now()) + " Finished initializing model")


    metrics = {}
    for file in os.listdir(model_save_path):
        if "ckpt" in file and "best" not in file and "last" not in file:      
            epoch = file.split("=")[-2][:2]
            pred_path = os.path.join(model_save_path,v "test_predictions_epoch{}.npy".format(epoch))
            model = model.load_from_checkpoint(os.path.join(model_save_path, file), in_channels=1, 
                                               outdim=outdim, loss=loss_type, scaler=scaler,var=bn_var, 
                                               eff_type=eff_type, save_path=pred_path, steps_per_epoch=steps_per_epoch, 
                                               lr=lr, LMA=LMA, minmax=minmax,logtheta=logscale_theta, opt=opt)
            
            # initialize checkpoint methods
            # save checkpoint every save_freq epochs
            checkpoint_cb = ModelCheckpoint(
                dirpath=model_save_path,
                monitor="val_loss",
                save_last=True,
                filename= "ckpt-{epoch:02d}-{val_loss:.2f}",
                every_n_epochs=save_freq,
                save_weights_only=False,
            )
            # save best checkpoint 
            checkpoint_cb_best = ModelCheckpoint(
                dirpath=model_save_path,
                monitor="val_loss",
                filename= "bestckpt-{epoch:02d}-{val_loss:.2f}",
                every_n_epochs=1,
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
                enable_progress_bar=True,
                logger=tb_logger,
                max_time=None#timedelta(hours=12)
            )

            print("Testing model ... ")
            test_loss = trainer.test(model, dataset, verbose=False) # avg_loss, avg_macro_metric, avg_micro_metric, avg_mss_metric   
            print(test_loss)
            avg_loss, macro, micro, mss = test_loss[0]["test_loss"], test_loss[0]["macro_metrics"], test_loss[0]["micro_metrics"], test_loss[0]["mss metrics"]
            metrics[epoch] = {"test loss": avg_loss, "macro": macro, "micro": micro, "mss": mss}
            print("Model saved at: {}".format(model_save_path))
            print("Average test loss: {}".format(test_loss))
            print("\n")
    df = pd.DataFrame.from_dict(metrics)       
    df.to_csv(os.path.join(model_save_path, "summarized_metrics.csv"))
    return metrics
