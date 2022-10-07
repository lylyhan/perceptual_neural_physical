from ast import Mod
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
#from pnp_synth.neural.cnn import DrumDataModule, wav2shape, EffNet
#from pnp_synth.neural import cnn
import sys
sys.path.append("../src/pnp_synth")
from neural import cnn
from neural.cnn import DrumDataModule, wav2shape, EffNet
#from ..src.pnp_synth.neural import cnn
#from ..src.pnp_synth.neural.cnn import  DrumDataModule, wav2shape, EffNet
import torchvision
import icassp23

batchsize = 64
epoch_max = 30
steps_per_epoch = 50
max_steps = steps_per_epoch * epoch_max
#feature parameters
Q = 12
J = 10
outdim = 5
cnn_type = "wav2shape" # efficientnet / wav2shape
loss_type = "ploss" # spec / pnp / ploss
weight_type = "None" #novol / pnp / None
model_save_path = os.path.join("../icassp23/models/", "_".join([cnn_type, loss_type, weight_type, str(J), str(Q)]))
os.makedirs(model_save_path, exist_ok=True)

if __name__ == "__main__":
    y_norms, scaler = icassp23.scale_theta()
    full_df = icassp23.load_fold(fold="full")
    #initialize dataset
    dataset = DrumDataModule(batch_size=batchsize,
                            data_dir='/home/han/data/drum_data/',
                            df=full_df,
                            weight_dir = '/home/han/data/drum_data/',
                            weight_type = weight_type, #novol, pnp
                            feature='cqt',
                            J = J,
                            Q = Q,
                            )

    print("finished initializing dataset")
    #initialize model, designate loss function
    if cnn_type == "wav2shape":
        model = wav2shape(in_channels=1, bin_per_oct=Q, outdim=outdim, loss=loss_type, scaler=scaler)
    elif cnn_type == "efficientnet":
        model = EffNet(in_channels=1, outdim=outdim, loss=loss_type, scaler=scaler)
    print("finished initializing model")

    #initialize checkpoint methods
    checkpoint_cb = ModelCheckpoint(dirpath=model_save_path,
                                    monitor="val_loss",
                                    save_last=True,
                                    filename="ckpt-{epoch:02d}-{val_loss:.2f}",
                                    save_weights_only=False,
                                    #every_n_train_steps=steps_per_epoch,
                                    )

    #initialize trainer, declare training parameters, possiibly in neural/cnn.py
    trainer = pl.Trainer(
                        accelerator='gpu',
                        devices=-1,
                        auto_select_gpus=True,
                        max_epochs=epoch_max,
                        max_steps = max_steps,
                        weights_save_path=model_save_path,
                        limit_train_batches=0.01,
                        limit_val_batches=0.5,
                        callbacks=[checkpoint_cb],
                        #limit_train_batches=n_batches_train
                        )
    #train
    trainer.fit(model, dataset)


    trainer.test(model, dataset,) 