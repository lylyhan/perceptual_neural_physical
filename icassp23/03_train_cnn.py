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
import torchvision
import icassp23
import sys

save_dir = sys.argv[1] #/home/han/data/

data_dir = os.path.join(save_dir, 'x')
weight_dir = os.path.join(save_dir, 'J')
model_dir = os.path.join(save_dir, 'f_W')
cqt_dir = data_dir



batchsize = 64 #should be smaller for spectral loss
epoch_max = 10
steps_per_epoch = 50
max_steps = steps_per_epoch * epoch_max
#feature parameters
Q = 12
J = 10
outdim = 4
cnn_type = "efficientnet" # efficientnet / wav2shape
loss_type = "ploss" # spec / weighted_p / ploss
weight_type = "None" #novol / pnp / None

if __name__ == "__main__":
    model_save_path = os.path.join(model_dir, "_".join([cnn_type, loss_type, weight_type, str(J), str(Q),"batchsize"+str(batchsize)]))
    os.makedirs(model_save_path, exist_ok=True) 
    y_norms, scaler = icassp23.scale_theta()
    full_df = icassp23.load_fold(fold="full")
    #initialize dataset
    dataset = DrumDataModule(batch_size=batchsize,
                            data_dir=data_dir, #path to hdf5 files
                            cqt_dir=cqt_dir,
                            df=full_df,
                            weight_dir=weight_dir, #path to gradient folders
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
                                    )

    #initialize trainer, declare training parameters, possiibly in neural/cnn.py
    trainer = pl.Trainer(
                        accelerator='gpu',
                        devices=-1,
                        auto_select_gpus=True,
                        max_epochs=epoch_max,
                        max_steps = max_steps,
                        weights_save_path=model_save_path,
                        limit_train_batches=steps_per_epoch, #if integer than it's #steps per epoch, if float then it's percentage
                        limit_val_batches=1.0,
                        limit_test_batches=1.0,
                        callbacks=[checkpoint_cb],
                        )
    #train
    trainer.fit(model, dataset)


    test_loss = trainer.test(model, dataset, verbose=True)
    print("average test loss", test_loss) 
