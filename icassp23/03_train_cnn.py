import pytorch_lightning as pl
#from pnp_synth.neural.cnn import DrumDataModule, wav2shape, EffNet
#from pnp_synth.neural import cnn
import sys
sys.path.append("../src/pnp_synth")
from neural import cnn
from neural.cnn import DrumDataModule, wav2shape, EffNet
#from ..src.pnp_synth.neural import cnn
#from ..src.pnp_synth.neural.cnn import  DrumDataModule, wav2shape, EffNet
import torchvision

batchsize = 64
epoch_max = 30
steps_per_epoch = 50
max_steps = steps_per_epoch * epoch_max
#feature parameters
Q = 12
J = 10
outdim = 5
cnn_type = "wav2shape" # efficientnet / wav2shape
loss_type = "ploss" #weighted_p / spec

if __name__ == "__main__":

    #initialize dataset
    dataset = DrumDataModule(batch_size=batchsize,data_dir='/home/han/data/drum_data/',
                            csv_dir='../data',
                            feature='cqt',
                            )

    print("finished initializing dataset")
    #initialize model, designate loss function
    if cnn_type == "wav2shape":
        model = wav2shape(in_channels=1, bin_per_oct=Q, outdim=outdim, loss=loss_type)
    elif cnn_type == "efficientnet":
        model = EffNet(in_channels=1, outdim=outdim, loss=loss_type)
    print("finished initializing model")

    #initialize checkpoint methods
    

    #initialize trainer, declare training parameters, possiibly in neural/cnn.py
    trainer = pl.Trainer(
                        accelerator='gpu',
                        devices=-1,
                        auto_select_gpus=True,
                        max_epochs=epoch_max,
                        max_steps = max_steps,
                        #callbacks=[progbar_callback, checkpoint_cb],
                        #limit_train_batches=n_batches_train
        
                        )
    #train
    trainer.fit(model, dataset)


