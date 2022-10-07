from re import X
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from nnAudio.features import CQT
from pnp_synth.neural import forward
from pnp_synth.neural import loss as losses
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from pnp_synth.physical import ftm
import auraloss
from pnp_synth import utils



#logscale param
eps = 1e-3


class ConvNormActivation2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1,
                 padding=0, groups=1, act=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.ReLU() if act else nn.Identity()
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


class wav2shape(pl.LightningModule):
    def __init__(self, in_channels, bin_per_oct,outdim, loss,scaler):
        super().__init__()
        self.block = nn.Sequential(
            #nn.BatchNorm1d(in_channels, eps=1e-05, momentum=0.1),
            ConvNormActivation2d(in_channels, 128, kernel_size=(bin_per_oct,8), padding="same"),
            nn.AvgPool2d(kernel_size=(1,8),padding=0),
            ConvNormActivation2d(128, 64, kernel_size=(bin_per_oct,4), padding="same"),
            ConvNormActivation2d(64, 64, kernel_size=(bin_per_oct,4), padding="same"),
            nn.AvgPool2d(kernel_size=(1,8),padding=0),
            ConvNormActivation2d(64, 8, kernel_size=(bin_per_oct,1), padding="same"),
            nn.Flatten(),
            nn.Linear(3840,64),#not sure in channel
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.ReLU(),
            nn.Linear(64,outdim),#add nonneg kernel regulation
            #nn.Linear.weight.data.clamp(min=0) #nonnegative constraint
            #LINEAR ACTIVATION??
        )
        self.loss_type = loss
        if self.loss_type == "ploss":
            self.loss = F.mse_loss
        elif self.loss_type == "weighted_p":
            self.loss = losses.loss_bilinear
        elif self.loss_type == "spec":
            self.loss = losses.loss_spec
            self.specloss = auraloss.freq.MultiResolutionSTFTLoss()
        self.scaler = scaler
        
    def forward(self, input_tensor):
        #weights (n filters, n_current channel, kernel 1, kernel2)
        #input (n_channel, bs, width, height)    
        input_tensor = input_tensor.unsqueeze(1).type(torch.float32)
        return self.block(input_tensor)

    
    def step(self, batch, fold):
        Sy = batch['feature']
        y = batch['y']
        weight = batch['weight']
        M = batch['M']
        outputs = self(Sy)
        
        #print("dimensions",M.shape,weight.shape,outputs.shape)

        if self.loss_type == "weighted_p":
            loss = self.loss(weight[:,None] * outputs, y, M)
        elif self.loss_type == "spec":
            loss = self.loss(weight[:,None] * outputs, y, self.specloss, self.scaler)
        else:
            loss = self.loss(weight[:,None] * outputs, y)

        tensorboard_logs = {fold + '_loss': loss}
        return {'loss': loss, 'log':tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=True)
        
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)

        return avg_loss

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.log('val_loss', avg_loss, on_step=False,
                 prog_bar=True, on_epoch=True)
        # use key 'log' to load Tensorboard
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

#dataset


class EffNet(pl.LightningModule):
    def __init__(self, in_channels,outdim,loss,scaler):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(out_channels=1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=(3,3))
        self.model = torchvision.models.efficientnet_b0(in_channels=in_channels, num_classes=outdim)
        self.loss_type = loss
        self.metrics = forward.pnp_forward
        if self.loss_type == "ploss":
            self.loss = F.mse_loss
        elif self.loss_type == "weighted_p":
            self.loss = losses.loss_bilinear
        elif self.loss_type == "spec":
            self.loss = losses.loss_spec
            self.specloss = auraloss.freq.MultiResolutionSTFTLoss()
        self.val_loss = None

        self.y_max, self.y_min = scaler.data_max_, scaler.data_min_

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1) 
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x) # adapt to efficientnet's mandatory 3 input channels
        x = self.model(x)
        #input_tensor = torch.stack([input_tensor]*3,dim=1).type(torch.float32)
        ##TODO: concatenate three channels to be real part, imaginary part, magnitude of the CQT spectrum
        return x

    def step(self, batch, fold):
        Sy = batch['feature']
        y = batch['y']
        weight = batch['weight']
        M = batch['M']

        outputs = self(Sy)
        if self.loss_type == "weighted_p":
            loss = self.loss(weight[:,None] * outputs, y, M)
        elif self.loss_type == "spec":
            loss = self.loss(weight[:,None] * outputs, y, self.specloss, self.scaler)
        else:
            loss = self.loss(weight[:,None] * outputs, y)

        tensorboard_logs = {fold + '_loss': loss}
       
        if fold == "val":
            self.log('val_loss', loss, on_step=False,
                 prog_bar=True, on_epoch=True)
        return {fold + '_loss': loss, 'log':tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def training_epoch_end(self, outputs):

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=True)
        
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)

        return avg_loss

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.log('val_loss', avg_loss, on_step=False,
                 prog_bar=True, on_epoch=True)
        # use key 'log' to load Tensorboard
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DrumData(Dataset):
    def __init__(self,
                 y_norms, #normalized groundtruth
                 ids,
                 audio_dir = '/home/han/data/drum_data', #dir to one of the hdf5 files
                 weights_dir = '/home/han/data/drum_data', #dir to folder that stores weights
                 weight_type = 'novol', #novol, pnp
                 fold='train',
                 feature='cqt',
                 J = 10,
                 Q = 12):
        super().__init__()

        self.fold = fold
        self.y_norms = y_norms # partial annotation corresponding to current set load_fold(fold)
        self.ids = ids
        self.audio_dir = audio_dir
        self.weights_dir = weights_dir
        self.weight_type = weight_type
        #normalization 
        
        self.feature = feature
        self.J = J
        self.Q = Q
        #temporary for han
        self.M = torch.tensor(np.load(os.path.join(weights_dir, fold+"_grad_jtfs.npy")),
                            dtype=torch.float32).cuda()
        self.sr = ftm.constants['sr']
        if feature == 'cqt':
            cqt_params = {
                    'sr': self.sr,
                    'n_bins': self.J * self.Q,
                    'hop_length': 256,
                    }
            #find fmin
            if 2**self.J * 32.7 >= cqt_params['sr']/2:
                fmin = 0.4 * cqt_params['sr'] / 2 ** self.J
            else:
                fmin = 32.7
            self.cqt = CQT(**cqt_params,fmin=fmin).cuda()
        
   

    def __getitem__(self, idx): #conundrum: id and y_norm belong to different data structures df_annotation has the id, self.y_norms has the y_norm
        
        y_norm = self.y_norms[idx,:]
        id = self.ids[idx]
        M = None
        weight = torch.tensor(1)
        if self.weight_type:
            #load JTJ
            #M = np.read(os.path.join(self.weights_dir, self.fold, str(id) + "_grad_jtfs.npy")) #(5, 5) on hpc
            #temporary for han
            M = self.M[idx, :, :]
            #compute riemannian
            if self.weight_type == "pnp":
                w,v = torch.linalg.eig(M)
                w = w.type(torch.float32)
                #take 2 biggest eigenvaluess
                weight = torch.sqrt((sorted(w)[-1]*sorted(w)[-2]))

        if self.feature == "cqt":
            #need to change this to loading hdf5 files 
            x, sr = sf.read(os.path.join(self.audio_dir, self.fold, str(id) + "_sound.wav"))
            #print(sr,self.sr)
            #assert int(sr) != int(self.sr)
            x = torch.tensor(x, dtype=torch.float32).cuda()
            Sy = self.cqt(x)[0]
            #logscale
            Sy = torch.log1p(Sy/eps)
            return {'feature': torch.abs(Sy), 'y': y_norm, 'weight': weight, 'M': M}

    def __len__(self):
        return self.y_norms.shape[0]

class DrumDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = '/home/han/data/drum_data/',
                 df=None,
                 weight_dir = '/home/han/data/drum_data',
                 weight_type = 'novol', #novol, pnp
                 batch_size: int = 32,
                 J = 10,
                 Q = 12,
                 feature='cqt'):
        super().__init__()
        self.data_dir = data_dir
  
        self.weight_dir = weight_dir
        self.weight_type = weight_type
        self.batch_size = batch_size
        self.feature = feature
        self.J = J
        self.Q = Q
        self.full_df = df


    def setup(self, stage=None):
        
        y_norms_train, scaler= utils.scale_theta(self.full_df, "train") 
        y_norms_test, scaler = utils.scale_theta(self.full_df, "test") 
        y_norms_val, scaler = utils.scale_theta(self.full_df, "val")

        train_ids = self.full_df[self.full_df["fold"]=="train"]['ID'].values
        test_ids = self.full_df[self.full_df["fold"]=="test"]['ID'].values
        val_ids = self.full_df[self.full_df["fold"]=="val"]['ID'].values

        self.train_ds = DrumData(y_norms_train, #partial dataframe
                                train_ids,
                                self.data_dir, 
                                self.weight_dir,
                                self.weight_type,
                                fold='train',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q)
        
        self.val_ds = DrumData(y_norms_val, #partial dataframe
                                val_ids,
                                self.data_dir, 
                                self.weight_dir,
                                self.weight_type,
                                fold='val',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q)

        self.test_ds = DrumData(y_norms_test, #partial dataframe
                                test_ids,
                                self.data_dir,
                                self.weight_dir,
                                self.weight_type,
                                fold='test',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q)


    def collate_batch(self, batch):
        Sy = torch.tensor(torch.stack([s['feature'] for s in batch])) #(64,120,257)
        y = torch.tensor([s['y'].astype(np.float32) for s in batch])
        weight = torch.tensor(torch.stack([s['weight'] for s in batch]))
        if type(batch[0]['M']) != type(None):
            M = torch.tensor(torch.stack([s['M'] for s in batch]))
        else:
            M = None
        return {'feature': Sy, 'y': y, 'weight': weight, 'M': M}

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=0)
