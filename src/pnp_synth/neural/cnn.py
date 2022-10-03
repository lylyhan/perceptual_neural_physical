from re import X
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from nnAudio.features import CQT

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from pnp_synth.physical import ftm
import auraloss

#logscale param
eps = 1e-3
Y_MAX = None
Y_MIN = None


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
    def __init__(self, in_channels, bin_per_oct,outdim, loss):
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
            self.loss = F.mse_loss
        elif self.loss_type == "spec":
            self.loss = self.loss_spec
            self.specloss = auraloss.freq.MultiResolutionSTFTLoss()
        
    def forward(self, input_tensor):
        #weights (n filters, n_current channel, kernel 1, kernel2)
        #input (n_channel, bs, width, height)    
        input_tensor = input_tensor.unsqueeze(1).type(torch.float32)
        return self.block(input_tensor)

    def loss_spec(self, outputs, y):
        #undo normalization
        y_o = y * (Y_MAX - Y_MIN) + Y_MIN
        outputs_o = outputs * (Y_MAX - Y_MIN) + Y_MIN
        #put through synth ##TODO: need the batch processing!!! or make a loop
        wav_gt = ftm.rectangular_drum(y_o, **ftm.constants)
        wav_pred = ftm.rectangular_drum(outputs_o, **ftm.constants)
        return self.specloss(wav_pred, wav_gt)

    def training_step(self, batch, batch_idx):
        Sy = batch['feature']
        y = batch['y']
        outputs = self(Sy)
        loss = self.loss(outputs, y)
        tensorboard_logs = {'train_loss': loss}
        return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        Sy = batch['feature']
        y = batch['y']
        # Forward pass
        outputs = self(Sy)              
        loss = self.loss(outputs, y)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        #do i add metrics computation here?
        Sy = batch['feature']
        y = batch['y']    
        outputs = self(Sy)
        loss = self.loss(outputs, y)
        return {"test_loss": loss}

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log' to load Tensorboard
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

#dataset


class EffNet(pl.LightningModule):
    def __init__(self, in_channels,outdim,loss):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(out_channels=1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=(3,3))
        self.model = torchvision.models.efficientnet_b0(in_channels=in_channels, num_classes=outdim)
        self.loss_type = loss
        if self.loss_type == "ploss":
            self.loss = F.mse_loss
        elif self.loss_type == "weighted_p":
            self.loss = F.mse_loss
        elif self.loss_type == "spec":
            self.loss = self.loss_spec
            self.specloss = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1) #image net requires 3 channels??
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x)
        x = self.model(x)
        #input_tensor = torch.stack([input_tensor]*3,dim=1).type(torch.float32)
        ##TODO: concatenate three channels to be real part, imaginary part, magnitude of the CQT spectrum
        return x
    
    def loss_spec(self, outputs, y):
        #undo normalization
        y_o = y * (Y_MAX - Y_MIN) + Y_MIN
        outputs_o = outputs * (Y_MAX - Y_MIN) + Y_MIN
        #put through synth ##TODO: need the batch processing!!! or make a loop
        wav_gt = ftm.rectangular_drum(y_o, **ftm.constants)
        wav_pred = ftm.rectangular_drum(outputs_o, **ftm.constants)
        return self.specloss(wav_pred, wav_gt)

    def training_step(self, batch, batch_idx):
        Sy = batch['feature']
        y = batch['y']

        outputs = self(Sy)
        loss = self.loss(outputs, y)
        tensorboard_logs = {'train_loss': loss}
        return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        Sy = batch['feature']
        y = batch['y']
        # Forward pass
        outputs = self(Sy)
        loss = self.loss(outputs, y)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        Sy = batch['feature']
        y = batch['y']
       
        outputs = self(Sy)
        loss = self.loss(outputs, y)
        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DrumData(Dataset):
    def __init__(self,
                 csv_dir='../data',
                 audio_dir = '/home/han/data/drum_data',
                 weights_dir = '/home/han/data/drum_data',
                 weight_type = 'grad', #diag, full, riemann
                 fold='train',
                 feature='cqt',
                 J = 10,
                 Q = 12):
        super().__init__()

        self.fold = fold
        self.csv_path = os.path.join(csv_dir, fold + '_param_log_v2.csv')
        self.audio_dir = audio_dir
        self.weights_dir = weights_dir
        self.weight_type = weight_type
        self.ys = pd.read_csv(self.csv_path).values
        self.feature = feature
        self.J = J
        self.Q = Q
        y_train = pd.read_csv(os.path.join(csv_dir,"train_param_log_v2.csv")).values[:,3:-1]
        self.y_max, self.y_min = self.compute_stats(y_train)
        Y_MAX, Y_MIN = self.y_max, self.y_min
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
        
    def compute_stats(self, y_train):
        return np.max(y_train, axis=0), np.min(y_train,axis=0)

    def __getitem__(self, idx): #idx smaller than length of current fold

        y = self.ys[idx, 3:-1] #including ground truth omega
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        #print("sanitycheck",y_norm)
        id = self.ys[idx, 2]
        if self.weight_type:
            weight = np.read(os.path.join(self.weights_dir, self.fold, str(id) + "_jtfs_grad.npy")) #assuming it's (#coef, 5)
            

        if self.feature == "cqt":
           
            x, sr = sf.read(os.path.join(self.audio_dir, self.fold, str(id) + "_sound.wav"))
            #print(sr,self.sr)
            #assert int(sr) != int(self.sr)
            x = torch.tensor(x, dtype=torch.float32).cuda()
            Sy = self.cqt(x)[0]
            #logscale
            Sy = torch.log1p(Sy/eps)
            return {'feature': torch.abs(Sy), 'y': y_norm}

    def __len__(self):
        return self.ys.shape[0]

class DrumDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = '/home/han/data/drum_data/',
                 csv_dir: str = '../data',
                 batch_size: int = 32,
                 feature='cqt'):
        super().__init__()
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.batch_size = batch_size
        self.feature = feature


    def setup(self, stage=None):
        self.train_ds = DrumData(self.csv_dir,self.data_dir, fold='train',
                                feature='cqt',
                                J = 10,
                                Q = 12)
        
        self.val_ds = DrumData(self.csv_dir,self.data_dir, fold='val',
                                feature='cqt',
                                J = 10,
                                Q = 12)

        self.test_ds = DrumData(self.csv_dir,self.data_dir, fold='test',
                                feature='cqt',
                                J = 10,
                                Q = 12)

    def collate_batch(self, batch):
        Sy = torch.tensor(torch.stack([s['feature'] for s in batch])) #(64,120,257)
        y = torch.tensor([s['y'].astype(np.float32) for s in batch])
        return {'feature': Sy, 'y': y}

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
