from decimal import Clamped
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
from pnp_synth.perceptual import metrics
import auraloss
from pnp_synth import utils
import h5py
import joblib


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
            nn.Sigmoid()
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
        self.outdim = outdim
        self.metric_macro = metrics.JTFSloss(self.scaler, "macro")
        self.metric_micro = metrics.JTFSloss(self.scaler, "micro")

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

         # match outputs and y dimension
        if outputs.shape[1] == 4 and y.shape[1] == 5:
            outputs = torch.cat((y[:,0][:,None], outputs),dim=1)
        assert outputs.shape[1] == 5

        #compute loss function
        if self.loss_type == "spec":
            loss = self.loss(outputs, y, self.specloss, self.scaler)
        else:
            if self.loss_type == "weighted_p":
                loss = self.loss(weight[:,None] * outputs, y, M)
            else:
                loss = self.loss(weight[:,None] * outputs, y)
        #compute metrics
        if fold == "test":
            self.metric_macro.update(outputs, y, weight)
            self.metric_micro.update(outputs, y, weight)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=False, sync_dist=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_macro_metric = self.metric_macro.compute() #torch.stack(x['metric'] for x in outputs).mean()
        avg_micro_metric = self.metric_micro.compute()
        self.log('test_loss', avg_loss)
        self.log('macro_metrics', avg_macro_metric)
        self.log('micro_metrics', avg_micro_metric)
        return avg_loss, avg_macro_metric, avg_micro_metric

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_step=False,
                 prog_bar=False, on_epoch=True)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class EffNet(pl.LightningModule):
    def __init__(self, in_channels,outdim,loss,scaler,var,LMA=None):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=(3,3))
        self.model = torchvision.models.efficientnet_b0(in_channels=in_channels, num_classes=outdim)
        self.batchnorm2 = nn.BatchNorm1d(outdim, eps=1e-5, momentum=0.1, affine=False)
        self.act = nn.Sigmoid()
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
        self.scaler = scaler
        self.outdim = outdim
        self.metric_macro = metrics.JTFSloss(self.scaler, "macro")
        self.metric_micro = metrics.JTFSloss(self.scaler, "micro")
        self.std = torch.sqrt(torch.tensor(var))
        self.monitor_valloss = torch.inf
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        if LMA:
            self.LMA_lambda0 = LMA['lambda']
            self.LMA_lambda = LMA['lambda']
            self.LMA_threshold = LMA['threshold']
            self.LMA_accelerator = LMA['accelerator']
            self.LMA_brake = LMA['brake']
            self.LMA_mode = LMA['mode']
            self.LMA_damping = LMA['damping']
        else:
            self.LMA_lambda0 = 1e+15
            self.LMA_lambda = 1e+15
            self.LMA_threshold = 1e+20
            self.LMA_accelerator = 0.1
            self.LMA_brake = 10
            self.LMA_mode = "adaptive"
            self.LMA_damping = "identity"
        self.best_params = self.parameters
        self.epoch = 0

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x) # adapt to efficientnet's mandatory 3 input channels
        x = self.model(x)
        x = self.batchnorm2(x) * self.std
        x = self.act(x)
        return x

    def step(self, batch, fold):
        Sy = batch['feature'].to(self.current_device)
        y = batch['y'].to(self.current_device).double()
        weight = batch['weight'].to(self.current_device)
        M = batch['M'].to(self.current_device).double()
        metric_weight = batch['metric_weight'].to(self.current_device)
        outputs = self(Sy)

        # match outputs and y dimension
        if outputs.shape[1] == 4 and y.shape[1] == 5:
            outputs = torch.cat((y[:,0][:,None], outputs),dim=1)
        assert outputs.shape[1] == 5

        #compute loss function
        if self.loss_type == "spec":
            loss = self.loss(outputs, y, self.specloss, self.scaler)
        else:
            if self.loss_type == "weighted_p":
                if fold == "val" or fold == "test":
                    D = torch.zeros(M.shape).double()
                elif self.LMA_damping == "identity":
                    D = torch.eye(M.shape[1]).double()[None, :, :]
                elif self.LMA_damping == "diag":
                    D = torch.diag_embed(M)
                D = self.LMA_lambda * D.to(self.current_device)
                M = M + D
                loss = self.loss(
                    weight[:, None] * outputs.double(),
                    y.double(),
                    M
                )
            else:
                loss = self.loss(weight[:,None] * outputs, y)
        #compute metrics
        if fold == "test":
            self.metric_macro.update(outputs, y, metric_weight)
            self.metric_micro.update(outputs, y, metric_weight)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=False)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_macro_metric = self.metric_macro.compute() #torch.stack(x['metric'] for x in outputs).mean()
        avg_micro_metric = self.metric_micro.compute()
        self.log('test_loss', avg_loss)
        self.log('macro_metrics', avg_macro_metric)
        self.log('micro_metrics', avg_micro_metric)
        return avg_loss, avg_macro_metric, avg_micro_metric

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()


        if self.loss_type == "weighted_p":
            if self.LMA_mode == "adaptive":
                # Levenburg-Marquardt Algorithm, lambda decay heuristics
                if avg_loss < self.monitor_valloss:
                    self.monitor_valloss = avg_loss
                    self.LMA_lambda = self.LMA_lambda * self.LMA_accelerator
                    self.best_params = self.parameters
                else:
                    if self.LMA_lambda * self.LMA_brake < self.LMA_threshold:
                        self.LMA_lambda = self.LMA_lambda * self.LMA_brake
                    else:
                        self.LMA_lambda = self.LMA_threshold
                    self.parameters = self.best_params
            elif self.LMA_mode == "scheduled":
                self.epoch += 1
                self.LMA_lambda = self.LMA_lambda * self.LMA_accelerator


        self.log('LMA_lambda', self.LMA_lambda)
        self.log('val_loss', avg_loss, on_step=False,
                 prog_bar=False, on_epoch=True)

        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DrumData(Dataset):
    def __init__(self,
                 y_norms, #normalized groundtruth
                 ids,
                 audio_dir, #path to audio hdf files
                 cqt_dir, #path to cached cqt
                 weights_dir, #path to weight files
                 weight_type, #novol, pnp
                 fold,
                 feature,
                 J,
                 Q):
        super().__init__()

        self.fold = fold
        self.y_norms = y_norms # partial annotation corresponding to current set load_fold(fold)
        self.ids = ids
        self.audio_dir = audio_dir #path to hdf5 file
        self.weights_dir = weights_dir
        self.weight_type = weight_type

        self.feature = feature
        self.J = J
        self.Q = Q
        self.fold = fold
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
            self.cqt_from_x = CQT(**cqt_params,fmin=fmin).cuda()

        # Initialize joblib Memory object
        # self.cqt_memory = joblib.Memory(cqt_dir, verbose=0)
        # self.cqt_from_id = self.cqt_memory.cache(self.cqt_from_id)


    def __getitem__(self, idx): #conundrum: id and y_norm belong to different data structures df_annotation has the id, self.y_norms has the y_norm
        y_norm = self.y_norms[idx,:] # look up which row is this id corresponding to ??
        id = self.ids[idx]
        M = None
        weight = torch.tensor(1)
        #load JTJ
        M, sigma = self.M_from_id(id)
        metric_weight = torch.sqrt((sorted(sigma)[-1]*sorted(sigma)[-2]))
        if self.weight_type != "None" and self.weight_type == "pnp":
            weight = metric_weight
        if self.feature == "cqt":
            Sy = self.cqt_from_id(id, eps)
            return {'feature': torch.abs(Sy), 'y': y_norm, 'weight': weight, 'M': M,
                    'metric_weight': metric_weight}

    def __len__(self):
        return len(self.ids)

    def M_from_id(self,id):
        #load from h5
        with h5py.File(self.weights_dir, "r") as f:
            M = torch.tensor(np.array(f['M'][str(id)]))
            sigma = torch.tensor(f['sigma'][str(id)])
        return M, sigma

        #load from numpy files
        #i_prefix = "icassp23_" + str(id).zfill(len(self.ids))
        #return np.load(os.path.join(self.weights_dir, self.fold, i_prefix + "_grad_jtfs.py"))

    def cqt_from_id(self, id, eps):
        with h5py.File(self.audio_dir, "r") as f:
            x = np.array(f['x'][str(id)])
        x = torch.tensor(x, dtype=torch.float32).cuda()
        Sy = self.cqt_from_x(x)[0]
        Sy = torch.log1p(Sy/eps)
        return Sy


class DrumDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 cqt_dir,
                 df,
                 weight_dir,
                 weight_type,
                 batch_size,
                 J,
                 Q,
                 feature,
                 num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.weight_dir = weight_dir
        self.weight_type = weight_type
        self.batch_size = batch_size
        self.feature = feature
        self.J = J
        self.Q = Q
        self.full_df = df #sorted df
        self.cqt_dir = cqt_dir


    def setup(self, stage=None):

        y_norms_train, scaler= utils.scale_theta(self.full_df, "train") #sorted by id
        y_norms_test, scaler = utils.scale_theta(self.full_df, "test")
        y_norms_val, scaler = utils.scale_theta(self.full_df, "val")

        train_ids = self.full_df[self.full_df["fold"]=="train"]['ID'].values #sorted by id
        test_ids = self.full_df[self.full_df["fold"]=="test"]['ID'].values
        val_ids = self.full_df[self.full_df["fold"]=="val"]['ID'].values

        """
        #temporary: only keep 1 file in the dataset
        temp_n = 64
        train_ids = np.array(train_ids[:temp_n])
        test_ids = np.array(test_ids[:temp_n])
        val_ids = np.array(val_ids[:temp_n])
        y_norms_train = y_norms_train[:temp_n,:]
        y_norms_test = y_norms_test[:temp_n,:]
        y_norms_val = y_norms_val[:temp_n,:]
        """


        self.train_ds = DrumData(y_norms_train, #partial dataframe
                                train_ids,
                                os.path.join(self.data_dir,"icassp23_train_audio.h5"),
                                self.cqt_dir,
                                os.path.join(self.weight_dir,"icassp23_train_M.h5"),
                                self.weight_type,
                                fold='train',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q)

        self.val_ds = DrumData(y_norms_val, #partial dataframe
                                val_ids,
                                os.path.join(self.data_dir,"icassp23_val_audio.h5"),
                                self.cqt_dir,
                                os.path.join(self.weight_dir,"icassp23_val_M.h5"),
                                self.weight_type,
                                fold='val',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q)

        self.test_ds = DrumData(y_norms_test, #partial dataframe
                                test_ids,
                                os.path.join(self.data_dir,"icassp23_test_audio.h5"),
                                self.cqt_dir,
                                os.path.join(self.weight_dir,"icassp23_test_M.h5"),
                                self.weight_type,
                                fold='test',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q)


    def collate_batch(self, batch):
        Sy = torch.stack([s['feature'] for s in batch]) #(64,120,257)x
        y = torch.tensor(np.array([s['y'].astype(np.float32) for s in batch]))
        weight = torch.stack([s['weight'] for s in batch])
        metric_weight = torch.stack([s['metric_weight'] for s in batch])
        if type(batch[0]['M']) != type(None):
            M = torch.stack([s['M'] for s in batch])
        else:
            M = None
        return {'feature': Sy, 'y': y, 'weight': weight, 'M': M, 'metric_weight': metric_weight}

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)
