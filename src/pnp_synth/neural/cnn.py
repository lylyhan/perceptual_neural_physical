import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
import torch
from torch import nn
from nnAudio.features import CQT
from pnp_synth.neural import loss as losses
from pnp_synth.neural import optimizer
import torchvision
import torch.nn.functional as F
import numpy as np
from pnp_synth.perceptual import metrics
import auraloss
from pnp_synth import utils
import h5py
import muda
import jams
import librosa



#from Sophia import SophiaG 

#logscale param
eps = 1e-3
#relu epsilon
eps_relu = torch.tensor(1e-5)


class EffNet(pl.LightningModule):
    def __init__(self, in_channels,outdim,loss,scaler,var,save_path, steps_per_epoch, eff_type = "b0", lr=1e-3, minmax=True, logtheta=True, LMA=None, opt="adam",mu=1):
        super().__init__()
        self.scaler = scaler
        self.lr = lr
        self.batchnorm1 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=(3,3))
        self.opt = opt
        #if self.opt == "adamW":
        #    self.automatic_optimization=False
        if eff_type == "b0":
            self.model = torchvision.models.efficientnet_b0(num_classes=outdim)#,in_channels=in_channels)
        elif eff_type == "b1":
            self.model = torchvision.models.efficientnet_b1(num_classes=outdim)
        elif eff_type == "b2":
            self.model = torchvision.models.efficientnet_b2(num_classes=outdim)
        elif eff_type == "b3":
            self.model = torchvision.models.efficientnet_b3(num_classes=outdim)
        elif eff_type == "b4":
            self.model = torchvision.models.efficientnet_b4(num_classes=outdim)
        elif eff_type == "b5":
            self.model = torchvision.models.efficientnet_b5(num_classes=outdim)
        elif eff_type == "b6":
            self.model = torchvision.models.efficientnet_b6(num_classes=outdim)
        elif eff_type == 'b7':
            self.model = torchvision.models.efficientnet_b7(num_classes=outdim)
        self.minmax = minmax
        self.logtheta = logtheta
        self.n_batches_train = steps_per_epoch
        #self.optimizer = self.configure_optimizers()
        ##LAST LAYER
        if self.minmax:
            #disable efficientnet last linear layer's bias
            if self.model.get_submodule('classifier')[1].bias.requires_grad:
                self.model.get_submodule('classifier')[1].bias.requires_grad = False
                assert torch.sum(self.model.get_submodule('classifier')[1].bias) == 0
            self.batchnorm2 = nn.BatchNorm1d(outdim, eps=1e-5, momentum=0.1, affine=True)
            self.act = nn.Tanh()
        else:
            if self.logtheta: #if logscaled no minmax allow negative predictions
                self.act = nn.Linear(in_features=outdim, out_features=outdim)
            else: #if no logscale no minmax require nonzero prediction
                if loss == "spec":
                    self.act = nn.Softplus() #guarantees nonzero and improves gradient
                else:
                    self.act = nn.LeakyReLU() #nn.Softplus()
                
            
        self.loss_type = loss
        if self.loss_type == "ploss":
            self.loss = F.mse_loss
        elif self.loss_type == "weighted_p":
            self.loss = losses.loss_bilinear
        elif self.loss_type == "spec":
            self.loss = losses.loss_spec
            self.specloss = auraloss.freq.MultiResolutionSTFTLoss()
        elif self.loss_type == "specl2":
            self.loss = losses.loss_spec
            self.specloss = losses.MultiScaleSpectralLoss(p=2)
        elif self.loss_type == "LMA":
            self.loss = losses.TimeFrequencyScatteringLoss(self.scaler)
        self.save_path = save_path
        if "ftm" in self.save_path:
            self.synth_type = "ftm"
        elif "am" in self.save_path:
            self.synth_type = "amchirp"
        elif "mersenne" in self.save_path:
            self.synth_type = "string"
        self.val_loss = None
        self.outdim = outdim
        self.metric_macro = metrics.JTFSloss(self.scaler, "macro", self.synth_type, logtheta)
        self.metric_micro = metrics.JTFSloss(self.scaler, "micro", self.synth_type, logtheta)
        self.metric_mss = metrics.MSSloss(self.scaler, self.synth_type, logtheta)
        self.mss_validation = metrics.MSSloss(self.scaler, self.synth_type, logtheta)
        self.jtfs_validation = metrics.JTFSloss(self.scaler, "macro", self.synth_type, logtheta)
        self.ploss_validation = []
        self.ploss_test = []
        self.std = torch.sqrt(torch.tensor(var))
        self.monitor_valloss = torch.inf
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        if LMA:
            self.LMA_accelerator = LMA['accelerator']
            self.LMA_brake = LMA['brake']
            self.LMA_mode = LMA['mode']
            self.LMA_damping = LMA['damping']
        else:
            self.LMA_lambda0 = 1e+20
            self.LMA_lambda = 1e+20
            self.LMA_threshold = 1e+20
            self.LMA_accelerator = 0.2
            self.LMA_brake = 1
            self.LMA_mode = "adaptive"
            self.LMA_damping = "id"
        self.best_params = self.parameters
        self.epoch = 0
        self.test_preds = []
        self.test_gts = []
        self.Ms = []
        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []
        self.step_count = 0
        self.update_hessian = 10
        if self.LMA_mode == "adaptive":
            self.LMA_lambda = None
        #this is for accomodating the new dataset
        self.mu = mu
    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x) # adapt to efficientnet's mandatory 3 input channels
        x = self.model(x)
        if self.minmax:
            x = self.batchnorm2(x) * self.std
        x = self.act(x) 
        #if not self.minmax and not self.logtheta and self.loss_type == "spec":
        #    x = torch.abs(x) + eps_relu
        return x

    def step(self, batch, fold, batch_idx):
        Sy = batch['feature'].to(self.current_device)
        y = batch['y'].to(self.current_device).double()
        weight = batch['weight'].to(self.current_device)
        try:
            M = batch['M'].to(self.current_device).double()
            M_mean = batch['M_mean'].to(self.current_device)
            self.LMA_lambda0 = batch['lambda0'].to(self.current_device)
            self.LMA_threshold = self.LMA_lambda0 # set threshold to be the initialized lambda
        except:
            M = None
            M_mean = None
            self.LMA_lambda0 = None
            self.LMA_theshold = None

        if self.LMA_lambda is None and self.LMA_mode == "adaptive":
            self.LMA_lambda = self.LMA_lambda0 #initialize lambda to be intiialized lambda
        try:
            metric_weight = batch['metric_weight'].to(self.current_device)
            JdagJ = batch['JdagJ'].to(self.current_device)
        except:
            metric_weight, JdagJ = None, None

        outputs = self(Sy) 
        assert outputs.shape[1] == self.outdim
        # match outputs and y dimension
        if outputs.shape[1] == 4 and y.shape[1] == 5:
            outputs = torch.cat((y[:,0][:,None], outputs),dim=1)
        
        #compute loss function
        if self.loss_type == "spec" or self.loss_type == "specl2":
            loss = self.loss(outputs, y, self.specloss, self.scaler, self.synth_type, self.logtheta)
        elif self.loss_type == "LMA":
            loss = self.loss(outputs, y, JdagJ)
        else:
            if self.loss_type == "weighted_p":
                if fold == "val" or fold == "test":
                    D = torch.zeros(M.shape).double()
                elif self.LMA_damping == "id":
                    D = torch.eye(M.shape[1]).double()[None, :, :]
                elif self.LMA_damping == "diag":
                    diags = torch.diagonal(M, dim1=-1, dim2=-2) #(bs, 5)
                    D = torch.diag_embed(diags)
                elif self.LMA_damping == "mean": #mean sigma is in ascending order
                    D = M_mean
                D = self.LMA_lambda * D.to(self.current_device)
                M = M + D
                loss = self.loss(
                    weight[:, None] * outputs.double(),
                    y.double(),
                    self.mu * M
                )
            else: #ploss
                loss = self.loss(weight[:,None].double() * outputs.double(), y.double())

        #compute metrics
        if fold == "test":
            self.metric_macro.update(outputs, y, metric_weight)
            self.metric_micro.update(outputs, y, metric_weight)
            self.metric_mss.update(outputs, y)
            self.test_preds.append(outputs)
            self.test_gts.append(y)
            self.Ms.append(M)
        
        if fold == "train":
            self.train_outputs.append(loss)
            if self.opt == None:
                opt = self.optimizers()
                def closure():
                    opt.zero_grad()
                    self.manual_backward(loss, retain_graph=True)#, create_graph=True)
                    return loss
                #self.update_lr(batch_idx)
                opt.step(closure=closure)

        elif fold == "test":
            self.test_outputs.append(loss)
            self.ploss_test.append(F.mse_loss(outputs.double(), y.double()))
        elif fold == "val":
            self.val_outputs.append(loss)
            #self.mss_validation.update(outputs, y)
            #if self.epoch % 10 == 0: 
            #    self.jtfs_validation.update(outputs, y, None)
            #compute comparable validation loss
            self.ploss_validation.append(F.mse_loss(outputs.double(), y.double()))
            #self.log("ploss metrics", F.mse_loss(outputs.double(), y.double()))
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test", batch_idx)

    def on_train_epoch_start(self):
        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []
        self.ploss_validation = []
        self.ploss_test = []
        self.log("lr", self.optimizer.param_groups[-1]['lr'])

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_outputs).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
    
    def on_test_epoch_end(self):
        avg_loss = torch.tensor(self.test_outputs).mean()
        avg_ploss_test = torch.tensor(self.ploss_test).mean()
        avg_macro_metric = self.metric_macro.compute() 
        avg_micro_metric = self.metric_micro.compute()
        avg_mss_metric = self.metric_mss.compute()
        self.log('test_loss', avg_loss)
        self.log('macro_metrics', avg_macro_metric)
        self.log('micro_metrics', avg_micro_metric)
        self.log('mss metrics', avg_mss_metric)
        self.log('ploss', avg_ploss_test)
        self.test_gts = torch.stack(self.test_gts)
        self.test_preds = torch.stack(self.test_preds)
        try:
            self.Ms = torch.stack(self.Ms)
            np.save(self.save_path, [[self.test_gts.detach().cpu().numpy(), 
                                self.test_preds.detach().cpu().numpy()],
                                self.Ms.detach().cpu().numpy()],allow_pickle=True)
        except:
            self.Ms = None
            np.save(self.save_path, [self.test_gts.detach().cpu().numpy(), 
                                    self.test_preds.detach().cpu().numpy()],
                                    allow_pickle=True)

        return avg_loss, avg_macro_metric, avg_micro_metric, avg_mss_metric, avg_ploss_test

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.val_outputs).mean() #current loss function's validation loss
        avg_ploss_validation = torch.tensor(self.ploss_validation).mean()
        if self.loss_type == "weighted_p":
            if self.LMA_mode == "adaptive":
                # Levenburg-Marquardt Algorithm, lambda decay heuristics
                print("what's going on", self.epoch, avg_loss, self.monitor_valloss, self.LMA_lambda)
                if avg_loss < self.monitor_valloss:
                    if self.epoch > 0:
                        self.monitor_valloss = avg_loss
                    self.LMA_lambda = self.LMA_lambda * self.LMA_accelerator
                    self.best_params = self.parameters # register best model
                else:
                    if self.LMA_lambda * self.LMA_brake < self.LMA_threshold:
                        self.LMA_lambda = self.LMA_lambda * self.LMA_brake
                    else:
                        self.LMA_lambda = self.LMA_threshold
                    self.parameters = self.best_params # revert to best model
                
                #disregard the monitor_valloss at the first evaluation
                #if self.epoch == 1:
                #    self.best_params = self.parameters
                #    self.monitor_valloss = avg_loss
                self.epoch += 1

            elif self.LMA_mode == "scheduled":
                self.epoch += 1
                self.LMA_lambda = self.LMA_lambda * self.LMA_accelerator

        if self.LMA_lambda:
            self.log('LMA_lambda', self.LMA_lambda)
        self.log('val_loss', avg_loss, on_step=False,
                 prog_bar=True, on_epoch=True)
        #avg_mss_validation = self.mss_validation.compute()
        #self.epoch += 1
        #if self.epoch % 10 == 0:
        #    avg_jtfs_validation = self.jtfs_validation.compute()
        #    self.log("epoch jtfs metrics", avg_jtfs_validation)
        #self.log("epoch mss metrics", avg_mss_validation)
        self.log("ploss_metrics", avg_ploss_validation)
        
        return {'val_loss': avg_loss, 'ploss_metrics': avg_ploss_validation}

    def configure_optimizers(self):
        if self.opt == "adamW":
            #self.model.automatic_optimization = False
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=1e-1) #decoupled weight decay regularization 
            self.optimizer = optim
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #    optim, T_0=1, T_mult=1, eta_min=1e-8,
            #    last_epoch=-1, verbose=0)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=3)
            
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    },
                }

        elif self.opt == "sophia":
            self.optimizer = optimizer.SophiaG(params=self.parameters(), lr=self.lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
            #self.automatic_optimization = False
            #self.optimizer = optimizer.Sophia(None, None, self.parameters())
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3)
            return{
                'optimizer': self.optimizer,
                'lr_scheduler':{
                    'scheduler': lr_scheduler,
                    'monitor': 'val_loss',
                }
            }

        elif self.opt == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=3)
            return{
                'optimizer': self.optimizer,
                'lr_scheduler':{
                    'scheduler': lr_scheduler,
                    'monitor': 'val_loss',
                },
            }

    def update_lr(self, batch_idx):
        sch = self.lr_schedulers()

        warmup_epochs = 3
        warmup_len = self.n_batches_train * warmup_epochs
        total_step = self.trainer.current_epoch * self.n_batches_train + batch_idx
        if total_step >= warmup_len:
            epoch_frac = total_step / self.n_batches_train
        else:
            # LR warmup for first epoch
            # `batch_idx + 1` to not start with `1` when `batch_idx == 0`
            epoch_frac = 1 - (total_step + 1) / warmup_len
        sch.step(epoch_frac)
        return sch

    #def on_before_optimizer_step(self, optimizer):
    #    self.clip_gradients(
    #        optimizer,
    #        gradient_clip_val=3,
    #        gradient_clip_algorithm='norm',
    #    )

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
                 Q,
                 sr,
                 noise_dir,
                 noise_mode="mix"): #path to noise audio hdf files
        super().__init__()

        self.fold = fold
        self.y_norms = y_norms # partial annotation corresponding to current set load_fold(fold)
        self.ids = ids
        self.audio_dir = audio_dir #path to hdf5 file
        self.noise_dir = noise_dir
        self.weights_dir = weights_dir
        self.weight_type = weight_type
        
        #make list of noise pitches
        if self.noise_dir:
            self.noise_inventory()
            self.isnoise = True
            self.noise_mode = noise_mode
            if self.noise_mode == "statgauss":
                self.noise_gen = muda.deformers.ColoredNoise(n_samples=1, color=['pink'], weight_max=0.08, weight_min=0.01)
        else:
            self.isnoise = False

        self.feature = feature
        self.J = J
        self.Q = Q
        self.fold = fold
        self.sr = sr
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
        try:
            self.M_mean, self.sigma_mean, self.lambda0 = self.make_M_mean()
        except:
            self.M_mean, self.sigma_mean, self.lambda0 = None, None, None
    
        # Initialize joblib Memory object
        # self.cqt_memory = joblib.Memory(cqt_dir, verbose=0)
        # self.cqt_from_id = self.cqt_memory.cache(self.cqt_from_id)


    def __getitem__(self, idx): #conundrum: id and y_norm belong to different data structures df_annotation has the id, self.y_norms has the y_norm
        y_norm = self.y_norms[idx,:] # look up which row is this id corresponding to ??
        id = self.ids[idx]
        M = None
        weight = torch.tensor(1)
        #load JTJ
        M, sigma, JdagJ = self.M_from_id(id)
        try:
            metric_weight = torch.sqrt((sorted(sigma)[-1]*sorted(sigma)[-2]))
        except:
            metric_weight = None
        if self.weight_type != "None" and self.weight_type == "pnp":
            weight = metric_weight
        if self.feature == "cqt":
            Sy = self.cqt_from_id(id, eps)
            return {'feature': torch.abs(Sy), 'y': y_norm, 'weight': weight, 'M': M,
                    'metric_weight': metric_weight, 'M_mean': self.M_mean, 'JdagJ': JdagJ, 'lambda0':self.lambda0}

    def __len__(self):
        return len(self.ids)

    def M_from_id(self,id):
        #load from h5
        if os.path.exists(self.weights_dir):
            with h5py.File(self.weights_dir, "r") as f:
                M = torch.tensor(np.array(f['M'][str(id)]))
                sigma = torch.abs(torch.tensor(f['sigma'][str(id)]))
                try:
                    JdagJ = torch.tensor(np.array(f['JdagJ'][str(id)]))
                except:
                    JdagJ = None
            return M, sigma, JdagJ
        else:
            return None, None, None


    def make_M_mean(self):
        #load from h5
        M_mean = None
        sigma_mean = None
        lambda_max = 0
        with h5py.File(self.weights_dir, "r") as f:
            ids = f['M'].keys()
            count = 0
            for id in ids:
                M = torch.tensor(np.array(f['M'][str(id)]))
                sigma,_ = torch.sort(torch.abs(torch.tensor(f['sigma'][str(id)])), descending=False)
                M_mean = M if M_mean is None else M_mean + M
                sigma_mean = sigma if sigma_mean is None else sigma_mean + sigma
                if max(sigma) > lambda_max:
                    lambda_max = max(sigma)
                count += 1
        return M_mean / count, sigma_mean / count, lambda_max ** 2


    def noise_inventory(self):
        ids = []
        pitches = []
        # only train / val division
        self.noise_ids = ids
        self.noise_pitches = pitches
        if "nonval" in self.noise_dir:
            with h5py.File(self.noise_dir, "r") as f: # this will be only the nonvalidated data
                for id in f["pitch"].keys():
                    pitch = np.array(f["pitch"][id])
                    ids.append(int(id))
                    pitches.append(pitch)
            # separate train/val/test of noises
            np.random.seed(100)
            rand_idx = np.arange(len(ids))
            np.random.shuffle(rand_idx)
            N_val = int(len(ids) // 10)
            N_train = int(len(ids) - N_val)
            # train test val split (NO TEST SET FROM THIS NOISE DIRECTORY)
            self.train_noise_ids = np.array(ids)[rand_idx[:N_train]]
            self.val_noise_ids = np.array(ids)[rand_idx[-N_val:]]
            self.test_noise_ids = self.val_noise_ids
        else:
            with h5py.File(self.noise_dir, "r") as f: # this will be only the nonvalidated data
                for id in f["x"].keys():
                    ids.append(int(id))
            self.train_noise_ids = np.array(ids)
            self.val_noise_ids = self.train_noise_ids
            self.test_noise_ids = self.val_noise_ids


    def cqt_from_id(self, id, eps):
        with h5py.File(self.audio_dir, "r") as f: # these are synth sounds
            x = np.array(f['x'][str(id)])
            theta = np.array(f['theta'][str(id)])
        # insert code to mix in noise
        if self.isnoise:
            if self.noise_mode != "statgauss":
                #randomly select noise, or the noise with the closest pitch 
                if "test" in self.audio_dir:
                    noise_ids = self.test_noise_ids
                elif "train" in self.audio_dir:                 
                    noise_ids = self.train_noise_ids
                elif "val" in self.audio_dir:       
                    noise_ids = self.val_noise_ids

                idx, idx2 = np.random.choice(np.arange(len(noise_ids)), size=2)
                id1, id2 = noise_ids[idx], noise_ids[idx2]

                with h5py.File(self.noise_dir, "r") as f:
                    noise1 = np.array(f["noise"][str(id1)])
                    noise2 = np.array(f["noise"][str(id2)])
                    sr_og = np.array(f["sr"][str(id1)])
                    sr_og2 = np.array(f["sr"][str(id2)])
                    noise1 = librosa.resample(noise1, orig_sr=sr_og, target_sr=self.sr)
                    noise2 = librosa.resample(noise2, orig_sr=sr_og2, target_sr=self.sr)

                    if self.noise_mode == "gaussian":
                        noise_synth = np.abs(noise1) * (np.random.normal(size=noise1.shape[0])*2 - 1) # extract noise temporal envelope
                        noise_synth = noise_synth / np.max(np.abs(noise_synth))
                        noise = noise_synth
                        start = None
                    elif self.noise_mode == "random":
                        noise = noise1
                        start = None
                    elif self.noise_mode == "mix":
                        # whichever is longer should go as second input
                        if len(noise2) > len(noise1):
                            noise = utils.mix_noise(np.random.rand(), noise1, noise2, mode="weight")
                        else:
                            noise = utils.mix_noise(np.random.rand(), noise2, noise1, mode="weight")
                        start = 0
                #randomly select snr
                snr = np.random.choice([1, 10, 40, 60])
                #mix noise
                x = utils.mix_noise(snr, noise, x, start=start)
            else:
                # temporary jams object to apply the transformation
                jam = jams.JAMS()
                j_orig = muda.jam_pack(jam, _audio=dict(y=x, sr=self.sr))
                for j_new in self.noise_gen.transform(j_orig):
                    x = j_new.sandbox.muda._audio["y"]
                   
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
                 sr,
                 scaler,
                 logscale,
                 feature,
                 num_workers,
                 noise_dir=None,
                 noise_mode="matched"):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.weight_dir = weight_dir
        self.weight_type = weight_type
        self.batch_size = batch_size
        self.feature = feature
        self.J = J
        self.Q = Q
        self.sr = sr
        self.logscale = logscale
        self.full_df = df #sorted df
        self.cqt_dir = cqt_dir
        self.scaler = scaler
        if "am" in weight_dir:
            self.synth_type = "amchirp"
            self.h5name = "amchirp"
        elif "ftm" in weight_dir:
            self.synth_type = "ftm"
            self.h5name = "ftm"
        elif "mersenne" in weight_dir:
            self.synth_type = "string"
            self.h5name = "mersenne24"
        self.noise_dir = noise_dir
        self.noise_mode = noise_mode

    def setup(self, stage=None):
        

        y_norms_train= utils.scale_theta(self.full_df, "train", self.scaler, self.logscale, self.synth_type) #sorted by id
        y_norms_test = utils.scale_theta(self.full_df, "test", self.scaler, self.logscale, self.synth_type)
        y_norms_val = utils.scale_theta(self.full_df, "val", self.scaler, self.logscale, self.synth_type)

        train_ids = self.full_df[self.full_df["fold"]=="train"]['ID'].values #sorted by id
        test_ids = self.full_df[self.full_df["fold"]=="test"]['ID'].values
        val_ids = self.full_df[self.full_df["fold"]=="val"]['ID'].values

        self.train_ds = DrumData(y_norms_train, #partial dataframe
                                train_ids,
                                os.path.join(self.data_dir, self.h5name + "_train_audio.h5"),
                                self.cqt_dir,
                                os.path.join(self.weight_dir, self.h5name + "_train_J.h5"), 
                                self.weight_type,
                                fold='train',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q,
                                sr = self.sr,
                                noise_dir=self.noise_dir,
                                noise_mode=self.noise_mode)

        self.val_ds = DrumData(y_norms_val, #partial dataframe
                                val_ids,
                                os.path.join(self.data_dir, self.h5name + "_val_audio.h5"),
                                self.cqt_dir,
                                os.path.join(self.weight_dir, self.h5name + "_val_J.h5"),
                                self.weight_type,
                                fold='val',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q,
                                sr = self.sr,
                                noise_dir=self.noise_dir,
                                noise_mode=self.noise_mode)

        self.test_ds = DrumData(y_norms_test, #partial dataframe
                                test_ids,
                                os.path.join(self.data_dir, self.h5name + "_test_audio.h5"),
                                self.cqt_dir,
                                os.path.join(self.weight_dir, self.h5name + "_test_J.h5"),
                                self.weight_type,
                                fold='test',
                                feature='cqt',
                                J = self.J,
                                Q = self.Q,
                                sr = self.sr,
                                noise_dir=self.noise_dir,
                                noise_mode=self.noise_mode)


    def collate_batch(self, batch):
        Sy = torch.stack([s['feature'] for s in batch]) #(64,120,257)x
        y = torch.tensor(np.array([s['y'].astype(np.float32) for s in batch]))
        weight = torch.stack([s['weight'] for s in batch])
        try:
            M = torch.stack([s['M'] for s in batch])
        except:
            M = None
        try:
            M_mean = torch.stack([s['M_mean'] for s in batch])
        except:
            M_mean = None
        try:
            metric_weight = torch.stack([s['metric_weight'] for s in batch])           
            JdagJ = torch.stack([s['JdagJ'] for s in batch])
        except: 
            metric_weight, JdagJ = None, None
        try:
            lambda0 = batch[0]['lambda0']
        except:
            lambda0 = None
        return {'feature': Sy, 'y': y, 'weight': weight, 'M': M, 'metric_weight': metric_weight, 'M_mean': M_mean, 'JdagJ': JdagJ, 'lambda0': lambda0}

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
