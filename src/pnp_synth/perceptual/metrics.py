from torchmetrics import Metric
from pnp_synth.neural import forward
# kymatio dev0.4.0
#from kymatio.torch import TimeFrequencyScattering as TimeFrequencyScattering1D
#from kymatio.scattering1d.frontend.torch_frontend import TimeFrequencyScatteringTorch as TimeFrequencyScattering1D
# kymatio 0.3.0 (wavespin)
from kymatio.torch import TimeFrequencyScattering1D
from pnp_synth import utils
import torch
import functools
import auraloss
import torch.nn as nn

class JTFSloss(Metric):
    def __init__(self, scaler, mode, synth_type, logscale):
        super().__init__()
        self.add_state("dist", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total",default=torch.tensor(0), dist_reduce_fx="sum")
        self.scaler = scaler
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.curr_device = device
        self.mode = mode
        self.synth_type = synth_type
        self.logscale = logscale

    def update(self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor): #update at each step
        assert preds.shape == target.shape 
        jtfs_params = utils.jtfsparam(self.synth_type)
        jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list").to(self.curr_device)
        jtfs_operator.average_global = True
        phi = functools.partial(utils.S_from_x, jtfs_operator=jtfs_operator)
        g = functools.partial(utils.x_from_theta, synth_type=self.synth_type, logscale=self.logscale)

        #loop over batch - temporary
        wav_gt = []
        wav_pred = []
        for i in range(target.shape[0]):
            wav_gt.append(forward.pnp_forward(target[i,:], 
                                            Phi=phi,
                                            g=g, 
                                            scaler=self.scaler))

            wav_pred.append(forward.pnp_forward(preds[i,:], 
                                            Phi=phi, 
                                            g=g, 
                                            scaler=self.scaler))
        jtfs_targets = torch.stack(wav_gt)
        jtfs_preds = torch.stack(wav_pred) #(bs, #path)

        #check if any output is nan
        if weights is None:
            weights = 1
        if self.mode == "macro":
            self.dist += torch.nanmean(torch.norm(jtfs_preds - jtfs_targets, p=2, dim=1)) #mean over each batch
        elif self.mode == "micro":
            self.dist += torch.nanmean(weights * torch.norm(jtfs_preds - jtfs_targets, p=2, dim=1)).squeeze()
        self.total += 1 #accumulate number of steps (number of batches)
        #return self.dist

    def compute(self):
        return self.dist / self.total #mean over batches
       

class MSSloss(Metric):
    def __init__(self, scaler, synth_type, logscale):
        super().__init__()
        self.add_state("dist", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total",default=torch.tensor(0), dist_reduce_fx="sum")
        self.scaler = scaler
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.curr_device = device
        self.synth_type = synth_type
        self.logscale = logscale

    def update(self, preds: torch.Tensor, target: torch.Tensor): #update at each step

        assert preds.shape == target.shape
        #put through synth ##TODO: need the batch processing!!! or make a loop
        #temporary loop
        wav_gt = []
        wav_pred = []
        g = functools.partial(utils.x_from_theta, synth_type=self.synth_type, logscale=self.logscale)
        for i in range(preds.shape[0]): 
            wav_gt.append(forward.pnp_forward(target[i,:], 
                        Phi=nn.Identity(), 
                        g=g, 
                        scaler=self.scaler))
            wav_pred.append(forward.pnp_forward(preds[i,:], 
                        Phi=nn.Identity(), 
                        g=g, 
                        scaler=self.scaler))
        
        wav_gt = torch.stack(wav_gt)
        wav_pred = torch.stack(wav_pred)
        self.dist += auraloss.freq.MultiResolutionSTFTLoss()(wav_pred.unsqueeze(1), wav_gt.unsqueeze(1))
        self.total += 1 #accumulate number of steps (number of batches)

    def compute(self):
        return self.dist / self.total #mean over batches 
       