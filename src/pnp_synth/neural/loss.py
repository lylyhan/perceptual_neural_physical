import torch
from pnp_synth.neural import forward
import torch.nn as nn
from pnp_synth import utils
#kymatio 0.3.0
from kymatio.torch import TimeFrequencyScattering1D
#kymatio 0.4.0
#from kymatio.torch import TimeFrequencyScattering as TimeFrequencyScattering1D
#from kymatio.scattering1d.frontend.torch_frontend import TimeFrequencyScatteringTorch as TimeFrequencyScattering1D
import functools
import torch.nn.functional as F


class DistanceLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def create_ops(*args, **kwargs):
        raise NotImplementedError

    def dist(self, x, y):
        if self.p == 1.0:
            return torch.abs(x - y).mean()
        elif self.p == 2.0:
            return torch.norm(x - y, p=self.p)

    def forward(self, x, y, transform_y=True):
        loss = torch.tensor(0.0).type_as(x)
        for op in self.ops:
            loss += self.dist(op(x), op(y) if transform_y else y)
        loss /= len(self.ops)
        return loss


class MultiScaleSpectralLoss(DistanceLoss):
    def __init__(
        self,
        max_n_fft=2048,
        num_scales=6,
        hop_lengths=None,
        mag_w=1.0,
        logmag_w=0.0,
        p=1.0,
    ):
        super().__init__(p=p)
        assert max_n_fft // 2 ** (num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = (
            [n // 4 for n in self.n_ffts] if not hop_lengths else hop_lengths
        )
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        self.ops = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i])
            for i, n_fft in enumerate(self.n_ffts)
        ]


class MagnitudeSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).type_as(x),
            return_complex=True,
        ).abs()


class LMAloss(nn.Module):
    #(JTJ)^(-1)J^T(\Phi g (\hat_{\theta}) - \Phi g(\theta))
    #(3 by 1)
    def __init__(self):
        super().__init__()
    
    def create_ops(*args, **kwargs):
        raise NotImplementedError

    def forward(self, x, y, JdagJ):
        loss = torch.tensor(0.0).type_as(x)
        print("dimension of x and y", x.shape, y.shape, JdagJ.shape)
        for op in self.ops:
            diff = op(y=x) - op(y=y)
            print("dimension of diff", diff.shape)
            loss += torch.norm(torch.nansum(JdagJ * diff[:,None,:], dim=-1), p=2)
        loss /= len(self.ops)
        return loss

class TimeFrequencyScatteringLoss(LMAloss):
    def __init__(
        self, scaler
                ):
        super().__init__()
        self.scaler = scaler
        self.create_ops()

    def create_ops(self):
        jtfs_operator = TimeFrequencyScattering1D(**utils.jtfs_params, out_type="list").cuda()
        jtfs_operator.average_global = True
        self.ops = [functools.partial(Phicircg, jtfs_operator=jtfs_operator, scaler=self.scaler)]

def Phicircg(y, jtfs_operator, scaler):
    Ss = []
    for i in range(y.shape[0]):
        theta = forward.inverse_scale(y[i,:], scaler)
        # Synthesis
        x = utils.x_from_theta(theta)
        # Spectral analysis
        S = utils.S_from_x(x, jtfs_operator)
        Ss.append(S.flatten())
    Ss = torch.stack(Ss)
    return Ss

def loss_spec(outputs, y, specloss, scaler):
    #put through synth ##TODO: need the batch processing!!! or make a loop
    #temporary loop
    wav_gt = []
    wav_pred = []
    for i in range(y.shape[0]):
        wav_gt.append(forward.pnp_forward(y[i,:], Phi=nn.Identity(), g=utils.x_from_theta, scaler=scaler))
        wav_pred.append(forward.pnp_forward(outputs[i,:], Phi=nn.Identity(), g=utils.x_from_theta, scaler=scaler))
    wav_gt = torch.stack(wav_gt)
    wav_pred = torch.stack(wav_pred)
    #print("is there any sound", torch.norm(wav_gt), torch.norm(wav_pred), outputs)
    loss = specloss(wav_pred, wav_gt)
    return loss

def loss_bilinear(outputs, y, M):
    diff = outputs - y
    loss = torch.bmm(torch.bmm(diff[:,None,:], M), diff[:,:,None])
    loss = torch.relu(loss) #/1e+5
    return 0.5*torch.mean(loss.squeeze())

def loss_interpolate(outputs, y, M):
    diff = outputs - y
    ploss = F.mse_loss(outputs, y)
    #print("ratio", ploss / torch.norm(y)) 
    if ploss / torch.norm(y) > 0.01:
        return ploss
    else:
        loss = torch.bmm(torch.bmm(diff[:,None,:], M), diff[:,:,None])
        loss = torch.relu(loss) #/1e+5
        return 0.5*torch.mean(loss.squeeze())

def loss_fid(outputs, y):
    mu_pred = torch.mean(outputs,axis=0)
    mu_gt = torch.mean(outputs, axis=0)
    cov_pred = torch.cov(outputs.T)
    cov_gt = torch.cov(y.T)
    CC = torch.matmul(cov_gt.double(), cov_pred.double())
    eigenvals = torch.linalg.eigvals(CC)
    trace_sqrt_CC = eigenvals.real.clamp(min=0).sqrt().sum()
    fid = ((mu_pred - mu_gt) ** 2).sum() + cov_pred.trace() + cov_gt.trace() - 2 * trace_sqrt_CC
    return fid