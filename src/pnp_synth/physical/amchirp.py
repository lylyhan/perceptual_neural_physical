import numpy as np
import torch

def gaussian(M, std, sym=True, device="cuda"):
    ''' Gaussian window converted from scipy.signal.gaussian
    '''
    if M < 1:
        return torch.array([])
    if M == 1:
        return torch.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    n = n.to(device)
    
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w

def generate_am_chirp(theta, logscale, bw=2, duration=4, sr=2**13):
    f_c = 10 ** theta[0] if logscale else theta[0]
    f_m = 10 ** theta[1] if logscale else theta[1]
    gamma = 10 ** theta[2] if logscale else theta[2]
    sigma0 = 0.1
    t = torch.arange(-duration/2, duration/2, 1/sr).to(theta.device)
    carrier = torch.sin(2*np.pi*f_c / (gamma*np.log(2)) * (2 ** (gamma*t) - 1))
    modulator = torch.sin(2 * torch.pi * f_m * t)
    window_std = sigma0 * bw / gamma
    window = gaussian(duration*sr, std=window_std*sr,device=theta.device)
    x = carrier * modulator * window * float(gamma)
    return x