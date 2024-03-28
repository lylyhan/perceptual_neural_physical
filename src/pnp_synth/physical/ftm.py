"""
This script implements Rabenstein's linear drum model
drum model with normalized side length ratio/ side length, in impulse form
"""
import numpy as np
import torch

constants = {
    "x1": 0.4,
    "x2": 0.4,
    "h": 0.03,
    "l0": np.pi,
    "m1": 10,
    "m2": 10,
    "sr": 22050,
    "dur":2**16
}


def rectangular_drum(theta, logscale, **constants):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w11 = 10 ** theta[0] if logscale else theta[0]
    p = 10 ** theta[2] if logscale else theta[2]
    D = 10 ** theta[3] if logscale else theta[3]
    #theta
    tau11 = theta[1]
    alpha_side = theta[4]
    l0 = torch.tensor(constants['l0']).to(device)

    l2 = l0 * alpha_side 
    pi = torch.tensor(np.pi, dtype=torch.float64).to(device)

    beta_side = alpha_side + 1 / alpha_side
    S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
    c_sq = (
        alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2 
        + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)
    ) * (l0 / np.pi)**2
    T = c_sq # scalar
    d1 = 2 * (1 - p * beta_side) / tau11
    d3 = -2 * p * alpha_side / tau11 * (l0 / pi) **2 

    EI = S ** 4 

    mu = torch.arange(1, constants['m1'] + 1).to(device) #(m1,)
    mu2 = torch.arange(1, constants['m2'] + 1).to(device) #(m2,)
    dur = constants['dur']
    
    n = (mu[:,None] * pi / l0) ** 2 + (mu2[None,:] * pi / l2)**2 #(m1,m2)
    n2 = n ** 2 
    K = torch.sin(mu[:,None] * pi * constants['x1']) * torch.sin(mu2[None,:] * pi * constants['x2']) #(m1,m2)

    beta = EI * n2 + T * n #(m1, m2)
    alpha = (d1 - d3 * n)/2 # nonlinear
    omega = torch.sqrt(torch.abs(beta - alpha**2))

    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants['sr'] / 2
    mode1_corr = constants['m1'] - max(torch.sum(mode_rejected, dim=0)) if constants['m1']-max(torch.sum(mode_rejected, dim=0))!=0 else constants['m1']
    mode2_corr = constants['m2'] - max(torch.sum(mode_rejected, dim=1)) if constants['m2']-max(torch.sum(mode_rejected, dim=1))!=0 else constants['m2']
    N = l0 * l2 / 4
    yi = (
        constants['h'] 
        * torch.sin(mu[:, None] * pi * constants['x1']) 
        * torch.sin(mu2[None, :] * pi * constants['x2']) 
        / omega #(m1, m2)
    ) 

    time_steps = torch.linspace(0, dur, dur).to(device) / constants['sr'] #(T,)
    y = torch.exp(-alpha[:,:,None] * time_steps[None, None, :]) * torch.sin(
        omega[:,:,None] * time_steps[None,None,:]
    ) # (m1, m2, T)

    y = yi[:,:,None] * y #(m1, m2, T)
    y_full = y * K[:,:,None] / N
    #mode_rejected = mode_rejected.unsqueeze(2).repeat(1,1,y_full.shape[-1])
    y_full = y_full[:mode1_corr, :mode2_corr, :]
    #y_full[mode_rejected] -= y_full[mode_rejected]
    y = torch.sum(y_full, dim=(0,1)) #(T,)
    y = y / torch.max(torch.abs(y))

    return y
