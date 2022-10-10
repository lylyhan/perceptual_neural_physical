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


def rectangular_drum(theta, **constants):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #theta
    w11 = 10 ** theta[0]
    tau11 = theta[1]
    p = 10 ** theta[2]
    D = 10 ** theta[3]
    alpha_side = theta[4]
    l0 = torch.tensor(constants['l0'])

    l2 = l0 * alpha_side 
    pi = torch.tensor(np.pi, dtype=torch.float64)

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

    mu = torch.arange(1, constants['m1'] + 1).to(device) #(m,)
    mu2 = torch.arange(1, constants['m2'] + 1).to(device) #(m,)
    dur = constants['dur']
    n = (mu * pi / l0) ** 2 + (mu2 * pi / l2)**2 #(m,)
    n2 = n ** 2 
    K = torch.sin(mu * pi * constants['x1']) * torch.sin(mu2 * pi * constants['x2']) #(m,)


    beta = EI * n2 + T * n #(m)
    alpha = (d1 - d3 * n)/2 # nonlinear
    omega = torch.sqrt(torch.abs(beta - alpha**2))

    #adaptively change mode number according to nyquist frequency
    temp = (omega / 2 / pi) <= constants['sr'] / 2
    mode_corr = torch.sum(temp.to(torch.int32),)
    
    N = l0 * l2 / 4
    yi = (
        constants['h'] 
        * torch.sin(mu[:mode_corr] * pi * constants['x1']) 
        * torch.sin(mu2[:mode_corr] * pi * constants['x2']) 
        / omega[:mode_corr] #(mode)
    )

    time_steps = torch.linspace(0, dur, dur).to(device) / constants['sr'] #(T,)
    y = torch.exp(-alpha[:mode_corr,None] * time_steps[None,:]) * torch.sin(
        omega[:mode_corr,None] * time_steps[None,:]
    ) # (mode,T)

    y = yi[:,None] * y #(mode,T)

    y = torch.sum(y * K[:mode_corr,None] / N,axis=0) #(T,)

    y = y / torch.max(torch.abs(y))

    return y