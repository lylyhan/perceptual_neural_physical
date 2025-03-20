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

constants_string = {
    "x": 0.1, # adapt to absolute position in meters
    "h": 0.03,
    "l0": np.pi,
    "m": 20,
    "sr": 22050,
    "dur":2**17
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

def physics2percep(S4, T, d1, d3, l, lm):
    c2 = T/lm
    sigma1 = d3/(2*lm) * (np.pi/l)**2 - d1/(2*lm)
    tau1 = 1/sigma1
    w1 = np.sqrt((S4-d3**2/(4*lm**2))*(np.pi/l)**4 + (c2 + d1*d3/(2*lm**2)) * (np.pi/l)**2 - d1**2/(4*lm**2))
    p = d3 * tau1 / (2*lm) * (np.pi/l)**2
    D = np.sqrt(S4 * (np.pi/l)**4 - (p*sigma1)**2) / w1
    return w1, tau1, p, D

def percep2physics(w1, tau1, p, D, l, lm):
    # convert perceptual parameters to PDE parameters
    #d1 = 2 * (1 - p * lm * (l/np.pi)**2 ) / tau1 # wrong
    d3 = 2 * p * lm * l**2 / (tau1 * np.pi**2)
    d1 = -2 * lm / tau1 + d3 * (np.pi/l)**2
    S4 = (l/np.pi)**4 * ((D*w1)**2 + (p/tau1)**2)
    c2 = (l/np.pi)**2 * (w1**2 * (1-D**2) + (1-p**2)/tau1**2)
    return d1, d3, S4, c2

# theta = {w1,tau1, p, D, lm, ell}
def linearstring_percep(theta, logscale, **constants_string):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # convert omega, tau, p, D into S, c, d1, d3
    w11 = 10 ** theta[0] if logscale else theta[0]
    p = 10 ** theta[2] if logscale else theta[2]
    D = 10 ** theta[3] if logscale else theta[3]
    #theta
    tau11 = theta[1]
    lm = theta[4]
    ell = theta[5]
    pi = torch.tensor(np.pi, dtype=torch.float64).to(device)
    dur = constants_string['dur']

    d1, d3, S4, c2 = percep2physics(w11, tau11, p, D, ell, lm)
    d1 = abs(d1)
    EI = S4 * lm
    Ts0 = c2 * lm

    mu = torch.arange(1, constants_string["m"] + 1).to(device)
    n = (mu * pi / ell) ** 2 
    n2 = n ** 2 
    K = torch.sin(mu * pi * constants_string["x"])

    beta = EI * n2 + Ts0 * (-n) #(m)
    alpha = (d1 + d3 * n)/(2*lm) # nonlinear
    omega = torch.sqrt(torch.abs(beta/(lm) - alpha**2))
    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants_string['sr'] / 2
    mode_corr = constants_string['m'] - torch.sum(mode_rejected)
   
    N = ell / 2
    yi = (
        constants_string['h']
        * torch.sin(mu * pi * constants_string["x"]) #this should be the listening position
        / omega #(mode)
    )
  
    time_steps = torch.linspace(0, dur, dur).to(device) / constants_string['sr'] #(T,)

    y = torch.exp(-alpha[:,None] * time_steps[ None, :]) * torch.sin(
        omega[:,None] * time_steps[None,:]
    ) # (m, T)

    y = yi[:, None] * y #(m, T)
    y_full = y * K[:,None] / N
    y_full = y_full[:mode_corr, :]
    y = torch.sum(y_full, dim=0) #(T,)
    y = y / torch.max(torch.abs(y))


    return y



#theta:{EI, T, d1, d3, lm, ell}
def linearstring_physics(theta, **constants_string):
    """
    unlike the convention in rabenstein's paper. d3 is always positive, so alpha=(d1+d3*n)/(2*lm)
    beta = EI n2 + Ts0 n (positive sign here)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # convert omega, tau, p, D into S, c, d1, d3
    EI = theta[0]
    Ts0 = 10 ** theta[1]
    d1 = theta[2]
    d3 = theta[3]
    lm = 10 ** theta[4]
    ell = 10 ** theta[5]
    pi = torch.tensor(np.pi, dtype=torch.float64).to(device)
    dur = constants_string['dur']
    pos_ratio = torch.rand(1).to(device) / 2 #randomly sample between 0 and 0.5


    mu = torch.arange(1, constants_string["m"] + 1).to(device)
    n = (mu * pi / ell) ** 2 
    n2 = n ** 2 
    K = torch.sin(mu * pi * pos_ratio)
  
    beta = EI * n2 + Ts0 * n #(m)
    alpha = (d1 + d3 * n)/(2*lm) # nonlinear
    # TODO: there should be constraint in how alpha should be, in case it exceeds beta!!!
    
    omega = torch.sqrt(beta/(lm) - alpha**2)
    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants_string['sr'] / 2
    mode_corr = constants_string['m'] - torch.sum(mode_rejected)
    if torch.sum(torch.isnan(omega)) > 0 or torch.min(omega) > 1200 * 2 * np.pi:
        return "exceeded pitch range"
    else:
        N = ell / 2
        yi = (
            constants_string['h']
            * torch.sin(mu * pi * pos_ratio) #this should be the listening position
            / omega #(mode)
        )
    
        time_steps = torch.linspace(0, dur, dur).to(device) / constants_string['sr'] #(T,)

        y = torch.exp(-alpha[:,None] * time_steps[ None, :]) * torch.sin(
            omega[:,None] * time_steps[None,:]
        ) # (m, T)

        y = yi[:, None] * y #(m, T)
        y_full = y * K[:,None] / N
        y_full = y_full[:mode_corr, :]
        y = torch.nansum(y_full, dim=0) #(T,)
        y = y / torch.max(torch.abs(y))


        return y
