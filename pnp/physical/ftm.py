"""
This script implements Rabenstein's linear drum model
drum model with normalized side length ratio/ side length, in impulse form
"""
import numpy as np
import math
import torch

n_samples = 2**16
sr = 22050

def getsounds_imp_linear_nonorm(m1,m2,x1,x2,h,tau11,w11,p,D,l0,alpha_side,sr):
    l2 = l0*alpha_side
    beta_side = alpha_side + 1/alpha_side
    S = l0/np.pi*((D*w11*alpha_side)**2 + (p*alpha_side/tau11)**2)**0.25
    c_sq = (alpha_side*(1/beta_side-p**2*beta_side)/tau11**2 + alpha_side*w11**2*(1/beta_side-D**2*beta_side))*(l0/np.pi)**2
    T = c_sq 
    d1 = 2*(1-p*beta_side)/tau11
    d3 = -2*p*alpha_side/tau11*(l0/np.pi)**2 

    EI = S**4 

    mu = np.arange(1,m1+1) #(0,1,2,..,m-1)
    mu2 = np.arange(1,m2+1)
    dur = n_samples
  

    n = (mu*np.pi/l0)**2+(mu2*np.pi/l2)**2 #eta 
    n2 = n**2 #(mu*np.pi/l0)**4+(mu2*np.pi/l2)**4 
    K = np.sin(mu*math.pi*x1)*np.sin(mu2*math.pi*x2) #mu pi x / l

    beta = EI*n2 + T*n #(m,1)
    alpha = (d1-d3*n)/2 # nonlinear
    omega = np.sqrt(np.abs(beta - alpha**2))
    #correct partials
    mode_corr = np.sum((omega/2/np.pi) <= sr/2) #convert omega to hz
        
    N = l0*l2/4
    yi = h * np.sin(mu[:mode_corr] * np.pi * x1) * np.sin(mu2[:mode_corr] * np.pi * x2) / omega[:mode_corr]


    time_steps = np.linspace(0,dur,dur)/sr
    y = np.exp(-alpha[:mode_corr,None] * time_steps[None,:]) * np.sin(omega[:mode_corr,None] * time_steps[None,:]) 
    y = yi[:,None] * y #(m,) * (m,dur)
    y = np.sum(y * K[:mode_corr,None] / N,axis=0) #impulse response itself

    return y


def getsounds_imp_linear_nonorm_torch(m1,m2,x1,x2,h,theta,l0):
    """
    This implements Rabenstein's drum model. The inverse SLT operation is done at the end of each second-
    -order filter, no normalization on length and side length ratio is done
    note that batch calculation is not allowed since each sound might result in different max allowed mode numbers
    """
    #print(w11,tau11,p,D,l0,alpha_side)
    w11 = theta[:,0]
    tau11 = theta[:,1]
    p = theta[:,2]
    D = theta[:,3]
    l0 = torch.tensor(l0)
    alpha_side = theta[:,4]

    l2 = l0 * alpha_side 
    s11 = -1 / tau11
    pi = torch.tensor(np.pi)

    beta_side = alpha_side + 1 / alpha_side
    S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
    c_sq = (alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2 + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)) * (l0 / np.pi)**2
    T = c_sq 
    d1 = 2 * (1 - p * beta_side) / tau11
    d3 = -2 * p * alpha_side / tau11 * (l0 / pi)**2 

    EI = S**4 

    mu = torch.arange(1,m1+1) #(0,1,2,..,m-1)
    mu2 = torch.arange(1,m2+1)
    dur = n_samples
    n = (mu[None,:] * pi / l0)**2 + (mu2[None,:] * pi / l2[:,None])**2 #eta 
    n2 = n**2 
    K = torch.sin(mu * pi * x1) * torch.sin(mu2 * pi * x2) #mu pi x / l (mode)


    beta = EI[:,None] * n2[None,:] + T[:,None] * n[None,:] #(1,bs,m)
    alpha = (d1[:,None] - d3[:,None] * n[None,:])/2 # nonlinear
    omega = torch.sqrt(torch.abs(beta - alpha**2))

    #insert adaptively change mode number
    temp = (omega/2/pi) <= sr / 2
    mode_corr = torch.sum(temp.to(torch.int32),) #each sample in the batch has its own mode_corr
    
    N = l0 * l2 / 4
    yi = h * torch.sin(mu[None,None,:mode_corr] * pi * x1) * torch.sin(mu2[None,None,:mode_corr] * pi * x2) / omega[:,:,:mode_corr] #(1,bs,mode)

    time_steps = torch.linspace(0,dur,dur) / sr #(T,)
    y = torch.exp(-alpha[:,:,:mode_corr,None] * time_steps[None,None,None,:]) * torch.sin(omega[:,:,:mode_corr,None] * time_steps[None,None,None,:]) # (1,bs,mode,T)

    y = yi[...,None] * y

    y = torch.sum(y * K[None,None,:mode_corr,None] / N[None,:,None,None],axis=-2) #impulse response itself
    y = y / torch.max(y,dim=-1).values[...,None]
    return y