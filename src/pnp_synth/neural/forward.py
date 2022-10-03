
import torch 

def pnp_forward(Phi, g, scaler, rescaled_param):
    # Inverse parameter scaling: this reimplements scaler.inverse_transform which is not differentiable.
    sc_max = torch.tensor(scaler.data_max_)
    sc_min = torch.tensor(scaler.data_min_)
    theta = rescaled_param * (sc_max - sc_min) + sc_min
    #theta = scaler.inverse_transform(rescaled_param)

    # Synthesis
    x = g(theta)

    # Spectral analysis
    S = Phi(x).flatten().squeeze()
    return S
