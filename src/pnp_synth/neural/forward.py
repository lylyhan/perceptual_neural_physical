import torch
from sklearn.preprocessing import MinMaxScaler


def inverse_scale(nu, scaler):
    "Apply inverse scaling theta = nu * (theta_max - theta_min) + theta_min"
    # NB: we use an explicit formula instead of scaler.inverse_transform
    # so as to preserve PyTorch differentiability.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta_max = torch.tensor(scaler.data_max_).to(device)
    theta_min = torch.tensor(scaler.data_min_).to(device)
    theta = (nu + 1) / 2 * (theta_max - theta_min) + theta_min
    return theta


def pnp_forward(nu, Phi, g, scaler):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a scaler h
    2. a synthesizer g
    3. a perceptual representation Phi
    """
    # Inverse parameter scaling
    theta = inverse_scale(nu, scaler)

    # Synthesis
    x = g(theta)
    # Spectral analysis
    S = Phi(x)
    return S
