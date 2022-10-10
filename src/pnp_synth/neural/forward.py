import torch
from sklearn.preprocessing import MinMaxScaler


def scale_theta(full_df):
    """
    Take DataFrame, scale training set to [0, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """
    # Fit scaler according to training set only
    train_df = full_df.loc[full_df["set"] == "train"]
    scaler = MinMaxScaler()
    train_thetas = train_df.values[:, 3:-1]
    scaler.fit(train_thetas)

    # Transform whole dataset with scaler
    thetas = full_df.values[:, 3:-1]
    nus = scaler.transform(thetas)
    return nus, scaler.data_max_, scaler.data_min_


def inverse_scale(nu, scaler):
    "Apply inverse scaling theta = nu * (theta_max - theta_min) + theta_min"
    # NB: we use an explicit formula instead of scaler.inverse_transform
    # so as to preserve PyTorch differentiability.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta_max = torch.tensor(scaler.data_max_).to(device)
    theta_min = torch.tensor(scaler.data_min_).to(device)
    theta = nu * (theta_max - theta_min) + theta_min
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
