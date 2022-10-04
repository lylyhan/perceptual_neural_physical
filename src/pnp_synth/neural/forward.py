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
    nus = scaler.transform(theta)
    return nus, scaler


def inverse_scale(nus, scaler):
    "Apply inverse scaling theta = nu * (theta_max - theta_min) + theta_min"
    # NB: we use an explicit formula instead of scaler.inverse_transform
    # so as to preserve PyTorch differentiability.
    theta_max = torch.tensor(scaler.data_max_)
    theta_min = torch.tensor(scaler.data_min_)
    thetas = nus * (theta_max - theta_min) + theta_min
    return thetas


def pnp_forward(Phi, g, scaler, nu):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """
    # Inverse parameter scaling
    theta = inverse_scale(nus, scaler)

    # Synthesis
    x = g(theta)

    # Spectral analysis
    S = Phi(x).flatten().squeeze()
    return S
