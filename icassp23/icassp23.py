jtfs_params = dict(
    J=14,  # scattering scale ~ 1000 ms
    shape=(2**17,),  # 2**16 of zero padding plus 2**16 of signal
    Q=12,  # number of filters per octave
    T=2**13,  # local temporal averaging
    F=2,  # local frequential averaging
    max_pad_factor=1,  # temporal padding cannot be greater than 1x support
    max_pad_factor_fr=1,  # frequential padding cannot be greater than 1x support
    average=True,  # average in time
    average_fr=True,  # average in frequency
)


def load_dataframe(folds):
    "Load DataFrame corresponding to the entire dataset (100k drum sounds)."
    fold_dfs = {}
    for fold in folds:
        csv_name = fold + "_param_log_v2.csv"
        csv_path = os.path.join("data", csv_name)
        fold_df = pd.read_csv(csv_path)
        fold_dfs[fold] = fold_df

    full_df = pd.concat(fold_dfs.values())
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    return full_df


def x_from_theta(theta):
    """Drum synthesizer, based on the Functional Transformation Method (FTM).
    We apply 2**16 samples of zero padding (~3 seconds) on the left."""
    x = pnp_synth.ftm.rectangular_drum(theta, **pnp_synth.ftm.constants)
    padding = (pnp_synth.ftm.constants["dur"], 0)
    x_padded = torch.nn.functional.pad(x, padding, mode="constant", value=0)
    return x_padded


def S_from_x(jtfs_operator, x):
    """
    Computes log-compressed Joint-Time Frequency Scattering.
    """
    # Sx is a tensor with shape (1, n_paths, n_time_frames)
    Sx = jtfs_operator(x)

    # remove leading singleton dimension and unpad
    Sx_unpadded = Sx[0, :, Sx.shape[-1] :]

    # flatten to shape (n_paths * n_time_frames,)
    Sx_flattened = Sx_unpadded.flatten()

    # apply "stable" log transformation
    log1p_Sx = log1p(Sx)

    return log1p_Sx


def pnp_forward_factory(scaler, jtfs_params):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """
    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params)

    Phi = functools.partial(S_from_theta, jtfs_operator=jtfs_operator)

    return functools.partial(
        pnp_synth.pnp_forward, Phi=S_from_x, g=x_from_theta, scaler=scaler
    )