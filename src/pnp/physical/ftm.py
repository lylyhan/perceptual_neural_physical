"""
This script implements Rabenstein's linear drum model
drum model with normalized side length ratio/ side length, in impulse form
"""
import numpy as np
import math

x1 = 0.4
x2 = 0.4
h = 0.03
l0 = np.pi
mode = 10
sr = 22050


def rectangular_drum(m1, m2, x1, x2, h, tau, omega, p, D, l0, alpha, sr):
    l2 = l0 * alpha
    beta_side = alpha + 1 / alpha
    S = l0 / np.pi * ((D * omega * alpha) ** 2 + (p * alpha / tau) ** 2) ** 0.25
    c_sq = (
        alpha * (1 / beta_side - p**2 * beta_side) / tau**2
        + alpha * omega**2 * (1 / beta_side - D**2 * beta_side)
    ) * (l0 / np.pi) ** 2
    T = c_sq
    d1 = 2 * (1 - p * beta_side) / tau
    d3 = -2 * p * alpha / tau * (l0 / np.pi) ** 2

    EI = S**4

    mu = np.arange(1, m1 + 1)
    mu2 = np.arange(1, m2 + 1)
    dur = 2**16
    Ts = 1 / sr
    tau = 1 / sr * np.arange(1, dur + 1)

    n = (mu * np.pi / l0) ** 2 + (mu2 * np.pi / l2) ** 2  # eta
    n2 = n**2
    K = np.sin(mu * math.pi * x1) * np.sin(mu2 * math.pi * x2)

    beta = EI * n2 + T * n  # (m,1)
    alpha = (d1 - d3 * n) / 2  # nonlinear
    omega = np.sqrt(np.abs(beta - alpha**2))
    # correct partials
    mode_corr = np.sum((omega / 2 / np.pi) <= sr / 2)  # convert omega to hz

    N = l0 * l2 / 4
    yi = (
        h
        * np.sin(mu[:mode_corr] * np.pi * x1)
        * np.sin(mu2[:mode_corr] * np.pi * x2)
        / omega[:mode_corr]
    )

    time_steps = np.linspace(0, dur, dur) / sr
    y = np.exp(-alpha[:mode_corr, None] * time_steps[None, :]) * np.sin(
        omega[:mode_corr, None] * time_steps[None, :]
    )
    y = yi[:, None] * y  # (m,) * (m,dur)
    y = np.sum(y * K[:mode_corr, None] / N, axis=0)  # impulse response itself

    return y
