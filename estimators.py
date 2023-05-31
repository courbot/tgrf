#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:15:46 2021

@author: courbot
"""
import numpy as np
import scipy.signal as si
from scipy.fft import rfft2 as pfft2
from scipy.fft import irfft2 as pifft2
from scipy.optimize import curve_fit

def est_sig_bruit(X, a_base, Y):
    """
    Estimate the noise standard deviation.

    Parameters:
    -----------
    X : array-like
        The data.
    a_base : array-like
        The base.
    Y : array-like
        The target.

    Returns:
    --------
    float
        The estimated noise standard deviation.
    """
    lx, ly = X.shape
    Hx = pifft2(pfft2(a_base) * pfft2(X.reshape(lx, ly))).real
    sig_noise = (Y - Hx).std()

    # Sanity check to ensure a reasonable estimate
    # sig_noise = max(Y.std() / 20, sig_noise)
    # sig_noise = min(Y.std() / 2, sig_noise)

    return sig_noise


def est_sig_bruit_ft(X, a_base_ft, Y):
    """
    Estimate the noise standard deviation from the Fourier transform.

    Parameters:
    -----------
    X : array-like
        The data.
    a_base_ft : array-like
        The Fourier transform of the base.
    Y : array-like
        The target.

    Returns:
    --------
    float
        The estimated noise standard deviation.
    """
    lx, ly = X.shape
    Hx = pifft2(a_base_ft * pfft2(X.reshape(lx, ly))).real
    sig_noise = (Y - Hx).std()

    sig_noise = min(sig_noise, Y.std())

    # Sanity check to ensure a reasonable estimate
    # sig_noise = max(Y.std() / 20, sig_noise)
    # sig_noise = min(Y.std() / 2, sig_noise)

    return sig_noise


def est_std(X):
    """
    Estimate the standard deviation of the data.

    Parameters:
    -----------
    X : array-like
        The data.

    Returns:
    --------
    float
        The estimated standard deviation.
    """
    return max(X.std(), 0.1)


def est_mean(X):
    """
    Estimate the mean of the data.

    Parameters:
    -----------
    X : array-like
        The data.

    Returns:
    --------
    float
        The estimated mean.
    """
    return X.mean()


def gau(x, r):
    """
    Gaussian function.

    Parameters:
    -----------
    x : array-like
        Input values.
    r : float
        Parameter.

    Returns:
    --------
    array-like
        Output values.
    """
    return np.exp(-(x**2) / r**2)


def exp(x, r):
    """
    Exponential function.

    Parameters:
    -----------
    x : array-like
        Input values.
    r : float
        Parameter.

    Returns:
    --------
    array-like
        Output values.
    """
    return np.exp(-np.abs(x) / r)


def est_r(X, fonc="gau"):
    """
    Estimate the parameter 'r' using autocorrelation.

    Parameters:
    -----------
    X : array-like
        The input data.
    fonc : str, optional
        Function name to fit ('gau' for Gaussian, 'exp' for exponential).

    Returns:
    --------
    float
        The estimated parameter 'r'.
    """
    lx, ly = X.shape
    mx = X.mean()
    

    # Compute the autocorrelation
    autocorr = si.fftconvolve(X - mx, X[::-1, ::-1] - mx, mode='same')[::-1, ::-1]
    autocorr /= autocorr.max()

    C = np.zeros(shape=(lx))
    nb_pt = np.zeros_like(C)
    dx, dy = np.ogrid[:lx, :ly]

    D = np.sqrt((dx - lx/2 + 1)**2 + (dy - ly/2 + 1)**2)
    D = np.round(D).astype(int)

    for a in range(autocorr.shape[0]):
        for b in range(autocorr.shape[1]):
            index = D[a, b]
            C[index] += autocorr[a, b]
            nb_pt[index] += 1

    lim = int(lx/2)
    correlogram = C[:lim] / nb_pt[:lim]

    where_zero = np.where(correlogram <= 0)[0]
    if len(where_zero) == 0:
        l = int(lx/2)
    else:
        l = where_zero[0]

    dxc = np.arange(correlogram.size)

    # Unewpexted behaviour : underfit in exponentials below.
    # To avoid this, set a reasonable initial value (5 here).
    if fonc == "gau":
        r_est, _ = curve_fit(gau, dxc[:l], correlogram[:l],p0=5)
    else:
        r_est, _ = curve_fit(exp, dxc[:l], correlogram[:l],p0=5)

    # Adjust the estimated parameter 'r'
    r_est = min(max(r_est[0], 4), 12)

    return r_est

