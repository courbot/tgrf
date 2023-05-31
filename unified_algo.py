#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:04:16 2022

@author: courbot
"""

# external
from scipy.fft import rfft2 as fft2
from scipy.fft import irfft2 as ifft2
import scipy.special as sp
import scipy.ndimage as ndi
import scipy.stats as st
import numpy as np

# homemade
import unified_HMC as uhmc
import unified_SEM as usem
import Parameters as p
import Gaussian_MRF as gmrf
import unified_functions as uf

    
# =============================================================================
# Run the proposed algorithms
# =============================================================================

def run_exp(Y, a_base, inference_model, EM=False, verbose=False):
    """
    Run the experiment for the given parameters.

    Parameters:
    -----------
    Y : ndarray[float, ndim=2]
        The observed data.
    a_base : ndarray[float, ndim=2]
        The base signal.
    inference_model : str
        The type of inference model to use.
    EM : bool, optional
        Whether to use the EM algorithm. Default is False.
    verbose : bool, optional
        Whether to print verbose output during the execution. Default is False.

    Returns:
    --------
    Xi_est : ndarray[float, ndim=2]
        The estimated latent variable.
    sig_noise : float
        The estimated noise standard deviation.
    """
    par_HMC = p.Param_HMC(inference_model)
    par_HMC.n_total = 100
    par_HMC.nb_batch = 20
    par_HMC.nb_rea = 10

    Theta_SEM, theta_series, lp, not_converged, X_SEM = usem.unified_SEM(
        Y, a_base, par_HMC, inference_model, nb_iter_max=30, nb_iter_min=10, verbose=verbose, EM=EM
    )

    if inference_model == 'single' or inference_model == 'single_exp':
        Xi_est = direct_inference_single(a_base, Y, Theta_SEM)
    else:
        par_HMC = p.Param_HMC(inference_model)
        par_HMC.n_total = 100
        par_HMC.nb_batch = 40
        par_HMC.nb_rea = 10

        MH_all_X, X_seq, lp_all, burnin = uhmc.sample_HMC_par_autostop(
            Y, Theta_SEM, a_base, inference_model, par_HMC, supervized=True, X_start=X_SEM, verbose_all=verbose
        )
        X_est = MH_all_X[0]

        Xi_est = uf.form_Xi(X_est, inference_model, Theta_SEM)

    return Xi_est, Theta_SEM.sig_noise


def direct_inference_single(a_base, Y, Theta):
    """
    Perform direct inference for a single component model.

    Parameters:
    -----------
    a_base : ndarray[float, ndim=2]
        The base signal.
    Y : ndarray[float, ndim=2]
        The observed data.
    Theta : object
        The parameter object containing the necessary parameters.

    Returns:
    --------
    Xpost : ndarray[float, ndim=2]
        The posterior estimate of the latent variable.
    """
    a_base_ft = fft2(a_base)
    base_hh = ifft2(fft2(a_base) ** 2)
    b = Theta.q + base_hh / Theta.sig_noise ** 2
    b_inv = gmrf.get_base_invert(b)

    Xpost = gmrf.calc_Xpost(Y, b_inv, a_base_ft, base_hh, Theta.sig_noise)
    return Xpost

# =============================================================================
# 
# =============================================================================


def sample_Xpost(a_base, Y, Theta):
    """
    Sample from the posterior distribution of the latent variable.

    Parameters:
    -----------
    a_base : ndarray[float, ndim=2]
        The base signal.
    Y : ndarray[float, ndim=2]
        The observed data.
    Theta : object
        The parameter object containing the necessary parameters.

    Returns:
    --------
    X : ndarray[float, ndim=2]
        The sampled latent variable.
    """
    base_hh = ifft2(fft2(a_base) ** 2)
    b0 = Theta.q + base_hh / Theta.sig_noise ** 2
    mu_star = direct_inference_single(a_base, Y, Theta)

    X = mu_star + gmrf.fourier_sampling_gaussian_field(b0, Y.shape[0], Y.shape[1])
    return X


def calc_residu(Y, a_base, X_est, model):
    """
    Calculate the residual between the observed data and the reconstructed signal.

    Parameters:
    -----------
    Y : ndarray[float, ndim=2]
        The observed data.
    a_base : ndarray[float, ndim=2]
        The base signal.
    X_est : ndarray[float, ndim=2]
        The estimated latent variable.
    model : object
        The model object containing the necessary parameters.

    Returns:
    --------
    std : float
        The standard deviation of the residual.
    """
    Recon = uf.form_Xi(X_est, model)
    resi = Y - ifft2(fft2(Recon) * fft2(a_base)).real
    return resi.std()



def run_oracle(Y, a_base, inference_model, Theta_true, verbose=False):
    """
    Run the oracle (supervized) algorithm for the given parameters.

    Parameters:
    -----------
    Y : ndarray[float, ndim=2]
        The observed data.
    a_base : ndarray[float, ndim=2]
        The base signal.
    inference_model : object
        The inference model object.
    Theta_true : object
        The true parameter object.
    verbose : bool, optional
        Whether to print verbose output during the execution. Default is False.

    Returns:
    --------
    Xi_est : ndarray[float, ndim=2]
        The estimated latent variable after running the oracle algorithm.
    """
    par_HMC = p.Param_HMC(inference_model)
    par_HMC.n_total = 100
    par_HMC.nb_batch = 40
    par_HMC.nb_rea = 20

    X_start = ndi.gaussian_filter(Y, 5)

    MH_all_X, X_seq, lp_all, burnin = uhmc.sample_HMC_par_autostop(Y, Theta_true,
                                                                   a_base, inference_model, par_HMC,
                                                                   supervized=True, X_start=X_start, verbose_all=verbose)
    X_est = MH_all_X[0]

    Xi_est = uf.form_Xi(X_est, inference_model, Theta_true)

    return Xi_est


# =============================================================================
# Image simulations for experiments
# =============================================================================

def simul_XY_onion(seed, airy_scale, SNR):
    """
    Simulate Xi, Y, and Airy pattern.

    Parameters:
    -----------
    seed : int
        Seed value for random number generation.
    airy_scale : float
        Scaling factor for the Airy pattern.
    SNR : float
        Signal-to-noise ratio in decibels.

    Returns:
    --------
    ndarray, ndarray, ndarray, float
        Y data, Xi data, Airy pattern base, and noise standard deviation.

    """

    lx, ly = 100, 100
    dim = (lx, ly)

    a_base = get_airy(lx, ly, scale=airy_scale)
    list_model = ('lognorm_exp', 'lognorm', 'logit', 'logit_exp', 'single_exp', 'single')

    np.random.seed(seed)

    Xlist = []
    sigma_GMRF = 1
    rx = 5 + 10 * np.random.random()
    sx = sigma_GMRF
    mx = 0
    a = 0
    b = 3

    for simul_model in list_model:
        np.random.seed(seed)
        if simul_model == 'lognorm_exp' or simul_model == 'lognorm':
            sx = 1
        if simul_model == 'logit' or simul_model == 'logit_exp':
            sx = 5
        if simul_model == 'single' or simul_model == 'single_exp':
            sx = 1

        arg = (mx, sx, rx)
        if simul_model == 'logit' or simul_model == 'lognorm' or simul_model == 'single':
            Theta_true = p.Param_GMRF(dim, arg, gau=True)
        else:
            Theta_true = p.Param_GMRF(dim, arg, gau=False)

        Theta_true.a = a
        Theta_true.b = b

        X = gmrf.sample_X(Theta_true, simul_model)
        Xi = uf.form_Xi(X, simul_model, Theta_true)
        Xi = (Xi - Xi.min()) / (Xi.max() - Xi.min())
        Xlist.append(Xi)

    dx, dy = np.mgrid[:lx, :ly]
    centre = np.array([int(lx / 2), int(ly / 2)])
    distance = np.sqrt((dx - centre[0])**2 + (dy - centre[1])**2)

    liste_r = (50, 43, 35, 25, 15)  # calibrated for 100-px images
    Xi = Xlist[0].copy()
    for i in range(5):
        Xi[distance < liste_r[i]] = Xlist[i + 1][distance < liste_r[i]]

    sig_noise = np.sqrt(1 / (lx * ly) * np.linalg.norm(Xi)**2 * 10**(-SNR / 10))

    Hx = np.fft.ifft2(np.fft.fft2(Xi) * np.fft.fft2(a_base)).real
    Y = Hx + st.norm.rvs(size=(Xi.shape[0], Xi.shape[1]), scale=sig_noise)

    return Xi, Y, a_base


def simul_XY(seed, airy_scale, SNR, sigma_GMRF, simul_model, lx=100, ly=100):
    """
    Simulate X and Y data.

    Parameters:
    -----------
    seed : int
        Seed value for random number generation.
    airy_scale : float
        Scaling factor for the Airy pattern.
    SNR : float
        Signal-to-noise ratio in decibels.
    sigma_GMRF : float
        Standard deviation for the GMRF model.
    simul_model : str
        Simulation model type.
    lx : int, optional
        Width of the data grid (default is 100).
    ly : int, optional
        Height of the data grid (default is 100).

    Returns:
    --------
    ndarray, ndarray, ndarray, Param_GMRF
        X data, Y data, Airy pattern base, and true parameter values.

    """

    dim = (lx, ly)

    a_base = get_airy(lx, ly, scale=airy_scale)

    np.random.seed(seed)

    rx = 5 + 10 * np.random.random()
    sx = sigma_GMRF
    mx = 0
    arg = (mx, sx, rx)
    if simul_model == 'logit' or simul_model == 'lognorm' or simul_model == 'single':
        Theta_true = p.Param_GMRF(dim, arg, gau=True)
    else:
        Theta_true = p.Param_GMRF(dim, arg, gau=False)

    # legacy - to be removed
    Theta_true.a = 0
    Theta_true.b = 1

    X = gmrf.sample_X(Theta_true, simul_model)

    Xi = uf.form_Xi(X, simul_model,Theta_true)
    Theta_true.sig_noise = np.sqrt(1 / (lx * ly) * np.linalg.norm(Xi)**2 * 10**(-SNR / 10))

    Hx = np.fft.ifft2(np.fft.fft2(Xi) * np.fft.fft2(a_base)).real
    Y = Hx + st.norm.rvs(size=(Xi.shape[0], Xi.shape[1]), scale=Theta_true.sig_noise)

    return X, Y, a_base, Theta_true

def get_airy(lx, ly, scale=2):
    """
    Generate an Airy pattern.

    Parameters:
    -----------
    lx : int
        Width of the pattern.
    ly : int
        Height of the pattern.
    scale : float, optional
        Scaling factor for the pattern (default is 2).

    Returns:
    --------
    ndarray
        The generated Airy pattern.

    Example:
    --------
    a_base = get_airy(256, 256, scale=2)
    """

    dx, dy = np.mgrid[:lx, :ly]
    dist = np.sqrt((dx - lx/2)**2 + (dy - ly/2)**2) / scale

    airy = np.zeros_like(dist)
    airy[dist != 0] = 4 * (sp.j1(dist[dist != 0]) / dist[dist != 0])**2
    airy[dist == 0] = 1
    airy /= airy.sum()

    a_base = np.fft.fftshift(airy)
    return a_base


def sample_Y_from_Xi(Xi, a_base, sig_noise):
    """
    Sample Y = HX + b.

    Parameters:
    -----------
    Xi : ndarray
        Input data.
    a_base : ndarray
        PSF.
    sig_noise : float
        Standard deviation of the noise.

    Returns:
    --------
    ndarray
        Sampled Y.

    Example:
    --------
    Y = sample_Y_from_Xi(Xi, a_base, sig_noise=0.1)
    """

    Hx = np.fft.ifft2(np.fft.fft2(Xi) * np.fft.fft2(a_base)).real

    Y = Hx + st.norm.rvs(size=(Xi.shape[0], Xi.shape[1]), scale=sig_noise)
    
    return Y