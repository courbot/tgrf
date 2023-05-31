#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 14:28:48 2022

@author: courbot
"""
import numpy as np
from scipy.fft import rfft2 as fft2
from scipy.fft import irfft2 as ifft2


def form_Xi(X, model, Theta=None):
    """
    Form the transformed parameter Xi based on the model type.

    Parameters:
    -----------
    X : ndarray[float, ndim=2]
        The input parameter.

    model : str
        The model type.

    Theta : object, optional
        Object representing additional parameters.

    Returns:
    --------
    Xi : ndarray[float, ndim=2]
        The transformed parameter.
    """
    if model in ['single', 'single_exp', 'cauchy']:
        return X
    elif model in ['pair', 'pair_exp']:
        return X[0] * np.exp(X[1])
    elif model in ['logit', 'logit_exp']:
        return sigmoid(X, Theta.a, Theta.b)
    elif model in ['lognorm', 'lognorm_exp']:
        return expo(X, Theta.a, Theta.b)
    else:
        print("Error on the model! (form Xi)")

def sigmoid(X, a, b):
    """
    Sigmoid function.

    Parameters:
    -----------
    X : ndarray[float, ndim=2]
        The input values.

    a : float
        Parameter a.

    b : float
        Parameter b.

    Returns:
    --------
    sig : ndarray[float, ndim=2]
        The sigmoid output.
    """
    a_fixed = a
    X_for_exp = np.clip(X, -20, 20)
    return a_fixed + b / (1 + np.exp(-X_for_exp))

def expo(X, a, b):
    """
    Exponential function.

    Parameters:
    -----------
    X : ndarray[float, ndim=2]
        The input values.

    a : float
        Parameter a.

    b : float
        Parameter b.

    Returns:
    --------
    exp_val : ndarray[float, ndim=2]
        The exponential output.
    """
    X_for_exp = np.clip(X, -20, 20)
    return a + b * np.exp(X_for_exp)

def loga(Z, a, b):
    """
    Logarithm function.

    Parameters:
    -----------
    Z : ndarray[float, ndim=2]
        The input values.

    a : float
        Parameter a.

    b : float
        Parameter b.

    Returns:
    --------
    log_val : ndarray[float, ndim=2]
        The logarithm output.
    """
    p = (Z - a) / b
    p[p < 0.01] = 0.01
    return np.log(p)

def logit(X, a, b):
    """
    Logit function.

    Parameters:
    -----------
    X : ndarray[float, ndim=2]
        The input values.

    a : float
        Parameter a.

    b : float
        Parameter b.

    Returns:
    --------
    logit_val : ndarray[float, ndim=2]
        The logit output.
    """
    a_fixed = a
    p = (X - a_fixed) / b
    p[p < 0.0001] = 0.0001
    p[p > 0.9999] = 0.9999
    return np.log(p / (1 - p))

def calc_det(q0):
    """
    Calculate the determinant of a circulant matrix of whose base inverse
    is q0.

    Parameters:
    -----------
    q0 : ndarray[float, ndim=2]
        The input base.

    Returns:
    --------
    det_Q : float
        The absolute value of the determinant of Q.
    """
    lx, ly = q0.shape
    N = lx * ly
    det = 0
    bqf = q0.flatten()
    nth_root = np.exp(np.complex(0, 2 * np.pi) / N)

    for i in range(lx * ly):
        det += bqf[i] * nth_root ** i

    det_Q = np.abs(det)
    return det_Q

def calc_precomputations(Theta, model):
    """
    Calculate precomputed values required for posterior calculation.

    Parameters:
    -----------
    Theta : object
        Object containing model parameters.

    model : str
        The model type.

    Returns:
    --------
    precomputed : tuple
        Tuple of precomputed values.
    """
    qx_ft = fft2(Theta.q)
    dQx = calc_det(Theta.q)
    return qx_ft, dQx

def calc_posterior(Y_ft, X, a_base_ft, Theta, precomputed, model):
    """
    Calculate the posterior probability.

    Parameters:
    -----------
    Y_ft : ndarray[float, ndim=2]
        The Fourier transform of the data.

    X : ndarray[float, ndim=2]
        The parameter values.

    a_base_ft : ndarray[float, ndim=2]
        The Fourier transform of the base measure.

    Theta : object
        Object containing model parameters.

    precomputed : tuple
        Tuple of precomputed values.

    model : str
        The model type.

    Returns:
    --------
    posterior : float
        The posterior probability.
    """
    priors = calc_prior(X - Theta.m, model, precomputed)
    likelihood = calc_likelihood(Y_ft, X, Theta, a_base_ft, model)
    return (priors + likelihood) / X.size

def calc_prior(X, model, precomputed):
    """
    Calculate the prior probability.

    Parameters:
    -----------
    X : ndarray[float, ndim=2]
        The parameter values.

    model : str
        The model type.

    precomputed : tuple
        Tuple of precomputed values.

    Returns:
    --------
    prior : float
        The prior probability.
    """
    qx_ft, dQx = precomputed
    prior = -0.5 * (ifft2(qx_ft * fft2(X)) * X).sum() - 0.5 * np.log(dQx)
    return prior

def calc_likelihood(Y_ft, X, Theta, a_base_ft, model):
    """
    Calculate the likelihood probability.

    Parameters:
    -----------
    Y_ft : ndarray[float, ndim=2]
        The Fourier transform of the data.

    X : ndarray[float, ndim=2]
        The parameter values.

    Theta : object
        Object containing model parameters.

    a_base_ft : ndarray[float, ndim=2]
        The Fourier transform of the base measure.

    model : str
        The model type.

    Returns:
    --------
    likelihood : float
        The likelihood probability.
    """
    Xi = form_Xi(X, model, Theta)
    HX = ifft2(fft2(Xi) * a_base_ft)
    residual = ifft2(Y_ft) - HX
    log_likelihood = -X.size * (0.5 / (Theta.sig_noise ** 2)) * np.linalg.norm(residual) ** 2 - 0.5 * X.size * np.log(2 * np.pi) - X.size * np.log(Theta.sig_noise)
    return log_likelihood

def gradient(X, Y_ft, Theta, a_base_ft, precomputed, model, HR=None):
    """
    Calculate the gradient of the posterior probability.

    Parameters:
    -----------
    X : ndarray[float, ndim=2]
        The parameter values.

    Y_ft : ndarray[float, ndim=2]
        The Fourier transform of the data.

    Theta : object
        Object containing model parameters.

    a_base_ft : ndarray[float, ndim=2]
        The Fourier transform of the base measure.

    precomputed : tuple
        Tuple of precomputed values.

    model : str
        The model type.

    HR : ndarray[float, ndim=2], optional
        Fourier transform of the residual.

    Returns:
    --------
    gradient : ndarray[float, ndim=2]
        The gradient of the posterior probability.
    """
    if model in ['single', 'single_exp']:
        X_ft = fft2(form_Xi(X, model, Theta) - Theta.m)
        qx_ft = precomputed[0]
        Residual_ft = Y_ft - X_ft * a_base_ft
        lp_grad = X.size / (Theta.sig_noise ** 2) * ifft2(Residual_ft * a_base_ft).real
        prior_grad = -ifft2(X_ft * qx_ft).real
    elif model in ['logit', 'logit_exp']:
        X_ft = fft2(form_Xi(X, model, Theta))
        qx_ft = precomputed[0]
        X_for_exp = np.clip(X, -20, 20)
        sigmo = 1 / (1 + np.exp(-X_for_exp))
        sigmo_derivative = Theta.b * sigmo * (1 - sigmo)
        Residual = ifft2(Y_ft - X_ft * a_base_ft).real
        HR = ifft2(a_base_ft * fft2(Residual))
        lp_grad = X.size / (Theta.sig_noise ** 2) * HR * sigmo_derivative
        prior_grad = -ifft2(fft2(X - Theta.m) * qx_ft).real
    elif model in ['lognorm', 'lognorm_exp']:
        X_ft = fft2(form_Xi(X, model, Theta))
        qx_ft = precomputed[0]
        X_for_exp = np.clip(X, -20, 20)
        expo_derivative = Theta.b * np.exp(X_for_exp)
        Residual = ifft2(Y_ft - X_ft * a_base_ft).real
        HR = ifft2(a_base_ft * fft2(Residual))
        lp_grad = X.size / (Theta.sig_noise ** 2) * HR * expo_derivative
        prior_grad = -ifft2(fft2(X - Theta.m) * qx_ft).real
    else:
        print("Error on the model! (gradient)")
    return (-lp_grad - prior_grad) / X.size