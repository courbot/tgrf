#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:23:13 2023

@author: courbot
"""

import numpy as np
from numpy.fft import rfft2 as pfft2
from numpy.fft import irfft2 as pifft2


def sample_Xpost(Y, Theta, fun):
    """
    Sample a GMRF according to the posterior normal distribution
    according to the parameters within Theta.

    Parameters:
    -----------
    Y : array-like
        Observed data.
    Theta : object
        Parameters for the GMRF model.
    fun : str
        Type of correlation function.

    Returns:
    --------
    X : array-like
        Sampled GMRF.
    """

    P, Q = Y.shape
    b = get_base(Theta.r, Theta.s, P, Q, fun)
    q = get_base_invert_numerical(b)
    base_hh = pifft2(Theta.a_base_ft ** 2)
    q_n = q + base_hh / Theta.sig_noise ** 2
    q_n_inv = get_base_invert_numerical(q_n)
    mu_star = calc_Xpost(Y - Theta.m,
                         q_n_inv, Theta.a_base_ft, base_hh,
                         Theta.sig_noise)

    X = mu_star + fourier_sampling_gaussian_field(q_n, Y.shape[0], Y.shape[1]) + Theta.m

    return X


def sample_X(Theta, model):
    """
    Returns X = the X from HX+b=Y, possibly combination of fields.
    Here we should just count the number of fields.

    Parameters:
    -----------
    Theta : object
        Parameters for the GMRF model.
    model : str
        Model type.

    Returns:
    --------
    X : array-like
        Sampled GMRF.
    """
    X = sample_from_fourier(Theta.q, Theta.m, Theta.lx, Theta.ly)
    return X


def calc_Xpost(Y, b_inv, a_base_ft, base_hh, sig_noise):
    """
    Compute the posterior mean of normal distribution of X|Y.

    Parameters:
    -----------
    Y : array-like
        Observed data.
    b_inv : array-like
        Inverse of the base matrix.
    a_base_ft : array-like
        Fourier transform of the base matrix.
    base_hh : array-like
        Fourier transform of the base matrix squared.
    sig_noise : float
        Noise standard deviation.

    Returns:
    --------
    X : array-like
        Posterior mean of X|Y.
    """
    HtY_ft = (a_base_ft * pfft2(Y))  # can be done once outside
    droite_ft = 1 / sig_noise ** 2 * HtY_ft
    X = pifft2(pfft2(b_inv) * (droite_ft)).real
    return X

def get_base_q(range1, sigma, lx, ly, fonc):
    """
    Compute the base matrix 'q' based on the given parameters.

    Parameters:
    -----------
    range1 : float
        Range parameter.
    sigma : float
        Sigma parameter.
    lx : int
        Length of the matrix in the x-direction.
    ly : int
        Length of the matrix in the y-direction.
    fonc : str
        Type of correlation function ('exp' or 'gau').

    Returns:
    --------
    q : array-like
        Base matrix 'q'.
    """
    r = np.zeros((lx, ly))

    for x in range(lx):
        for y in range(ly):
            if fonc == "exp":
                r[x, y] = sigma**2 * corr_exp(0, x, 0, y, lx, ly, range1)
            else:
                r[x, y] = sigma**2 * corr_gau(0, x, 0, y, lx, ly, range1)

    q = get_base_invert_numerical(r)
    return q

def get_base_invert(b):
    '''
    If b is the base of a matrix B, returns bi, the base of B^-1 with the
    direct formula using Fourier space
    '''
    lx, ly = b.shape
    B = np.fft.fft2(b, norm='ortho')
    np.seterr(all='ignore')
    res = 1 / (lx * ly) * np.real(np.fft.ifft2(np.power(B, -1), norm='ortho'))
    np.seterr(all='raise')
    return res

def get_Xpost(a_base_ft, Y, Theta, fun):
    """
    Compute the posterior mean of the normal distribution of X|Y.

    Parameters:
    -----------
    a_base_ft : array-like
        Fourier transform of the base matrix.
    Y : array-like
        Observed data.
    Theta : object
        Parameters for the GMRF model.
    fun : str
        Type of correlation function.

    Returns:
    --------
    X_post : array-like
        Sampled GMRF.
    """
    P, Q = Y.shape
    b = get_base(Theta.r, Theta.s, P, Q, fun)
    q = get_base_invert_numerical(b)
    base_hh = pifft2(a_base_ft ** 2)
    q_n = q + base_hh / Theta.sig_noise ** 2
    q_n_inv = get_base_invert_numerical(q_n)
    X_post = calc_Xpost(Y - Theta.m, q_n_inv, a_base_ft, base_hh, Theta.sig_noise) + Theta.m
    
    return X_post


def get_base(range1, sigma, lx, ly, fonc):
    """
    Compute the base of the covariance matrix given its parameters.

    Parameters:
    -----------
    range1 : float
        Range parameter.
    sigma : float
        Sigma parameter.
    lx : int
        Length of the base in the x-direction.
    ly : int
        Length of the base in the y-direction.
    fonc : str
        Type of correlation function.

    Returns:
    --------
    r : array-like
        Base of the covariance matrix.
    """
    r = np.zeros((lx, ly))
    
    for x in range(lx):
        for y in range(ly):
            if fonc == "exp":
                r[x, y] = sigma ** 2 * corr_exp(0, x, 0, y, lx, ly, range1)
            else:
                r[x, y] = sigma ** 2 * corr_gau(0, x, 0, y, lx, ly, range1)
    return r


def get_base_invert_numerical(b):
    """
    Compute the inverse base matrix with the direct formula using Fourier space.

    Parameters:
    -----------
    b : array-like
        Base of the matrix.

    Returns:
    --------
    res : array-like
        Inverse base matrix.
    """
    lx, ly = b.shape
    bigN = 1.  # np.finfo(float).eps/b.min()
    NB = pfft2(bigN * b, norm='ortho')
    mask = (NB.real > 1e-6)
    iNB = np.zeros_like(NB)
    iNB[mask] = np.power(NB[mask], -1)
    if (mask).sum() != 0:
        iNB[mask == 0] = iNB[mask].max()
    res = 1 / (lx * ly * bigN) * np.real(pifft2(iNB, norm='ortho'))
    return res


def fourier_sampling_gaussian_field(r, lx, ly):
    """
    Sample a GMRF given its base r.

    Parameters:
    -----------
    r : array-like
        Base matrix.
    lx : int
        Length of the GMRF in the x-direction.
    ly : int
        Length of the GMRF in the y-direction.

    Returns:
    --------
    Y : array-like
        Sampled GMRF.
    """
    X = np.zeros((lx, ly), dtype=int)
    Z = np.random.normal(X, 1)
    Z = Z.astype(np.complex)
    Z += 1j * np.random.normal(X, 1)
    Z = Z.reshape((lx, ly))
    L = np.fft.fft2(r)
    Y = np.real(np.fft.fft2(np.multiply(np.power(L, -0.5), Z), norm="ortho"))
    return Y


def sample_from_fourier(qb, mb, lx, ly):
    """
    Sample a GMRF given the base matrix, mean, and dimensions.

    Parameters:
    -----------
    qb : array-like
        Base matrix.
    mb : array-like
        Mean of the GMRF.
    lx : int
        Length of the GMRF in the x-direction.
    ly : int
        Length of the GMRF in the y-direction.

    Returns:
    --------
    X : array-like
        Sampled GMRF.
    """
    X = mb + fourier_sampling_gaussian_field(qb, lx, ly)
    return X


def euclidean_dist_torus(x1, x2, y1, y2, lx, ly):
    """
    Compute the distance on the torus between (x1,y1) and (x2,y2).

    Parameters:
    -----------
    x1 : int
        x-coordinate of the first point.
    x2 : int
        x-coordinate of the second point.
    y1 : int
        y-coordinate of the first point.
    y2 : int
        y-coordinate of the second point.
    lx : int
        Length of the torus in the x-direction.
    ly : int
        Length of the torus in the y-direction.

    Returns:
    --------
    dist : float
        Euclidean distance on the torus.
    """
    return np.sqrt(min(np.abs(x1 - x2), lx - np.abs(x1 - x2)) ** 2 +
                   min(np.abs(y1 - y2), ly - np.abs(y1 - y2)) ** 2)


def corr_exp(x1, x2, y1, y2, lx, ly, r):
    """
    Point-wise exponential correlation.

    Parameters:
    -----------
    x1 : int
        x-coordinate of the first point.
    x2 : int
        x-coordinate of the second point.
    y1 : int
        y-coordinate of the first point.
    y2 : int
        y-coordinate of the second point.
    lx : int
        Length of the base in the x-direction.
    ly : int
        Length of the base in the y-direction.
    r : float
        Range parameter.

    Returns:
    --------
    corr : float
        Exponential correlation between two points.
    """
    try:
        return np.exp(-euclidean_dist_torus(x1, x2, y1, y2, lx, ly) / r)
    except FloatingPointError:
        return 0


def corr_gau(x1, x2, y1, y2, lx, ly, r):
    """
    Point-wise Gaussian correlation.

    Parameters:
    -----------
    x1 : int
        x-coordinate of the first point.
    x2 : int
        x-coordinate of the second point.
    y1 : int
        y-coordinate of the first point.
    y2 : int
        y-coordinate of the second point.
    lx : int
        Length of the base in the x-direction.
    ly : int
        Length of the base in the y-direction.
    r : float
        Range parameter.

    Returns:
    --------
    corr : float
        Gaussian correlation between two points.
    """
    try:
        return np.exp(-euclidean_dist_torus(x1, x2, y1, y2, lx, ly) ** 2 / r ** 2)
    except FloatingPointError:
        return 0


def gau(x, r):
    """
    Gaussian function.

    Parameters:
    -----------
    x : float
        Input value.
    r : float
        Range parameter.

    Returns:
    --------
    val : float
        Gaussian value.
    """
    return np.exp(-(x ** 2) / r ** 2)


def exp(x, r):
    """
    Exponential function.

    Parameters:
    -----------
    x : float
        Input value.
    r : float
        Range parameter.

    Returns:
    --------
    val : float
        Exponential value.
    """
    return np.exp(-np.abs(x) / r)