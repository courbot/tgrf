#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:54:24 2021

@author: courbot
"""
import numpy as np
from scipy.fft import rfft2 as fft2

import estimators as e
import Gaussian_MRF as gmrf
import unified_functions as uf




class Param_HMC:
    """
    Class representing the parameters for the Hybrid Monte Carlo (HMC) algorithm.

    Attributes:
    sigma_mh -- Metropolis-Hastings proposal standard deviation.
    n_total -- Total number of samples.
    trajectory_length -- Length of the Hamiltonian trajectory.
    timestep -- Size of the timestep.
    nb_rea -- Number of realizations.
    nb_batch -- Number of batches.
    nb_field -- Number of fields.

    Methods:
    __init__ -- Initialize the Param_HMC object.
    """
    sigma_mh = 1 * (2.38 / np.sqrt(100 * 100))
    n_total = 200
    trajectory_length = 30  # T (changed on 02/05/22 from 20)
    timestep = 0.01  # (changed on 02/05/22 from 0.005)
    nb_rea = 10
    nb_batch = 10
    nb_field = 1

    def __init__(self, model):
        """
        Initialize the Param_HMC object.

        Arguments:
        model -- Model type ('single' or 'pair').
        """
        if model == 'single':
            self.nb_field = 1
        elif model == 'pair':
            self.nb_field = 2


class Param_GMRF:
    m = 0
    s = 1
    r = 5
    a = 0
    b = 1
    sig_noise = 0.1
    with_sigma = False

    def __init__(self, argsdim, args=None, sig_noise=None, compute_q=True, gau=True):
        """
        Initialize the Param_GMRF class.

        Parameters:
        -----------
        argsdim : tuple
            The dimensions of the parameters.
        args : array-like, optional
            The parameters. Default value is None.
        sig_noise : float, optional
            The noise standard deviation. Default value is None.
        compute_q : bool, optional
            Flag to compute q. Default value is True.
        gau : bool, optional
            Flag for Gaussian distribution. Default value is True.
        """
        self.lx, self.ly = argsdim

        if args is not None:
            if len(args) == 3:
                self.m, self.s, self.r = args
                self.size = 4
            elif len(args) == 5:
                self.m, self.s, self.r, self.a, self.b = args
                self.size = 6
        else:
            self.m, self.s, self.r = 0, 1, 5
            compute_q = False

        if sig_noise is not None:
            self.sig_noise = sig_noise

        if compute_q:
            if gau:
                self.q = gmrf.get_base_q(self.r, self.s, self.lx, self.ly, "gau")
            else:
                self.q = gmrf.get_base_q(self.r, self.s, self.lx, self.ly, "exp")

    def print(self):
        """Print the parameters."""
        print("m = %.2f" % self.m)
        print("s = %.2f" % self.s)
        print("r = %.2f" % self.r)
        print("a = %.2f" % self.a)
        print("b = %.2f" % self.b)
        print("sig_noise = %.2f" % self.sig_noise)

    def lineprint(self):
        """Print the parameters in a single line."""
        print("m = %.2f, s = %.2f, r = %.2f, a = %.2f, b = %.2f, sig_noise = %.2f" % (
            self.m, self.s, self.r, self.a, self.b, self.sig_noise))

    def toarray(self):
        """Convert the parameters to an array."""
        return np.array([self.m, self.s, self.r, self.a, self.b, self.sig_noise])

    def uniform_sample(self):
        """Sample the parameters uniformly."""
        self.m = -1 + 2 * np.random.random()
        self.s = 0.5 + 10 * np.random.random()
        self.r = 5 + np.random.random() * 10
        self.a = -2 + 4 * np.random.uniform()
        self.b = 0.1 + 2 * np.random.uniform()
        self.sig_noise = 0.01 + np.random.random()

    def est_param_GMRF(self, X, fonc="gau"):
        """Estimate the GMRF parameters."""
        self.lx, self.ly = X.shape[-2], X.shape[-1]
        if X.ndim == 2:
            self.m = e.est_mean(X)
            self.s = e.est_std(X)
            self.r = e.est_r(X, fonc)
            self.r = max(self.r, 3)
        else:
            self.m = e.est_mean(X[0])
            self.s = e.est_std(X[0])
            self.r = e.est_r(X[0], fonc)

        self.q = gmrf.get_base_q(self.r, self.s, self.lx, self.ly, fonc)

    def est_param_from(self, X, Y, a_base, fonc="gau", model='single'):
        """
        Estimate the GMRF parameters from the given data.

        Parameters:
        -----------
        X : array-like
            The data.
        Y : array-like
            The target.
        a_base : array-like
            The base.
        fonc : str, optional
            The function type. Default value is "gau".
        model : str, optional
            The model type. Default value is "single".
        """
        if model != 'cauchy':
            self.est_param_GMRF(X, fonc=fonc)
            if model == 'logit' or model == 'logit_exp':
                self.est_ab_logit(Y, X, fft2(a_base))
            elif model == 'lognorm' or model == 'lognorm_exp':
                self.est_ab_lognorm(Y, X, fft2(a_base))
        else:
            self.est_param_GMRF(X, fonc=fonc)
            self.s /= 10

        self.sig_noise = e.est_sig_bruit_ft(uf.form_Xi(X, model, self), fft2(a_base), Y)

    def est_param_from_Xi(self, Xi, Y, a_base, fonc="gau", model='single'):
        """
        Estimate the GMRF parameters from Xi.

        Parameters:
        -----------
        Xi : array-like
            The data.
        Y : array-like
            The target.
        a_base : array-like
            The base.
        fonc : str, optional
            The function type. Default value is "gau".
        model : str, optional
            The model type. Default value is "single".
        """
        self.sig_noise = e.est_sig_bruit_ft(Xi, fft2(a_base), Y)

        if model == 'logit' or model == 'logit_exp':
            X = uf.logit(Xi, self.a, self.b)
        elif model == 'lognorm' or model == 'lognorm_exp':
            X = uf.loga(Xi, self.a, self.b)
        elif model == 'single' or model == 'single_exp':
            X = Xi.copy()

        self.est_param_GMRF(X, fonc=fonc)

        if model == 'logit' or model == 'logit_exp':
            self.est_ab_logit(Y, X, fft2(a_base))
        elif model == 'lognorm' or model == 'lognorm_exp':
            self.est_ab_lognorm(Y, X, fft2(a_base))

    def est_param_from_ft(self, X, Y, a_base_ft, fonc="gau", model='single'):
        """
        Estimate the GMRF parameters from the Fourier transform.

        Parameters:
        -----------
        X : array-like
            The data.
        Y : array-like
            The target.
        a_base_ft : array-like
            The Fourier transform of the base.
        fonc : str, optional
            The function type. Default value is "gau".
        model : str, optional
            The model type. Default value is "single".
        """
        if model == 'logit' or model == 'logit_exp':
            self.est_ab_logit(Y, X, a_base_ft)
        elif model == 'lognorm' or model == 'lognorm_exp':
            self.est_ab_lognorm(Y, X, a_base_ft)

        self.est_param_GMRF(X, fonc=fonc)

        if not self.with_sigma:
            self.sig_noise = e.est_sig_bruit_ft(uf.form_Xi(X, model, self), a_base_ft, Y)

    def est_ab_lognorm(self, Y, X, a_base_ft):
        """
        Estimate the parameters 'a' and 'b' for the lognormal model.

        Parameters:
        -----------
        Y : array-like
            The target.
        X : array-like
            The data.
        a_base_ft : array-like
            The Fourier transform of the base.
        """
        self.a = 0
        self.b = 1

    def est_ab_logit(self, Y, X, a_base_ft):
        """
        Estimate the parameters 'a' and 'b' for the logistic model.

        Parameters:
        -----------
        Y : array-like
            The target.
        X : array-like
            The data.
        a_base_ft : array-like
            The Fourier transform of the base.
        """
        self.a = 0
        self.b = 1
