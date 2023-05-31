#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:59:31 2022

@author: courbot
"""

import scipy.ndimage as ndi
import time
import numpy as np

import Parameters as p
import unified_HMC as uhmc
import unified_functions as uf
import unified_algo as ua

def init_SEM(Y, a_base, model, fonc, par_HMC=None):
    """
    Initialize the Stochastic Expectation-Maximization (SEM) parameters and estimate Xi_est.

    Parameters:
    -----------
    Y : array-like
        Input data.
    a_base : array-like
        Base matrix.
    model : str
        SEM model type.
    fonc : str
        Function name.
    par_HMC : None, optional
        Parameters for the Hybrid Monte Carlo (HMC) method.

    Returns:
    --------
    Theta : object
        Initialized SEM parameters.
    Xi_est : array-like
        Estimated Xi values.
    """

    if model == 'single' or model == 'single_exp':
        # Apply Gaussian filtering to the input data
        Xi_est = ndi.gaussian_filter(Y, 1)

    else:
        if model == 'logit' or model == 'lognorm':
            # Run the exponential function with EM estimation
            Xi_est_single, sn = ua.run_exp(Y, a_base, 'single', EM=True)
        elif model == 'logit_exp' or model == 'lognorm_exp' or model == 'cauchy':
            # Run the exponential function with EM estimation
            Xi_est_single, sn = ua.run_exp(Y, a_base, 'single_exp', EM=True)

        if model == 'logit' or model == 'logit_exp':
            # Apply the logistic function to Xi_est_single
            Xi_est = uf.logit(Xi_est_single, 0, 1)
        elif model == 'cauchy':
            # Copy Xi_est_single as it is
            Xi_est = Xi_est_single.copy()
        else:
            # Apply the logarithmic function to Xi_est_single
            Xi_est = uf.loga(Xi_est_single, 0, 1)

    # Randomly initialize the SEM parameters
    Theta = random_param(Y.shape[0], Y.shape[1], fonc=fonc)
    # Estimate parameters based on Xi_est and Y using the specified model and function
    Theta.est_param_from(Xi_est, Y, a_base, model=model, fonc=fonc)

    return Theta, Xi_est


def random_param(lx, ly, fonc="gau"):
    """
    Generate random parameters for the GMRF model.

    Parameters:
    -----------
    lx : int
        Number of rows in the grid.
    ly : int
        Number of columns in the grid.
    fonc : str, optional
        The function type. Default is "gau".

    Returns:
    --------
    param : p.Param_GMRF
        Randomly generated parameters for the GMRF model.
    """
    mx = -1 + 2 * np.random.random()
    sx = 0.5 + 10 * np.random.random()
    rx = 5 + np.random.random() * 10
    a = 0
    b = 1

    if fonc == "gau":
        return p.Param_GMRF((lx, ly), (mx, sx, rx, a, b))
    else:
        return p.Param_GMRF((lx, ly), (mx, sx, rx, a, b), gau=False)



def unified_SEM(Y, a_base, par_HMC, model, nb_iter_max=20, nb_iter_min=10, verbose=True, EM=True):
    """
    Perform Stochastic Expectation-Maximization (SEM) on the provided data Y using the specified model and HMC parameters.
    If EM=True, the EM version is used, otherwise the SEM version is used. 
    
    Parameters:
    -----------
    Y : array-like
        The input data.
    a_base : array-like
        The base matrix for Y.
    par_HMC : Param_HMC
        The HMC parameters.
    model : str
        The name of the model used. Possible values are 'logit', 'lognorm', 'single', 'cauchy', 'logit_exp', 'lognorm_exp', and 'single_exp'.
    nb_iter_max : int, optional
        The maximum number of iterations allowed. Default value is 20.
    nb_iter_min : int, optional
        The minimum number of iterations allowed. Default value is 10.
    verbose : bool, optional
        Whether to print progress messages or not. Default value is True.
    EM : bool, optional
        Whether to use the EM algorithm or not. Default value is True.
    
    Returns:
    --------
    Theta_SEM : Theta
        The final estimated theta.
    theta_series : array-like
        An array of estimated thetas for each iteration.
    lp : array-like
        An array of log-likelihood values for each iteration.
    not_converged : int
        The number of parameters that did not converge.
    X_est : array-like
        The estimated X.
    """
    
    start = time.time()
    
    # Determine the type of function used based on the model type
    if model == 'logit' or model == 'lognorm' or model == 'single' or model == 'cauchy':
        fonc = "gau"
    elif model == 'logit_exp' or model == 'lognorm_exp' or model == 'single_exp':
        fonc = 'exp'
    else:
        print("Wrong model ! (unified_SEM)")
    
    # Initialize SEM with an initial theta and X_est
    Theta, X_est = init_SEM(Y, a_base, model, fonc, par_HMC)
    
    N_param = Theta.toarray().size
    theta_series = np.zeros(shape=(nb_iter_max, N_param))
    theta_series[0, :] = Theta.toarray() 
    lp = []

    
    # Iterate through SEM until convergence or maximum iterations reached
    for r in range(1, nb_iter_max):
        if verbose and EM:
            print('\r  EM : iteration %.0f...' % r, end='')
        if verbose and not EM:
            print('\r  SEM : iteration %.0f...' % r, end='')
        
        # Sample along the posterior distribution
        if model == 'single' or model == 'single_exp':
            
            if EM:
                X_est = ua.direct_inference_single(a_base, Y, Theta)
            else:
                X_est = ua.sample_Xpost(a_base, Y, Theta)
            lp_locall = 0
        else:
            X_init_HMC = X_est.copy()
            X_seq, X_fin, lp_locall = uhmc.sample_from_nuts(Y, Theta, a_base, model, X_init_HMC, n_iter=par_HMC.n_total)
            
            if EM == 2:
                MH_all_X, X_seq, lp_all, burnin = uhmc.sample_HMC_par_autostop(Y, Theta, a_base, model, par_HMC, supervized=True, X_start=X_init_HMC, verbose_all=verbose)
                X_est = MH_all_X[0]
            else:
                if EM:
                    X_est = np.mean(X_seq, axis=0)
                else:
                    X_est = X_fin.copy()
            
            # Storage
            if r == 1:
                lp = lp_locall.copy()
            else:
                lp = np.append(lp, lp_locall)
        
        # Reestimate the parameters
        Theta.est_param_from(X_est, Y, a_base, model=model, fonc=fonc)
        
        # Convenience storage
        theta_series[r, :] = Theta.toarray()
        
        # Convergence analysis
        not_converged = 0
        for id_p in range(N_param):
            not_converged += evaluate_convergence(theta_series[:, id_p], r, lim=nb_iter_min, thr=0.05)
            if not_converged > 0:
                break
        
        if not_converged == 0:
            # End of SEM - empirical convergence
            break
    
    theta_series = theta_series[:r + 1, :]
    Theta_SEM = get_unified_theta(theta_series, r, model, Theta.lx, Theta.ly, fonc=fonc)
    
    end = time.time() - start
    if verbose:
        print("")
        print("  SEM stops at iteration %.0f - elapsed = %.2f seconds." % (r, end))
    
    return Theta_SEM, theta_series, lp, not_converged, X_est


def evaluate_convergence(param, i, lim=10, thr=1e-2):
    """
    Evaluate the convergence of a parameter sequence.
    
    Parameters:
    -----------
    param : array-like
        The parameter sequence.
    i : int
        The current iteration.
    lim : int, optional
        The number of previous iterations to consider. Default value is 10.
    thr : float, optional
        The threshold for convergence. Default value is 1e-2.
    
    Returns:
    --------
    out : bool
        True if the parameter sequence has converged, False otherwise.
    """
    if i > lim:
        mean_par = param[i - lim:i].mean()
        if np.abs(mean_par) != 0:
            ecart_relatif = np.abs(mean_par - param[i]) / np.abs(mean_par)
        else:
            ecart_relatif = np.abs(mean_par - param[i])
        out = ecart_relatif > thr
    else:
        out = True  # No convergence
    return out


def get_theta_from_array(arr, model, lx, ly, fonc):
    """
    Create a Theta object from an array of parameters based on the specified model and function type.
    
    Parameters:
    -----------
    arr : array-like
        Array of parameters.
    model : str
        The name of the model.
    lx : int
        The number of columns in the grid.
    ly : int
        The number of rows in the grid.
    fonc : str
        The type of function.
    
    Returns:
    --------
    Theta_SEM : Theta
        The Theta object.
    """
    if model == 'single' or model == 'single_exp' or model == 'logit' or model == 'logit_exp' or model == 'lognorm' or model == 'lognorm_exp' or model == 'cauchy':
        if fonc == 'gau':
            gau = True
        else:
            gau = False
        Theta_SEM = p.Param_GMRF((lx, ly), arr[:5], sig_noise=arr[5], gau=gau)
    elif model == 'pair':
        Theta_SEM = p.Param_couple((lx, ly), arr[:3], arr[3:6], arr[6])
    else:
        print("Bad model name (ged_unified theta)")
    return Theta_SEM
    
def get_unified_theta(theta_series, r, model, lx, ly, fonc='gau'):
    """
    Create a unified Theta object from a series of estimated thetas.
    
    Parameters:
    -----------
    theta_series : array-like
        The series of estimated thetas.
    r : int
        The number of iterations.
    model : str
        The name of the model.
    lx : int
        The number of columns in the grid.
    ly : int
        The number of rows in the grid.
    fonc : str, optional
        The type of function. Default value is 'gau'.
    
    Returns:
    --------
    Theta_SEM : Theta
        The unified Theta object.
    """
    theta_avg = get_avg_result_arr(theta_series, r)
    Theta_SEM = get_theta_from_array(theta_avg, model, lx, ly, fonc)
    return Theta_SEM

def get_avg_result_arr(param,i):
    return param[i+1-10:i+1,:].mean(axis=0)
