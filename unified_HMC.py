#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 14:36:44 2022

@author: courbot
"""

# external
import numpy as np
from scipy.fft import rfft2 as fft2
from scipy.fft import irfft2 as ifft2
import time
import multiprocessing as mp
import gc

# homemade
import Gaussian_MRF as gmrf
import unified_functions as uf
import nuts as nuts


def sample_from_nuts(Y, Theta, a_base, model, X_start, n_iter, progress=False):
    """
    Perform sampling from the No-U-Turn Sampler (NUTS) algorithm.

    Parameters:
    -----------
    Y : array-like
        Input data.
    Theta : object
        Object containing parameters for the model.
    a_base : array-like
        Array of baseline values.
    model : str
        Model type.
    X_start : array-like
        Initial X values.
    n_iter : int
        Number of iterations.
    progress : bool, optional
        Whether to display progress, by default False.

    Returns:
    --------
    chain : list
        List of sampled chains.
    final_sample : array-like
        Final sampled X values.
    lnprob : float
        Log probability.
    """
    def f_grad_for_nuts(X_vec):
        """
        Calculate the posterior and gradient for the No-U-Turn Sampler (NUTS) algorithm.
        Defined here to ease variable sharing.
        Parameters:
        -----------
        X_vec : array-like
            Vectorized X values.
    
        Returns:
        --------
        post_out : array-like
            Posterior values.
        grad_out : array-like
            Gradient values.
        """
    
        N = 1 # X_vec.size (Commented out as it's not being used)
        sig_noise = Theta.sig_noise
    
        X = X_vec.reshape(lx, ly)
        Xi = uf.form_Xi(X, model, Theta)
    
        # Future : deserves a separate function per model
        if model == 'cauchy':
            U = X_vec.reshape(lx, ly)
            lam = 0.03
            # gam = 1
    
            u_i_j = U.copy()
            u_ip_j = np.roll(u_i_j, 1, axis=0)
            u_i_jp = np.roll(u_i_j, 1, axis=1)
            u_im_j = np.roll(u_i_j, -1, axis=0)
            u_im_jp = np.roll(u_i_j, (-1, +1), axis=(0, 1))
            u_ip_jm = np.roll(u_i_j, (+1, -1), axis=(0, 1))
            u_im_jm = np.roll(u_i_j, (-1, -1), axis=(0, 1))
            u_i_jm = np.roll(u_i_j, -1, axis=1)
    
            C1 = lam ** 2 + (u_ip_j - u_i_j) ** 2 + (u_i_jp - u_i_j) ** 2
            C2 = lam ** 2 + (u_i_j - u_im_j) ** 2 + (u_im_jp - u_im_j) ** 2
            C3 = lam ** 2 + (u_ip_jm - u_im_jm) ** 2 + (u_i_j - u_i_jm) ** 2
    
            pointwise = C1
            log_pointwise = -3 / 2 * np.log(pointwise)
            log_central = 0
            log_prior = (log_pointwise + log_central).sum()
    
            grad_prior = -3 / 2 * 2 * ((u_i_j - u_ip_j) + (u_i_j - u_i_jp)) / C1
            grad_prior += -3 / 2 * 2 * (u_i_j - u_im_j) / C2
            grad_prior += -3 / 2 * 2 * (u_i_j - u_i_jm) / C3
    
            Residual_ft = (Y_ft - (fft2(U) * a_base_ft))
            HR = ifft2(a_base_ft * Residual_ft).real
            lp_grad = N / sig_noise ** 2 * HR
    
            norm_residual = np.linalg.norm(ifft2(Residual_ft))
            log_likelihood = -N * (0.5 / (sig_noise ** 2)) * norm_residual ** 2 - 0.5 * N * np.log(2 * np.pi) - N * np.log(
                sig_noise)
    
            post_out = (log_prior + log_likelihood).flatten() / N
            grad_out = (grad_prior + lp_grad).flatten() / N
    
        else:
            qx_ft, dQx = precomputed[0], precomputed[1]
            QX = ifft2(fft2(X) * qx_ft)
    
            X_for_exp = np.clip(X, -20, 20)
    
            if model == 'logit' or model == 'logit_exp':
                sigmo = 1 / (1 + np.exp(-X_for_exp))
                phi_deriv = Theta.b * sigmo * (1 - sigmo)
            elif model == "lognorm" or model == 'lognorm_exp':
                phi_deriv = Theta.b * np.exp(X_for_exp)
            else:
                print("error on the model !! (sample from nuts)")
    
            Residual_ft = (Y_ft - (fft2(Xi) * a_base_ft))
            HR = ifft2(a_base_ft * Residual_ft)
            lp_grad = N / sig_noise ** 2 * HR * phi_deriv
    
            grad_out = -(-QX + lp_grad).flatten() / N
    
            norm_residual = np.linalg.norm(ifft2(Residual_ft))
            log_likelihood = -N * (0.5 / (sig_noise ** 2)) * norm_residual ** 2 - 0.5 * N * np.log(2 * np.pi) - N * np.log(
                sig_noise)
    
            log_prior = -0.5 * (QX * X).sum() - 0.5 * np.log(dQx)
    
            post_out = (log_likelihood + log_prior).flatten() / N
    
        return post_out, grad_out

    
    M = n_iter + 1
    if model == 'cauchy':
        Madapt = 5
    else:
        Madapt = 5
    delta = 0.3
    Y_ft = fft2(Y)
    a_base_ft = fft2(a_base)
    lx, ly = Y.shape

    if model != 'cauchy':
        precomputed = uf.calc_precomputations(Theta, model)
    X0 = X_start.flatten()

    # Perform NUTS sampling
    samples, lnprob, epsilon = nuts.nuts6(f_grad_for_nuts, M, Madapt, X0, Theta, model, delta, progress)
    chain = [samples[i, :].reshape(lx, ly) for i in range(M)]

    return chain, samples[-1, :].reshape(lx, ly), lnprob

def sample_Xi(Theta,model):
    return uf.form_Xi(gmrf.sample_X(Theta,model),model,Theta)



# =============================================================================
# Parralel HMC
# =============================================================================
def single_for_par(Theta, a_base, Y, verbose, par_HMC, model, Xi_init=None, supervized=True):
    """
    Runs a single-unit Hamiltonian Monte Carlo (HMC) to be used in a parallel setting.
    This function is model-independent.

    Parameters:
    -----------
    Theta : object
        Object containing parameters for the model.
    a_base : array-like
        Array of baseline values.
    Y : array-like
        Input data.
    verbose : bool
        Whether to display detailed information.
    par_HMC : object
        Object containing parameters for the HMC algorithm.
    model : str
        Model type.
    Xi_init : array-like, optional
        Initial Xi values, by default None.
    supervized : bool, optional
        Whether to use supervized learning, by default True.

    Returns:
    --------
    chainX : array-like
        Chain samples of X.
    acceptance_rate : float
        Acceptance rate.
    lp : array-like
        Log probabilities of the posterior.
    MH_fin_X : array-like
        Final X estimates.
    """

    np.random.seed()

    if Xi_init is None:
        Xi_init = sample_Xi(Theta, model)  # Sample initial Xi values
        
    chainX, X_est, lp = sample_from_nuts(Y, Theta, a_base, model, Xi_init, n_iter=par_HMC.n_total)
    
    MH_fin_X = X_est.reshape(par_HMC.nb_field, Theta.lx, Theta.ly)
    acceptance_rate = 0

    return chainX, acceptance_rate, lp, MH_fin_X

  
def sample_HMC_par(Y, Theta, a_base, model, par_HMC, supervized=True, X_start=None, verbose_all=True):
    """
    Produces several independent Hamiltonian Monte Carlo (HMC) chains.
    Stops when the average posterior is stable and checks posterior at the end of each batch.
    Returns the average end of chains and other results.

    Parameters:
    -----------
    Y : array-like
        Input data.
    Theta : object
        Object containing parameters for the model.
    a_base : array-like
        Array of baseline values.
    model : str
        Model type.
    par_HMC : object
        Object containing parameters for the HMC algorithm.
    supervized : bool, optional
        Whether to use supervized learning, by default True.
    X_start : array-like, optional
        Initial X values, by default None.
    verbose_all : bool, optional
        Whether to display detailed information, by default True.

    Returns:
    --------
    MH_all_X : array-like
        Average end of chains.
    X_seq : list
        List of chain samples.
    lp_all : array-like
        Log probabilities of the posterior.
    burnin : bool
        Whether burn-in is finished.
    """

    lx, ly = Theta.lx, Theta.ly
    start = time.time()
    verbose_chain = False  # To monitor individual chain behavior

    nb_proc = np.minimum(mp.cpu_count(), 31)
    pool = mp.Pool(processes=nb_proc - 1, maxtasksperchild=1)

    burnin = True

    lp_all = np.zeros(shape=(par_HMC.nb_rea, par_HMC.n_total * par_HMC.nb_batch))
    current_X = np.zeros(shape=(par_HMC.nb_rea, lx, ly))
    for batch in range(par_HMC.nb_batch):
        if verbose_all:
            print('\r     ParHMC : running batch no. %.0f...' % batch, end='')

        # Starting point
        if batch == 0:
            X_init = [X_start for r in range(par_HMC.nb_rea)]
        else:
            X_init = current_X

        # Compute in parallel
        out = pool.starmap(single_for_par,
                           [(Theta, a_base, Y, verbose_chain, par_HMC, model, X_init[r], supervized) for r in range(par_HMC.nb_rea)])

        # Retrieve
        for r in range(par_HMC.nb_rea):
            chainX, _, lp_loc, _ = out[r]
            current_X[r] = chainX[-1]
            lp_all[r, batch * par_HMC.n_total: (batch + 1) * par_HMC.n_total] = np.array(lp_loc)[1:]

        # Check stability of the posterior - Burnin and convergence evaluation
        if not burnin:
            break

        burnin = check_burnin(lp_all, batch, par_HMC)

    pool.close()
    pool.join()
    gc.collect()

    if verbose_all:
        print('')

    lp_all = lp_all[:, :(batch + 1) * par_HMC.n_total]  # Truncate the posterior

    MH_all_X = np.zeros(shape=(par_HMC.nb_field, lx, ly))
    for r in range(par_HMC.nb_rea):
        MH_all_X += out[r][3].reshape(par_HMC.nb_field, lx, ly) / (1.0 * par_HMC.nb_rea)
    X_seq = [out[r][3] for r in range(par_HMC.nb_rea)]

    end = time.time() - start

    if verbose_all:
        if burnin:
            print("     ParHMC : burnin NOT before max. batch %.0f" % batch)
        else:
            print("     ParHMC : burnin finished before Batch %.0f : time to break" % batch)
        print("               - computation time: %.4f" % end)

    return MH_all_X, X_seq, lp_all, burnin

def sample_HMC_par_autostop(Y, Theta, a_base, model, par_HMC, supervized=True, X_start=None, verbose_all=True):
    """
    Produces several independent HMC chains and stops when the average posterior is stable.
    Checks the posterior at the end of each batch and returns the average end of chains.
    Runs one more batch when burn-in is attained.

    Parameters:
    -----------
    Y : array-like
        Input data.
    Theta : object
        Object containing parameters for the model.
    a_base : array-like
        Array of baseline values.
    model : str
        Model type.
    par_HMC : object
        Object containing parameters for the HMC algorithm.
    supervized : bool, optional
        Whether to use supervised learning, by default True.
    X_start : array-like, optional
        Initial X values, by default None.
    verbose_all : bool, optional
        Whether to display detailed information, by default True.

    Returns:
    --------
    MH_all_X : array-like
        Final X estimates.
    X_seq : list
        List of X sequences for each chain.
    lp_all : array-like
        Log probabilities of the posterior.
    burnin : bool
        Indicates whether the burn-in phase was completed.
    """

    lx, ly = Theta.lx, Theta.ly
    start = time.time()
    verbose_chain = False  # To monitor individual chain behavior

    nb_proc = np.minimum(mp.cpu_count(), 31)
    pool = mp.Pool(processes=nb_proc - 1, maxtasksperchild=1)

    burnin = True
    autostop = False

    lp_all = np.zeros(shape=(par_HMC.nb_rea, par_HMC.n_total * par_HMC.nb_batch))
    current_X = np.zeros(shape=(par_HMC.nb_rea, lx, ly))
    
    for batch in range(par_HMC.nb_batch):
        if verbose_all:
            print('\r     ParHMC: running batch no. %.0f...' % batch, end='')

        # Starting point
        if batch == 0:
            if X_start is not None:
                X_init = [X_start for _ in range(par_HMC.nb_rea)]  # Either None or given
            else:
                X_init = [X_start for _ in range(par_HMC.nb_rea)]
        else:
            X_init = current_X

        # Compute in parallel
        out = pool.starmap(single_for_par, [(Theta, a_base, Y, verbose_chain, par_HMC, model, X_init[r], supervized) for r in range(par_HMC.nb_rea)])

        # Retrieve results
        for r in range(par_HMC.nb_rea):
            chainX, _, lp_loc, _ = out[r]
            current_X[r] = chainX[-1]
            lp_all[r, batch * par_HMC.n_total: (batch + 1) * par_HMC.n_total] = np.array(lp_loc)[1:]

        # Check stability of the posterior - Burn-in and convergence evaluation
        if burnin:
            burnin = check_burnin(lp_all, batch, par_HMC)
            if burnin == False:
                end_burnin = (batch + 1) * par_HMC.n_total
        else:  # Burn-in finished
            autostop = stop_potential_scale_reduction(lp_all[:, :(batch + 1) * par_HMC.n_total], end_burnin, threshold=2)

        if autostop == True:
            break;

    pool.close()
    pool.join()
    gc.collect()
    if verbose_all:
        print('')

    # Time to collect
    lp_all = lp_all[:, :(batch + 1) * par_HMC.n_total]  # Truncate the posterior

    MH_all_X = np.zeros(shape=(par_HMC.nb_field, lx, ly))
    for r in range(par_HMC.nb_rea):
        MH_all_X += out[r][3].reshape(par_HMC.nb_field, lx, ly) / (1.0 * par_HMC.nb_rea)
    X_seq = [out[r][3] for r in range(par_HMC.nb_rea)]

    end = time.time() - start
    if verbose_all:
        # Need to adapt these messages to autoscale
        if burnin:
            print("     ParHMC: burn-in NOT before max. batch %.0f" % batch)
        else:
            print("     ParHMC: burn-in finished before Batch %.0f: time to break" % batch)
        print("               - Computation time: %.4f" % (end))

    return MH_all_X, X_seq, lp_all, burnin

def check_burnin(lp_all, batch, par_HMC):
    """
    Checks the stability of the posterior at the end of the current batch
    with respect to the end of the previous batch.
    If the variation is lower than 5%, it indicates that the burn-in phase is finished.

    Parameters:
    -----------
    lp_all : array-like
        Log probabilities of the posterior.
    batch : int
        Current batch number.
    par_HMC : object
        Object containing parameters for the HMC algorithm.

    Returns:
    --------
    burnin : bool
        Indicates whether the burn-in phase is finished.
    """

    burnin = True

    if batch >= 5:
        lp_mean = lp_all[:, (batch + 1) * par_HMC.n_total - 1].mean()  # Average of values of the posterior along batches
        lp_old = lp_all[:, (batch) * par_HMC.n_total - 1].mean()

        if np.abs(lp_mean - lp_old) / np.abs(lp_mean) < 0.05:  # On average, less than 5% variation between current and old posterior
            burnin = False

    return burnin


def stop_potential_scale_reduction(lp_all, end_burnin, threshold):
    """
    Implements the potential scale reduction criterion to determine convergence.

    Parameters:
    -----------
    lp_all : array-like
        Log probabilities of the posterior.
    end_burnin : int
        Index indicating the end of the burn-in phase.
    threshold : float
        Threshold value for the potential scale reduction.

    Returns:
    --------
    convergence : bool
        Indicates whether the convergence criterion is met.
    """

    seq = lp_all[:, end_burnin:]
    m = seq.shape[0]  # Number of chains (rea) with j
    n = seq.shape[1]  # Length of chains (rea) with i

    # Calculate psi_p_j and psi_pp
    psi_p_j = seq.mean(axis=1)
    psi_pp = psi_p_j.mean(axis=0)

    # Calculate between-chain variance (B) and within-chain variance (W)
    B = n / (m - 1) * ((psi_p_j - psi_pp) ** 2).sum(axis=0)
    sj2 = 1 / (n - 1) * ((seq - psi_p_j[:, np.newaxis]) ** 2).sum(axis=1)
    W = 1 / m * sj2.sum(axis=0)

    # Calculate marginal posterior and potential scale reduction
    marginal_post = (n - 1) / n * W + B / n
    if W == 0:
        potential_red = 1000
    else:
        potential_red = np.sqrt(marginal_post / W)

    return potential_red < threshold

