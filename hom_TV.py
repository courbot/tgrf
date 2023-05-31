#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on may, 2023.

@author: Jean-Baptiste Courbot - jean-baptiste.courbot@uha.fr
"""
import numpy as np
import time
from pyfftw.interfaces.numpy_fft import fft2 as pfft2
import prox_tv 
import numpy.fft as nf


def run_tv(Y, a_base, sig_noise, solver='TV1', verbose=False):
    """
    Run the total variation (TV) solver for compressive sensing.

    Arguments:
    Y -- Input image.
    a_base -- Base image.
    sig_noise -- Noise standard deviation.
    solver -- Solver type ('TV1' or 'TV2').
    verbose -- Flag indicating whether to print verbose output.

    Returns:
    X_tv -- Estimated image.
    """
    # Compute the Optical Transfer Function (OTF)
    OTF0 = pfft2(a_base)
    OTF = np.zeros(shape=(OTF0.shape[0] + 1, OTF0.shape[1] + 1, 1))
    OTF[1:, 1:, :] = OTF0.reshape(100, 100, 1).real

    # Prepare the input image
    YY = np.zeros(shape=(100, 100, 1))
    YY[:, :, 0] = Y

    # Set parameters
    lambda_start = 1
    target_std = sig_noise * 1.05
    nb_step = 20

    # Perform the simple homotopy algorithm
    X_tv = simple_homotopy(solver, YY, OTF, nb_step, lambda_start, target_std,
                           verbose=verbose, use_lambda_start=False)

    return X_tv[:, :, 0]


def simple_homotopy(solver, Y, OTF, nb_step, lambda_start, target_std,
                    verbose=False, use_lambda_start=False):
    """
    Free adaptation of the Simple Homotopy Algorithm for Compressive Sensing:
        https://pdfs.semanticscholar.org/c1c3/c78f3ef85618b9aa34569b661ecc23dcf66e.pdf

    Arguments:
    solver -- Solver type ('TV1' or 'TV2').
    Y -- Input image.
    OTF -- Optical Transfer Function.
    nb_step -- Number of steps for the homotopy algorithm.
    lambda_start -- Initial lambda value.
    target_std -- Target standard deviation.
    verbose -- Flag indicating whether to print verbose output.
    use_lambda_start -- Flag indicating whether to use lambda_start.

    Returns:
    lam_used -- Array of lambda values used.
    X_tv -- Estimated image.
    liste_l1 -- List of L1 values.
    liste_l2 -- List of L2 values.
    diracs_out -- Output diracs.
    """
    P, Q, R = Y.shape

    print("###### Entering the homotopy algorithm with target std=%.6f" % target_std)
    start = time.time()
    lam_used = []
    lam_used.append(lambda_start)
    previous_x = None

    for i in range(nb_step):
        nb_iter_gp_max = 50
        gamma_start = 5
        beta = 0.95
        with_absorption = False
        stop_rate = 1e-3

        if solver == 'TV1':
            p = 1
        elif solver == 'TV2':
            p = 2

        if verbose:
            print("I run the TV solver with lam = %.4f" % lam_used[i])

        X_tv, x_sauv, _, _ = run_acc_prox_gradient(Y, nb_iter_gp_max, lam_used[i], OTF, gamma_start, beta,
                                                   stop_rate, p, with_absorption, x_init=previous_x)

        previous_x = np.copy(X_tv)
        X_est_conv = our_convolution(X_tv, OTF)

        current_std = (Y - X_est_conv).std()

        if current_std <= target_std:
            break
        else:
            if verbose:
                print("###### Target STD not attained for lam = %.4f" % lam_used[i])
                print("###### We have std_current = %.4f > std_target = %.4f" % (current_std, target_std))

            new_lam = 0.5 * lam_used[i]
            lam_used.append(new_lam)


    print("######")
    print("Simple homotopy search found lamda = %.4f and the corresponding diracs in %.0f steps" % (lam_used[i], i + 1))
    print("Elapsed time: %.2f" % (time.time() - start))

    return X_tv



def run_acc_prox_gradient(y, nb_iter_gp, lam, OTF, gamma_start, beta, stop_rate, p, with_absorption=False, x_init=None):
    """
    Run accelerated proximal gradient algorithm.

    Arguments:
    y -- Input image.
    nb_iter_gp -- Number of iterations for gradient proximal.
    lam -- Regularization parameter.
    OTF -- Optical Transfer Function.
    gamma_start -- Initial gamma value for line search.
    beta -- Decay factor for gamma.
    stop_rate -- Threshold for stopping based on dual gap rate.
    p -- Parameter for TV (Total Variation) calculation.
    with_absorption -- Flag indicating whether to consider absorption.
    x_init -- Initial image estimate.

    Returns:
    x_gp -- Final image estimate.
    x_sauv -- Array of saved image estimates.
    valeur_gamma -- Array of gamma values used.
    dual_gap -- Array of dual gap values.
    """
    verbose = False
    valeur_gamma = np.zeros(shape=nb_iter_gp)
    dual_gap = np.zeros(shape=nb_iter_gp)
    rate_dg = np.zeros(shape=nb_iter_gp)
    x_sauv = np.zeros(shape=(y.shape[0], y.shape[1], y.shape[2], nb_iter_gp))

    if verbose:
        print("  Gradient proximal.")

    x_courant = np.copy(y)
    x_precedent = np.zeros_like(x_courant)

    start = time.time()

    for iter_gp in range(nb_iter_gp):
        omega_k = iter_gp / (iter_gp + 3)
        xbar_courant = x_courant + omega_k * (x_courant - x_precedent)

        gamma, prox_th = line_search_gamma(y.real, xbar_courant.real, OTF, gamma_start, beta, lam, p, verbose=False,
                                           real=True)

        gamma_start = gamma
        x_precedent = np.copy(x_courant)
        x_courant = np.copy(prox_th)

        dual_gap[iter_gp] = np.abs(calc_dual_gap(x_courant, y, lam, OTF, p))
        x_sauv[:, :, :, iter_gp] = x_courant
        valeur_gamma[iter_gp] = gamma

        if iter_gp > 10:
            G_prec = dual_gap[iter_gp - 11:iter_gp - 1].mean()
            rate_dg[iter_gp] = np.abs(dual_gap[iter_gp] - G_prec) / G_prec

        if rate_dg[iter_gp] < stop_rate and iter_gp > 20:
            if verbose:
                print("  Gradient proximal stopped at iteration %.0f due to small variation in dual gap." % iter_gp)
            break

    x_gp = np.copy(x_courant)

    if verbose:
        print("  End of gradient proximal.")
        end = time.time() - start
        print("  Computation time = %.2f s" % end)

    return x_gp, x_sauv[:, :, :, :iter_gp + 1], valeur_gamma[:iter_gp + 1], dual_gap[:iter_gp + 1]



def line_search_gamma(y, x_courant, OTF, gamma_0, beta, lam, p, real=True, verbose=False):
    """
    Perform line search to find the optimal value of gamma.
    
    Documentation : https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

    Arguments:
    y -- Input image.
    x_courant -- Current image estimate.
    OTF -- Optical Transfer Function.
    gamma_0 -- Initial gamma value.
    beta -- Decay factor for gamma.
    lam -- Regularization parameter.
    p -- Parameter for TV (Total Variation) calculation.
    real -- Flag indicating whether the image is purely real.
    verbose -- Flag indicating whether to print verbose output.

    Returns:
    gamma -- Optimal value of gamma.
    prox_th -- Result of the proximal operation.
    """
    gamma = np.copy(gamma_0)
    gradient_x = calc_gradient2(x_courant, y, OTF)  # gradient = H^T * (Hx-y)
    prox_th = np.zeros_like(x_courant, dtype=complex)
    stop = False
    nb_iter = 0

    # Right-hand side of the line search inequality
    ls_elt_droite_1 = np.linalg.norm(our_convolution(x_courant, OTF)) ** 2  # Real

    while not stop:
        # Apply proximal operator separately for real and complex components
        if real:
            prox_th = prox_tv.tvgen(x_courant.real - gamma * gradient_x.real, [lam, lam], [1, 2], [p, p], n_threads=10)
        else:
            prox_th.imag = prox_tv.tvgen(x_courant.imag - gamma * gradient_x.imag, [lam, lam], [1, 2], [p, p], n_threads=10)
            prox_th.real = prox_tv.tvgen(x_courant.real - gamma * gradient_x.real, [lam, lam], [1, 2], [p, p], n_threads=10)

        Gt = 1 / gamma * (x_courant - prox_th)

        # Left-hand side of the line search inequality
        ls_elt_gauche = 0.5 * np.linalg.norm(our_convolution(x_courant - gamma * Gt, OTF)) ** 2  # Compatible with complex

        ls_elt_droite_2 = -gamma * (np.conj(gradient_x) * Gt).sum()  # Compatible with complex

        ls_elt_droite_3 = gamma / 2 * np.linalg.norm(Gt) ** 2  # Real

        stop = (ls_elt_gauche < (ls_elt_droite_1 + ls_elt_droite_2 + ls_elt_droite_3))

        if not stop:
            gamma = beta * gamma

        nb_iter += 1
        if nb_iter > 20:
            break

    return gamma, prox_th


def calc_dual_gap(x_current, y, lam, OTF, p):
    """
    Calculate the dual gap, which is the difference between the solution of the primal problem and the dual problem.
    
    Sources : 
    http://www.lix.polytechnique.fr/bigdata/mathbigdata/wp-content/uploads/2014/11/2014_11_3.pdf
    http://proceedings.mlr.press/v48/dunner16-supp.pdf

    Arguments:
    x_current -- Current solution of the primal problem.
    y -- Input image.
    lam -- Regularization parameter.
    OTF -- Optical Transfer Function.
    p -- Parameter for TV (Total Variation) calculation.

    Returns:
    dg -- Dual gap value.
    """
    Hx = our_convolution(x_current, OTF)
    r = Hx - y

    dx, dy = np.gradient(-Hx[:, :, 0])
    div = dx + dy

    lam_tv = calc_tv(x_current, p) * lam

    dg = np.linalg.norm(r) ** 2 + lam_tv + (r * y).sum() + lam * (x_current * div).max()

    return dg

def calc_tv(vol, p=1):
    """
    Calculate the TV (Total Variation) of a volume.

    Arguments:
    vol -- Input volume.
    p -- Parameter for TV calculation.

    Returns:
    TV -- TV value.
    """
    p = p * 1.0
    gx, gy = np.gradient(vol[:, :, 0])
    TV = np.sum(np.abs(gx) ** p + np.abs(gy) ** p) ** (1 / p)

    return TV

def calc_gradient2(x, y, OTF):
    """
    Calculate the gradient using an alternative method.

    Arguments:
    x -- Input image.
    y -- Output image.
    OTF -- Optical Transfer Function.

    Returns:
    grad -- Gradient of the image.
    """
    Hx = our_convolution(x, OTF)
    grad = our_convolution(Hx - y, OTF)

    return grad

def our_convolution(G, OTF):
    """
    Perform convolution between input image G and Optical Transfer Function (OTF).

    Arguments:
    G -- Input image of shape (M, N, 1), where M and N are the dimensions of the image.
    OTF -- Optical Transfer Function of shape (M, N, 1), where M and N are the dimensions of the OTF.

    Returns:
    out -- Convolved image of shape (M, N, 1).
    """
    
    # Fourier transform of the input image G
    G_fourier = nf.fft2(nf.ifftshift(G[:, :, 0]).real)

    # Convolve the Fourier transformed image with the OTF
    conv_fourier = G_fourier * OTF[1:, 1:, 0]
    
    # Inverse Fourier transform to obtain the convolved image
    out = np.zeros(shape=(100, 100, 1))
    out[:, :, 0] = nf.fftshift(nf.ifft2(conv_fourier)).real

    return out



