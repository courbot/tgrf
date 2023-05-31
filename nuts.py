"""
This package implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011).

Adfapted from: https://github.com/mfouesneau/NUTS

Content
-------

The package mainly contains:
  nuts6                     return samples using the NUTS
  test_nuts6                example usage of this package

and subroutines of nuts6:
  build_tree                the main recursion in NUTS
  find_reasonable_epsilon   Heuristic for choosing an initial value of epsilon
  leapfrog                  Perfom a leapfrog jump in the Hamiltonian space
  stop_criterion            Compute the stop condition in the main loop


A few words about NUTS
----------------------

Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a Markov chain Monte
Carlo (MCMC) algorithm that avoids the random walk behavior and sensitivity to
correlated parameters, biggest weakness of many MCMC methods. Instead, it takes
a series of steps informed by first-order gradient information.

This feature allows it to converge much more quickly to high-dimensional target
distributions compared to simpler methods such as Metropolis, Gibbs sampling
(and derivatives).

However, HMC's performance is highly sensitive to two user-specified
parameters: a step size, and a desired number of steps.  In particular, if the
number of steps is too small then the algorithm will just exhibit random walk
behavior, whereas if it is too large it will waste computations.

Hoffman & Gelman introduced NUTS or the No-U-Turn Sampler, an extension to HMC
that eliminates the need to set a number of steps.  NUTS uses a recursive
algorithm to find likely candidate points that automatically stops when it
starts to double back and retrace its steps.  Empirically, NUTS perform at
least as effciently as and sometimes more effciently than a well tuned standard
HMC method, without requiring user intervention or costly tuning runs.

Moreover, Hoffman & Gelman derived a method for adapting the step size
parameter on the fly based on primal-dual averaging.  NUTS can thus be used
with no hand-tuning at all.

In practice, the implementation still requires a number of steps, a burning
period and a stepsize. However, the stepsize will be optimized during the
burning period, and the final values of all the user-defined values will be
revised by the algorithm.

reference: arXiv:1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
Carlo", Matthew D. Hoffman & Andrew Gelman
"""
import numpy as np
from numpy import log, exp, sqrt

try:
    import tqdm
except ImportError:
    tqdm = None


import Gaussian_MRF as gmrf
import unified_functions as uf


def leapfrog(theta, r, grad, epsilon, f, Theta, model):
    """
    Perform a leapfrog jump in the Hamiltonian space.

    Parameters:
    -----------
    theta : ndarray[float, ndim=1]
        Initial parameter position.

    r : ndarray[float, ndim=1]
        Initial momentum.

    grad : float
        Initial gradient value.

    epsilon : float
        Step size.

    f : callable
        A function that returns the log probability and gradient evaluated at theta.
        logp, grad = f(theta)

    Theta : object
        Object representing additional parameters.

    model : object
        Object representing the model.

    Returns:
    --------
    thetaprime : ndarray[float, ndim=1]
        New parameter position.

    rprime : ndarray[float, ndim=1]
        New momentum.

    gradprime : float
        New gradient.

    logpprime : float
        New log probability.
    """

    # Make half step in momentum
    rprime = r + 0.5 * epsilon * grad

    # Make full step in parameter space
    thetaprime = theta + epsilon * rprime  # Perform a leapfrog step
    # thetaprime = uhmc.threshold_acceptable(theta + epsilon * rprime, Theta, model, eps=0.001) # Alternative step with threshold

    # Compute new gradient and log probability
    logpprime, gradprime = f(thetaprime)

    # Make another half step in momentum
    rprime = rprime + 0.5 * epsilon * gradprime

    return thetaprime, rprime, gradprime, logpprime


def find_reasonable_epsilon(theta0, grad0, logp0, f, Theta, model, precomputed):
    """
    Heuristic for choosing an initial value of epsilon.

    Parameters:
    -----------
    theta0 : ndarray[float, ndim=1]
        Initial parameter position.

    grad0 : ndarray[float, ndim=1]
        Initial gradient.

    logp0 : float
        Initial log probability.

    f : callable
        A function that returns the log probability and gradient evaluated at theta.
        logp, grad = f(theta)

    Theta : object
        Object representing additional parameters.

    model : object
        Object representing the model.

    precomputed : object
        Object containing precomputed values.

    Returns:
    --------
    epsilon : float
        Initial step size.
    """

    epsilon = 10.0
    r0 = gmrf.sample_X(Theta, model).flatten()

    # Determine the direction of epsilon by performing a leapfrog step.
    _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, epsilon, f, Theta, model)

    # Adjust epsilon to avoid infinite likelihood values or out-of-range parameter values.
    k = 1.0
    while np.isinf(logpprime) or np.isinf(gradprime).any():
        k *= 0.5
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon * k, f, Theta, model)

    epsilon = 0.5 * k * epsilon

    # Calculate the log acceptance probability.
    logacceptprob = logpprime - logp0 + uf.calc_prior(rprime.reshape(Theta.lx, Theta.ly), model, precomputed) - uf.calc_prior(r0.reshape(Theta.lx, Theta.ly), model, precomputed)
    a = 1.0 if logacceptprob > np.log(0.5) else -1.0

    # Adjust epsilon until the acceptance probability crosses 0.5.
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2.0 ** a)
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f, Theta, model)
        logacceptprob = logpprime - logp0

    return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0,Theta,model,precomputed):
    """The main recursion."""
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f,Theta,model)
#        joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
        joint = logpprime + uf.calc_prior(r.reshape(Theta.lx,Theta.ly),model,precomputed)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        sprime = int((logu - 1000.) < joint)
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        # Compute the acceptance probability.
        if joint - joint0 > 10 : alphaprime = 1
        elif joint-joint0 < -10: alphaprime = np.exp(-10)
        else: alphaprime = min(1., np.exp(joint - joint0))
        #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0,Theta,model,precomputed)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0,Theta,model,precomputed)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0,Theta,model,precomputed)
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime


def nuts6(f, M, Madapt, theta0, Theta,model, delta=0.6, progress=False):
    """
       Implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS paper (Hoffman & Gelman, 2011).
    
       Runs Madapt steps of burn-in, during which it adapts the step size parameter epsilon,
       then starts generating samples to return.
    
       Note: The initial step size is tricky and not exactly the one from the initial paper.
       In fact, the initial step size could be given by the user in order to avoid potential problems.
    
       Parameters:
       -----------
       f : callable
       A function that returns the log probability and gradient evaluated at theta.
       logp, grad = f(theta)
    
       M : int
       Number of samples to generate.
    
       Madapt : int
       The number of steps of burn-in/how long to run the dual averaging algorithm to fit the step size epsilon.
    
       theta0 : ndarray[float, ndim=1]
       Initial guess of the parameters.
    
       Theta : object
       Object representing additional parameters.
    
       model : object
       Object representing the model.
    
       delta : float, optional
       Targeted acceptance fraction (default is 0.6).
    
       progress : bool, optional
       Whether to show progress (default is False).
    
       Returns:
       --------
       samples : ndarray[float, ndim=2]
       M x D matrix of samples generated by NUTS. samples[0, :] = theta0.
    
       lnprob : ndarray[float, ndim=1]
       Log probabilities corresponding to each sample.
    
       epsilon : float
       Final step size used by the algorithm.
       """


    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    lnprob = np.empty(M + Madapt, dtype=float)

    logp, grad = f(theta0)
    samples[0, :] = theta0
    lnprob[0] = logp

    precomputed = uf.calc_precomputations(Theta,model)
    # Choose a reasonable first epsilon by a simple heuristic.
    epsilon = find_reasonable_epsilon(theta0, grad, logp, f,Theta,model,precomputed)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    Hbar = 0


    for m in progress_range(1, M + Madapt, progress=progress):

        r0 = gmrf.sample_X(Theta,model).flatten()

        #joint lnp of theta and momentum r
        joint = logp + uf.calc_prior(r0.reshape(Theta.lx,Theta.ly),model,precomputed)

        # Resample u ~ uniform([0, exp(joint)]).
        # Equivalent to (log(u) - joint) ~ exponential(1).
        logu = float(joint - np.random.exponential(1, size=1))

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = r0[:]
        rplus = r0[:]
        gradminus = grad[:]
        gradplus = grad[:]

        j = 0  # initial heigth j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.

        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint,Theta,model,precomputed)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint,Theta,model,precomputed)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                samples[m, :] = thetaprime[:]
                lnprob[m] = logpprime
                logp = logpprime
                grad = gradprime[:]
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Increment depth.
            j += 1
            if j >=7: # maximum three depth
                s = 0 # stop here
        
        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar
    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    return samples, lnprob, epsilon

def progress_range(minimum, maximum, progress=True):
    """
    A range-like function that displays progress information.

    Parameters:
    -----------
    minimum : int
        Lower bound of the range.
    maximum : int
        Upper bound of the range.

    Keyword Arguments:
    ------------------
    progress : bool, optional
        If True, show progress information. If False, display nothing.

    Yields:
    -------
    int
        The current iteration value within the range.

    Example:
    --------
    for i in progress_range(0, 100, progress=True):
        # Perform iterative tasks
        pass
    """
    if not progress:
        for i in range(minimum, maximum):
            yield i
    elif tqdm is not None:
        for i in tqdm.trange(minimum, maximum):
            yield i
    else:
        for i in range(minimum, maximum):
            if i % 100 == 0:
                print('iteration %i/%i' % (i, maximum - minimum))
            yield i