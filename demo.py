"""
This demo can be run from the command line.  Change the demo_simple etc...
parameters below and then, from the command line:
    $ python demo.py
"""
import numpy as np
import scipy as sp
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt

import homework_03.src.utils as utils
import homework_03.src.cross_validator as cross_validator
import homework_03.src.simulator as simulator
import homework_03.src.linear_reg as linear_reg

################################################################################
## Change these to True/False to see different tests run
################################################################################


demo_simple = False
demo_pandas = True
demo_L2_regularization = False
demo_pinv_regularization = False
demo_cross_val = False
demo_large_mx = False


################################################################################
## Simple least squares regression with perfect data
## Show that the fit gets better as N increases
################################################################################
if demo_simple:
    print "\n-------- Simple least squares with perfect data -----------"
    K = 20
    eps = 0.1
    w = sp.randn(K, 1)

    for N in range(K, 100, 20):
        X = sp.randn(N, K)
        E = eps * sp.randn(N, 1)
        Y = simulator.fwd_model(X, w, E=E)

        w_hat = linear_reg.fit(X, Y)
        error = utils.get_relative_error(w_hat, w)
        print "N = %d, error = %.2e" % (N, error)

################################################################################
## Demonstrate handling of Pandas DataFrame/Series input/output
################################################################################
if demo_pandas:
    print "\n-------- Simple least squares with pandas objects -----------"
    N = 6
    K = 3
    eps = 0.1

    var_names = ['age', 'height', 'coolness']
    people_names = ['ian', 'daniel', 'chang', 'cathy', 'rachel', 'david']
    w = pd.Series(sp.randn(K), index=var_names)
    X = pd.DataFrame(sp.randn(N, K), index=people_names, columns=var_names)

    E = eps * sp.randn(N, 1)
    Y = simulator.fwd_model(X, w, E=E)

    w_hat = linear_reg.fit(X, Y)
    error = utils.get_relative_error(w_hat, w)

    print "N = %d, error = %.2e\nw_hat = \n%s" % (N, error, w_hat)

################################################################################
## L2 regularized least squares regression with correlated X
################################################################################
if demo_L2_regularization:
    print "\n-------- L2 regularized least squares with correlated X ----"
    N = 20
    K = 10
    sigma_eps = 0.3
    sigma_w = 1
    corr_len_X = 50.  # Correlation length
    delta = 1 * (sigma_eps / float(sigma_w))**2  # This is the "ideal" value

    error_list = []
    SNR_list = []
    for runnum in range(5000):
        # Get w, X, E, and then Y = Xw + E
        w = sigma_w * sp.randn(K, 1)
        Sigma = simulator.get_corr_matrix(K, corr_len_X)
        X = simulator.gaussian_samples(N, K, Sigma)
        E = sigma_eps * sp.randn(N, 1)
        Y = simulator.fwd_model(X, w, E)

        # Recover w
        w_hat = linear_reg.fit(X, Y, delta=delta)
        error = utils.get_relative_error(w_hat, w)
        error_list.append(error)
        SNR = linalg.norm(Y - E) / linalg.norm(E)  # Signal to Noise Ratio
        SNR_list.append(SNR)

    print "SNR = %.1f, delta = %.2e, error = %.3f" % (
        np.mean(SNR_list), delta, np.mean(error_list))

################################################################################
## Cross validation with correlated X
################################################################################
if demo_cross_val:
    print "\n-------- Cross validation with correlated X -----------"
    N = 200
    K = 10
    sigma_eps = 0.5
    sigma_w = 1
    corr_len_X = 50.  # Correlation length
    delta_0 = 1 * (sigma_eps / float(sigma_w))**2
    delta_list = delta_0 * np.logspace(-2, 1.0, 10)

    w = sigma_w * sp.randn(K, 1)
    Sigma = simulator.get_corr_matrix(K, corr_len_X)
    X = simulator.gaussian_samples(N, K, Sigma)

    # get w, x, e, and then y = xw + e
    E = sigma_eps * sp.randn(N, 1)
    Y = simulator.fwd_model(X, w, E)
    SNR = linalg.norm(Y - E) / linalg.norm(E)  # Signal to Noise Ratio
    print "SNR = %.1f" % SNR

    gof = cross_validator.cross_val(X, Y, delta_list)
    gof.plot(lw=5, style=['-', '--'])
    plt.show()

################################################################################
## pinv regularization
## Set cutoff slightly less than 1 / condition-number to cut off just the last
## Few singular directions
## Use linalg.svd and linear_reg._get_pinv for advanced exploration
################################################################################
if demo_pinv_regularization:
    print "\n-------- pinv regularization demo -----------"
    N = 2000
    K = 1000
    sigma_eps = 5  # Look at SNR to see the "true" noise level
    sigma_w = 1
    corr_len_X = 500.  # Correlation length
    cutoff = 1 / 200.  # Adjust this to regularize

    error_list = []
    SNR_list = []
    for runnum in range(5):
        # Get w, X, E, and then Y = Xw + E
        w = sigma_w * sp.randn(K, 1)
        Sigma = simulator.get_corr_matrix(K, corr_len_X)
        X = simulator.gaussian_samples(N, K, Sigma)
        E = sigma_eps * sp.randn(N, 1)
        Y = simulator.fwd_model(X, w, E)
        SNR = linalg.norm(Y - E) / linalg.norm(E)  # Signal to Noise Ratio
        SNR_list.append(SNR)

        # Recover w
        w_hat = linear_reg.fit(X, Y, method='pinv', cutoff=cutoff)
        error = utils.get_relative_error(w_hat, w)
        error_list.append(error)

    print "SNR = %.1f, cutoff = %.2e, cond = %.1e, error = %.3f" % (
        np.mean(SNR_list), cutoff, linalg.cond(X), np.mean(error_list))


################################################################################
## Large matrix
## Large enough correlated matrix causes error.  Check the condtion number.
################################################################################
if demo_large_mx:
    print "\n-------- Large matrix demo -----------"
    N = 2000 
    K = 1000
    sigma_eps = 0.  # Even with no additive noise the solution breaks down
    sigma_w = 1
    corr_len_X = 50000.  # Correlation length
    delta = 0 * (sigma_eps / float(sigma_w))**2

    error_list = []
    for runnum in range(2):
        # Get w, X, E, and then Y = Xw + E
        w = sigma_w * sp.randn(K, 1)
        Sigma = simulator.get_corr_matrix(K, corr_len_X)
        X = simulator.gaussian_samples(N, K, Sigma)
        E = sigma_eps * sp.randn(N, 1)
        Y = simulator.fwd_model(X, w, E)

        # Recover w
        w_hat = linear_reg.fit(X, Y, method='direct_inv')
        error = utils.get_relative_error(w_hat, w)
        error_list.append(error)

    print "K = %d, delta = %.2e, cond = %.1e, error = %.3f" % (
        K, delta, linalg.cond(X), np.mean(error_list))

