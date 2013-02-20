import numpy as np
import scipy as sp
import pandas as pd

import homework_03.src.linear_reg as linear_reg
import homework_03.src.simulator as simulator
import homework_03.src.utils as utils

"""
Module for performing 5-fold cross validation over a one-dimensional parameter
delta.  
"""

def cross_val(X, Y, delta_list):
    """
    Returns goodness of fit metrics for delta in delta_list as a pandas
    DataFrame.  Uses the 'direct_inv' method to solve the problem.

    Parameters
    ----------
    X : N x K np.ndarray
    Y : N x 1 np.ndarray
    delta_list : List of nonnegative numbers
        Values of delta to use
    """
    # Get np.ndarray versions of both
    _out_wrapper, X, Y = utils.process_input(X, Y)
    # Get the idx_list to create the folds
    #N, K = X.shape
    #fold_idx_list = _get_idx_list(N)
    N, K = X.shape
    fold_idx_list = _get_idx_list(N)

    # Initialize your goodness-of-fit ("gof") DataFrame:
    D = len(delta_list)
    initial_errors = np.ones((D, 2)) * np.nan
    gof = pd.DataFrame(
        initial_errors, index=delta_list, columns=['train_error', 'cv_error'])
    gof.index.name = 'delta'

    # Loop through delta_list
    for delta in delta_list:
        cv_errs_4_this_delta = []
        train_errs_4_this_delta = []
        # Loop throug the fold_idx_list
        # Average the errors for each fold to get the entries in gof
        # To compute error use utils.get_relative_error()
        #
        # Use:
        #     linear_reg.fit
        #     _get_xy_traincv
        #     utils.get_relative_error
        #     simulator.fwd_model

        gof.ix[delta, 'cv_error'] = np.mean(cv_errs_4_this_delta)
        gof.ix[delta, 'train_error'] = np.mean(train_errs_4_this_delta)

    return gof


def _get_xy_traincv(X, Y, istart, istop):
    """
    Returns slices of X and Y used for training and cv.  The training
    set should be e.g.:  X[istart: istop, :], and the cv set should
    be everything else.

    Parameters
    ----------
    X, Y : np.ndarrays, ndim=2
    istart, istop : Nonnegative integers
        Indexes into the rows of the X, Y arrays

    Returns
    -------
    The tuple (Xtrain, Xcv, Ytrain, Ycv)
    """
    # Hint:  Use np.concatenate
    pass


def _get_idx_list(N):
    """
    Returns a list of 6 integers between 0 and N-1.  They can be used to 
    divide {0,...,N-1} up into 5 slices, the first four having equal size,
    and the last one taking up any remainder.  

    Parameters
    ----------
    N : Integer >= 5

    Example
    -------
    >>> idx_list = _get_idx_list(12)
    >>> print idx_list
    [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
    >>> first_slice = X[idx_list[0][0]: idx_list[0][1]]
    >>> second_slice = X[idx_list[1][0]: idx_list[1][1]]
    """
    # Hint: Use lists and tuples
    pass