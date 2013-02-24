import numpy as np
import scipy as sp
import pandas as pd
from numpy import linalg
from numpy import sqrt
from pandas import Series, DataFrame


def get_relative_error(estimate, reality):
    """
    Compares estimate to reality and returns the the mean-square error:

    |estimate - reality|_F / |reality|_F  where F is the Frobenius norm.
    """
    #pass

    numerator=0
    denominator=0

    for i in range(estimate.shape[0]):
        numerator+=(estimate[i]-reality[i])**2
        denominator+=(reality[i])**2

    return sqrt(numerator)/sqrt(denominator)



def process_input(X, Y):
    """
    Allows one to handle input that is either numpy arrays or pandas
    DataFrames/Series.

    If both are pandas objects, assumes we want the rows (index) of X and Y to
    match.

    Returns
    -------
    _out_wrapper : Function
        Takes in N x 1 np.array, returns Series if both X and Y were DataFrame
        and/or Series, and returns a K x 1 np.ndarray if either X or Y was a
        np.ndarray
    x_val : N x K np.ndarray
    y_val : N x 1 np.ndarray
    """
    if isinstance(Y, Series):
        Y = DataFrame({0: Y})
    if isinstance(X, Series):
        idx = 0
        if hasattr(Y, 'index'):
            idx = Y.index[0]
        X = DataFrame.from_dict({idx: X}, orient='index')

    is_x_frame = isinstance(X, DataFrame)
    is_y_frame = isinstance(Y, DataFrame)

    # If both X and Y are DataFrame, then align them, make a reuturn handler,
    # and extract their values.
    if is_x_frame:
        variables = X.columns
        if is_y_frame:
            X, Y = _check_dim(X, Y)
            X, Y = _align(X, Y)
            Y = Y.values
        X, Y = _reshape_dim(X, Y)
        X, Y = _check_dim(X.values, Y)
        return lambda Z: Series(np.squeeze(Z), variables), X, Y

    # Else if only Y is a DataFrame, extract the values
    elif is_y_frame:
        Y = Y.values

    X, Y = _reshape_dim(X, Y)
    X, Y = _check_dim(X, Y)
    return lambda x:x, X, Y


def _check_dim(X, Y):
    """
    Checks the dimensions of X, Y and raises ValueError if they are wrong.

    Parameters
    ----------
    X, Y : Pandas DataFrame
    """
    if len(X) != len(Y):
        raise ValueError('X and Y must have same length')

    if Y.shape[1] > 1:
        raise ValueError('Y must be a single column')

    return X, Y


def _reshape_dim(X, Y):
    """
    Adds extra dimensions to X, Y if necessary.

    Parameters
    ----------
    X, Y : np.ndarray

    Returns
    -------
    X, Y : Reshaped into (N x K) and (N x 1) np.ndarray
    """
    if X.ndim < 2:
        X = X[np.newaxis, :]  #newaxis is just None actually

    if Y.ndim < 2:
        Y = Y[:, np.newaxis]

    return X, Y


def _align(X, Y):
    """
    Re-index Y so that it has the same index as X.

    Parameters
    ----------
    X, Y : Pandas DataFrames or Series

    Returns
    -------
    X, Y : Reindexed version
    """
    return X, Y.reindex(X.index)
