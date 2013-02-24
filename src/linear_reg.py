import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame
from numpy import linalg
from numpy import maximum
from numpy import dot
from numpy import transpose
from numpy import multiply
from numpy import newaxis
#from numpy import wrap

import homework_03.src.utils as utils

"""
Module for fitting and prediction using a linear regression module.
"""

def fit(X, Y, method='direct_inv', **kwargs):
    """
    Parameters
    ----------
    X : N x K numpy array
    Y : N x 1 numpy array
    method : String, either 'direct_inv', 'pinv', or 'cg'
    delta : Nonnegative scalar or K x 1 numpy array
    **kwargs : Extra arguments to be passed to the solvers

    Returns
    -------
    w: Kx1 parameters that minimizes |Xw - Y|^2 + delta |w|^2.

    Notes
    -----
    If X is a 1d array it is converted to a 1xK row vector
    If Y is a 1d array it is converted to a Nx1 column vector
    If X and/or Y are pandas objects, utils.process_input handles things.
        Students:  I know I am safe doing this only in fit() since fit() is the
        only public method of this module.
    """
    _out_wrapper, x_val, y_val = utils.process_input(X, Y)

    # Call the appropriate private method to get results
    method_dict = {'direct_inv': _solve_direct_inv, 'pinv': _solve_pinv}
    results = method_dict[method](x_val, y_val, **kwargs)

    return _out_wrapper(results)


def _solve_direct_inv(X, Y, delta=0):
    """
    Finds the minimizer w of |Xw - Y| by attempting to solve the normal
    equation: (X^TX + delta I) w = X^TY by inverting (X^TX + delta I).

    Parameters
    ----------
    X : N x K numpy array
    Y : N x 1 numpy array
    delta : Scalar or length K numpy vector
        The regularization parameter.  If a vector, then use delta * np.eye(K)

    Notes
    -----
    Uses numpy's linalg module to do the inversion.
    """
    
    matrix_to_invert = X.T.dot(X) + delta*np.eye(X.shape[1])
    return (linalg.inv(matrix_to_invert).dot(X.T)).dot(Y)

def _solve_pinv(X, Y, cutoff=0):
    """
    Uses a pseudo inverse to approximately solve Xw = Y.

    Parameters
    ----------
    X : N x K numpy array
    Y : N x 1 numpy array
    cutoff : Nonnegative real
        Don't try to invert singular dimensions associated with singular
        values less than or equal to cutoff times the maximal singular value.
    """
    
    return _get_pinv(X,cutoff).dot(Y)
    #pass


def _get_pinv(X, cutoff):
    """
    Gets the Moore-Penrose pseudoinverse with appropriate cutoff.

    Notes
    -----
    Uses numpy's linalg.svd function to get an svd, then explicitly forms
    the pseudoinverse.  DON'T Just call numpy's pinv function!!!

    Due to our definition of cutoff, cutoff >= 1 should result in the 0 matrix
    being returned, and cutoff == 0 should result in the pseudo inverse.
    """
    # Get the svd.  Hint: You can set full_matrices=False.
    U, s, V = np.linalg.svd(X, full_matrices=False)
  
    # Form S_inv, which has 1/S[i] for it's ii entry (provided this entry
    # does not get cut off.
    """
    sigmas = []

    for sigma in s:
        if abs(sigma)>cutoff:
            sigmas.append(sigma)

  #  print sigmas
    sigmasnp=np.array(sigmas)
    S_inv=np.diag(1/sigmasnp)

    #return linalg.pinv(X,cutoff)
#    print S_inv
 #   print V.T[0:S_inv.shape[0]]
    # Form the pseudoinverse
    return V.T[0:S_inv.shape[0]].dot(S_inv).dot(U.T[0:S_inv.shape[0]])

    #pass
    """
    m = U.shape[0]
    n = V.shape[1]
    cutoff = cutoff*maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.;
    res = dot(transpose(V), multiply(s[:, newaxis],transpose(U)))
    return (res)