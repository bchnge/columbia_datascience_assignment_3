import numpy as np
import scipy as sp
import pandas as pd
from numpy import linalg

import homework_03.src.utils as utils


def get_corr_matrix(K, corr_len):
    """
    Returns Sigma, a K x K correlation matrix.

    Parameters
    ----------
    K : Nonnegative integer
        Returned matrix is K x K
    corr_len : Nonnegative real
        Sigma[i, j] = np.exp(-|i - j| / corr_len)
        If corr_len == 0, then return uncorrelated columns.
    """
    ## The following might be helpful for you to verify that you're
    # doing stuff right:
    # The covariance of columns of X is sp.cov(X.T)
    #cov_error = utils.get_relative_error(sigma, sp.cov(X.T))

    # Start with an identity then populate off diagonal entries
    #pass
    Sigma = np.eye(K)

    for i in range(Sigma.shape[0]):
        for j in range(Sigma.shape[1]):
            Sigma[i][j]=np.exp(-abs(i-j)/corr_len)
    return Sigma



def gaussian_samples(N, K, Sigma=None):
    """
    Returns an N x K Gaussian matrix where the correlation of the columns
    (as N -> infinity) is Sigma.

    Parameters
    ----------
    N, K : Returned matrix is N x K
    Sigma : Positive definite matrix
        Correlation matrix of the columns
        If None, return uncorrelated columns.
    """
    ## The following might be helpful for you to verify that you're
    # doing stuff right:
    # The covariance of columns of X is sp.cov(X.T)
    #cov_error = utils.get_relative_error(sigma, sp.cov(X.T))

    # Start with an identity then populate off diagonal entries
    #pass
    mean = []
 
    for i in range(K):
        mean.append(0)

    newSigma=[]
    newSigma=Sigma

    if newSigma==None:
        newSigma=np.eye(K)

    return np.random.multivariate_normal(mean,newSigma,(N))    


def fwd_model(X, w, E=0):
    """
    Return the "forward model" results Y = Xw + E

    Parameters
    ----------
    X : N x K numpy array or DataFrame or Series
    w : K x 1 numpy array or DataFrame or Series
    E : Scalar or N x 1 numpy array ONLY!

    Returns
    -------
    Y : N x 1 numpy array or DataFrame or Series
    """
    _out_wrapper, X_T, w = utils.process_input(X.T, w)
    X = X_T.T

    N, K = X.shape

    result = X.dot(w).reshape(N, 1) + E
    
    return _out_wrapper(result)
