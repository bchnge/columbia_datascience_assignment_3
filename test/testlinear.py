import unittest
import sys
import numpy as np
import scipy as sp
from numpy import linalg
from numpy.testing import assert_allclose
from numpy.linalg.linalg import LinAlgError

from pandas import Series, DataFrame
from pandas.util.testing import assert_series_equal

import homework_03.src.utils as utils
import homework_03.src.linear_reg as linear_reg
import homework_03.src.cross_validator as cross_validator
import homework_03.src.simulator as simulator

"""
"""


class TestLinearReg(unittest.TestCase):
    """
    Tests the linear_reg module
    """
    def setUp(self):
        self.X = np.arange(6).reshape(3, 2)
        self.w = np.arange(10, 12).reshape(2, 1)
        self.Y = self.X.dot(self.w)
        self.singmx = np.ones((3,2))
        self.A = np.arange(4).reshape(2,2)
        self.Ainv = linalg.inv(self.A)

        self.w_series = Series(np.squeeze(self.w), ['x1', 'x2'],
                               dtype=float)
        self.X_frame = DataFrame(self.X,
                                 ['obs1', 'obs2', 'obs3'],
                                 ['x1', 'x2'])
        self.X_series = self.X_frame.irow(0);

        self.Y_frame = DataFrame(self.Y,
                                 ['obs1', 'obs2', 'obs3'])
        self.Y_series = self.Y_frame.icol(0);

    def test_solve_direct_inv_1(self):
        result = linear_reg._solve_direct_inv(self.X, self.Y, 0)
        assert_allclose(result, self.w, atol=1e-5)

    def test_solve_direct_inv_2(self):
        result = linear_reg._solve_direct_inv(self.X, self.Y, 1)
        benchmark = np.array([[  9.075 ], [ 11.3625]])
        assert_allclose(result, benchmark, atol=1e-4)

    def test_solve_direct_inv_3(self):
        result = linear_reg._solve_direct_inv(self.X, self.Y, 1e10)
        benchmark = np.array([[  0 ], [ 0 ]])
        assert_allclose(result, benchmark, atol=1e-3)

    def test_solve_direct_inv_4(self):
        fun = linear_reg._solve_direct_inv
        self.assertRaises(LinAlgError, fun, self.singmx, self.Y, 0)

    def test_solve_direct_inv_5(self):
        result = linear_reg._solve_direct_inv(self.X, self.Y, np.array([1, 2]))
        benchmark = np.array([[  12. ], [ 9. ]])
        assert_allclose(result, benchmark, atol=1e-4)

    def test_solve_pinv_1(self):
        result = linear_reg._solve_pinv(self.X, self.Y, 0)
        assert_allclose(result, self.w)

    def test_solve_pinv_2(self):
        result = linear_reg._solve_pinv(self.X, self.Y, 0.5)
        benchmark = np.array([[ 8.89872649 ], [ 11.82850154]])
        assert_allclose(result, benchmark, atol=1e-3)

    def test_solve_pinv_3(self):
        result = linear_reg._solve_pinv(self.X, self.Y, 1)
        benchmark = np.array([[  0 ], [ 0 ]])
        assert_allclose(result, benchmark, atol=1e-3)

    def test_solve_pinv_4(self):
        result = linear_reg._solve_pinv(self.singmx, self.Y, 0)
        benchmark = np.array([[  26.5 ], [ 26.5 ]])
        assert_allclose(result, benchmark, atol=1e-3)

    def test_fit_1(self):
        result = linear_reg.fit(self.X, self.Y)
        assert_allclose(result, self.w)

    def test_fit_pandas(self):
        result = linear_reg.fit(self.X_frame, self.Y_frame)
        assert_series_equal(result, self.w_series)

        result = linear_reg.fit(self.X_frame, self.Y_frame)
        assert_series_equal(result, self.w_series)

        result = linear_reg.fit(self.X_frame, self.Y_series)
        assert_series_equal(result, self.w_series)

        self.assertRaises(ValueError, linear_reg.fit,
                          self.X_series, self.Y_series)


    def test_fit_2(self):
        result = linear_reg.fit(self.X, self.Y, delta=1)
        benchmark = np.array([[  9.075 ], [ 11.3625]])
        assert_allclose(result, benchmark)

    def test_fit_3(self):
        X = np.array([1, 2, 3])
        Y = np.array([2])
        result = linear_reg.fit(X, Y, method='pinv')
        benchmark = np.array([[ 0.14285714], [ 0.28571429], [ 0.42857143]])
        assert_allclose(result, benchmark)

    def test_get_pinv_1(self):
        result = linear_reg._get_pinv(self.A, 0)
        assert_allclose(result, self.Ainv, atol=1e-5)

    def test_get_pinv_2(self):
        result = linear_reg._get_pinv(self.A, 1)
        assert_allclose(result, np.zeros(self.A.shape), atol=1e-5)

    def test_get_pinv_3(self):
        result = linear_reg._get_pinv(self.A, 0.5)
        benchmark = np.array(
            [[ 0.03262379,  0.1381966 ], [ 0.0527864 ,  0.2236068 ]])
        assert_allclose(result, benchmark, atol=1e-3)

    def test_get_pinv_4(self):
        result = linear_reg._get_pinv(self.singmx, 0.5)
        benchmark = 0.16666666 * np.ones((2, 3))
        assert_allclose(result, benchmark, atol=1e-3)

    def tearDown(self):
        pass


class TestUtils(unittest.TestCase):
    """
    Tests the utils module
    """
    def setUp(self):
        self.X = np.arange(6).reshape(3, 2)

    def test_get_relative_error(self):
        result = utils.get_relative_error(self.X, 2 * self.X)
        assert_allclose(result, 0.5, atol=1e-4)

    def tearDown(self):
        pass


class TestCrossValidator(unittest.TestCase):
    """
    Tests the cross_validator module
    """
    def setUp(self):
        self.X = np.arange(26).reshape(13, 2)
        self.w = np.arange(2).reshape(2, 1)
        self.Y = self.X.dot(self.w)

    def test_cross_val(self):
        delta_list = [0, 1, 2]
        result = cross_validator.cross_val(self.X, self.Y, delta_list)
        values_benchmark = np.array([[  3.44962492e-14,   7.89434618e-14],
           [  2.47233822e-02,   3.15966871e-02],
           [  3.67193138e-02,   2.69168719e-02]])
        assert_allclose(result.values, values_benchmark, atol=1e-5)
        self.assertEqual(result.columns.tolist(), ['train_error', 'cv_error'])
        self.assertEqual(result.index.tolist(), [0, 1, 2])

    def test_get_idx_list_1(self):
        result = cross_validator._get_idx_list(13)
        benchmark = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 13)]
        self.assertEqual(result, benchmark)

    def test_get_idx_list_2(self):
        self.assertRaises(ValueError, cross_validator._get_idx_list, 3)

    def test_get_xy_traincv(self):
        istart, istop = 2, 5
        Xtrain, Xcv, Ytrain, Ycv = cross_validator._get_xy_traincv(
            self.X, self.Y, istart, istop)

        Xtrain_benchmark = np.array([[4, 5], [6, 7], [8, 9]])
        assert_allclose(Xtrain, Xtrain_benchmark)

        Xcv_benchmark = np.array(
            [[ 0,  1], [ 2,  3], [10, 11], [12, 13], [14, 15],\
                    [16, 17], [18, 19], [20, 21], [22, 23], [24, 25]])
        assert_allclose(Xcv, Xcv_benchmark)

        Ytrain_benchmark = np.array([[5], [7], [9]])
        assert_allclose(Ytrain, Ytrain_benchmark)

        Ycv_benchmark = np.array(
            [[ 1], [ 3], [11], [13], [15], [17], [19], [21], [23], [25]])
        assert_allclose(Ycv, Ycv_benchmark)

    def tearDown(self):
        pass


class TestSimulator(unittest.TestCase):
    """
    Tests the simulator module
    """
    def setUp(self):
        pass

    def test_fwd_model(self):
        X = np.arange(6).reshape(3, 2)
        w = np.arange(2).reshape(2, 1)
        E = np.arange(3).reshape(3, 1)
        result = simulator.fwd_model(X, w, E=E)
        benchmark = np.array([[1], [4], [7]])
        assert_allclose(result, benchmark)

    def test_get_corr_matrix(self):
        K = 3
        corr_len = 1
        result = simulator.get_corr_matrix(K, corr_len)
        benchmark = np.array(
           [[ 1.        ,  0.36787944,  0.13533528],
           [ 0.36787944,  1.        ,  0.36787944],
           [ 0.13533528,  0.36787944,  1.        ]])
        assert_allclose(result, benchmark, atol=1e-4)

    def test_get_gaussian_samples(self):
        N = 1000000
        K = 2
        Sigma = np.array([[1, 0.5], [0.5, 1]])
        samples = simulator.gaussian_samples(N, K, Sigma)
        result = np.cov(samples.T)
        assert_allclose(result, Sigma, atol=1e-2)

def gaussian_samples(N, K, Sigma=None):

    def tearDown(self):
        pass
