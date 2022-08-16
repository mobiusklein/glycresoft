# Excerpted from scipy.stats._kde and .stats._stats to allow releasing the GIL

from libc cimport math
cimport cython
cimport numpy as np

from numpy.math cimport PI
from numpy.math cimport INFINITY
from numpy.math cimport NAN
from numpy cimport ndarray, int64_t, float64_t, intp_t

import warnings

import numpy as np
import scipy.stats, scipy.special


ctypedef fused real:
    float
    double
    long double


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline int gaussian_kernel_estimate_inner(
    real[:, :] points_,
    real[:, :] values_,
    real[:, :] xi_,
    real[:, :] estimate,
    real[:, :] whitening,
    int n,
    int m,
    int d,
    int p,
) nogil:
    cdef:
        int i, j, k
        real residual, arg, norm

    # Evaluate the normalisation
    norm = math.pow((2 * PI), (- d / 2.))
    for i in range(d):
        norm *= whitening[i, i]

    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])
                arg += residual * residual

            arg = math.exp(-arg / 2.) * norm
            for k in range(p):
                estimate[j, k] += values_[i, k] * arg
    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
def gaussian_kernel_estimate(points, values, xi, precision, dtype, real _=0):
    """
    def gaussian_kernel_estimate(points, real[:, :] values, xi, precision)
    Evaluate a multivariate Gaussian kernel estimate.
    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimensions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    precision : array_like with shape (d, d)
        Precision matrix for the Gaussian kernel.
    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    cdef:
        real[:, :] points_, xi_, values_, estimate, whitening
        int i, j, k
        int n, d, m, p
        real arg, residual, norm

    n = points.shape[0]
    d = points.shape[1]
    m = xi.shape[0]
    p = values.shape[1]

    if xi.shape[1] != d:
        raise ValueError("points and xi must have same trailing dim")
    if precision.shape[0] != d or precision.shape[1] != d:
        raise ValueError("precision matrix must match data dims")

    # Rescale the data
    whitening = np.linalg.cholesky(precision).astype(dtype, copy=False)
    points_ = np.dot(points, whitening).astype(dtype, copy=False)
    xi_ = np.dot(xi, whitening).astype(dtype, copy=False)
    values_ = values.astype(dtype, copy=False)

    # Create the result array and evaluate the weighted sum
    estimate = np.zeros((m, p), dtype)

    with nogil:
        gaussian_kernel_estimate_inner(
            points_,
            values_,
            xi_,
            estimate,
            whitening,
            n,
            m,
            d,
            p,
        )

    return np.asarray(estimate)


@cython.binding(True)
def evaluate_gaussian_kde(self, points):
    """Evaluate the estimated pdf on a set of points.
    Parameters
    ----------
    points : (# of dimensions, # of points)-array
        Alternatively, a (# of dimensions,) vector can be passed in and
        treated as a single point.
    Returns
    -------
    values : (# of points,)-array
        The values at each point.
    Raises
    ------
    ValueError : if the dimensionality of the input points is different than
                    the dimensionality of the KDE.
    """
    points = np.atleast_2d(np.asarray(points))

    d, m = points.shape
    if d != self.d:
        if d == 1 and m == self.d:
            # points was passed in as a row vector
            points = np.reshape(points, (self.d, 1))
            m = 1
        else:
            msg = "points have dimension %s, dataset has dimension %s" % (d,
                self.d)
            raise ValueError(msg)

    output_dtype = np.common_type(self.covariance, points)
    itemsize = np.dtype(output_dtype).itemsize
    if itemsize == 4:
        spec = 'float'
    elif itemsize == 8:
        spec = 'double'
    elif itemsize in (12, 16):
        spec = 'long double'
    else:
        raise TypeError('%s has unexpected item size %d' %
                        (output_dtype, itemsize))
    result = gaussian_kernel_estimate[spec](self.dataset.T, self.weights[:, None],
                                            points.T, self.inv_cov, output_dtype)
    return result[:, 0]