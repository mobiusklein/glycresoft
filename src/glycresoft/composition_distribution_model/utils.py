from __future__ import print_function

import numpy as np
from scipy.linalg import lapack


def display_table(names, values, sigfig=3, filter_empty=1, print_fn=None):
    if print_fn is None:
        print_fn = print
    values = np.array(values)
    maxlen = len(max(names, key=len)) + 2
    fstring = ("%%0.%df" % sigfig)
    for i in range(len(values)):
        if values[i, :].sum() or not filter_empty:
            print_fn(names[i].ljust(maxlen) + ('|'.join([fstring % f for f in values[i, :]])))


# Cholesky decomposition-based inverse based upon
# https://stackoverflow.com/a/58719188/1137920

inds_cache = {}


def _upper_triangular_to_symmetric(ut: np.ndarray):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fast_positive_definite_inverse(M: np.ndarray) -> np.ndarray:
    if M.size == 0:
        return np.zeros_like(M)

    # Compute the Cholesky decomposition of a symmetric positive-definite matrix
    cholesky, info = lapack.dpotrf(M)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(M))
    # Compute the inverse of `M` from its Cholesky decomposition
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    _upper_triangular_to_symmetric(inv)
    return inv
