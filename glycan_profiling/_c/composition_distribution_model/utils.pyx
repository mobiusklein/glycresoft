cimport cython
cimport numpy as np


@cython.boundscheck(False)
cpdef bint is_diagonal(np.ndarray[double, ndim=2] mat):
    cdef:
        size_t i, j, count
        size_t n, m
    n = mat.shape[0]
    m = mat.shape[1]
    if n != m:
        return False
    count = 0
    for i in range(n):
        for j in range(m):
            if i == j:
                continue
            count += mat[i, j] != 0
    return count == 0