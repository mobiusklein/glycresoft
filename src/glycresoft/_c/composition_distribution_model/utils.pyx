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


# A replacement for a namedtuple
cdef class GlycanPriorRecord(object):

    def __init__(self, score, matched):
        self.score = score
        self.matched = matched

    def __reduce__(self):
        return self.__class__, (self.score, self.matched)

    def __getitem__(self, i):
        if i == 0:
            return self.score
        elif i == 1:
            return self.matched
        else:
            raise IndexError(i)

    def __len__(self):
        return 2

    def __repr__(self):
        return "{self.__class__.__name__}({self.score}, {self.matched})".format(self=self)