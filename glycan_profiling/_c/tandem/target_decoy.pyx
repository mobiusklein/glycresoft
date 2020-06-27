cimport cython

from cpython.list cimport PyList_GetItem

cimport numpy as np
import numpy as np
np.import_array()
from numpy.math cimport isnan


cdef class ScoreCell(object):
    cdef:
        public double score
        public double value

    def __init__(self, score, value):
        self.score = score
        self.value = value

    def __getitem__(self, i):
        if i == 0:
            return self.score
        elif i == 1:
            return self.value
        else:
            raise ValueError(i)

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.score
        yield self.value

    def __repr__(self):
        return "{self.__class__.__name__}({self.score}, {self.value})".format(self=self)


cdef class NearestValueLookUp(object):
    cdef:
        public list items

    def __init__(self, items):
        if isinstance(items, dict):
            items = items.items()
        self.items = sorted([ScoreCell(*x) for x in items if not np.isnan(x[0])], key=lambda x: x[0])

    cpdef Py_ssize_t _find_closest_item(self, double value):
        cdef:
            list array
            Py_ssize_t lo, hi, n, i, mid, best_index
            ScoreCell xc
            double error_tolerance, err, best_error, x

        array = self.items
        lo = 0
        hi = len(array)
        n = hi

        error_tolerance = 1e-3

        if isnan(value):
            return lo

        if lo == hi:
            return lo

        while hi - lo:
            i = (hi + lo) // 2
            xc = <ScoreCell>PyList_GetItem(array, i)
            x = xc.score
            err = x - value
            if abs(err) < error_tolerance:
                mid = i
                best_index = mid
                best_error = abs(err)
                i = mid - 1
                while i >= 0:
                    xc = <ScoreCell>PyList_GetItem(array, i)
                    x = xc.score
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    xc = <ScoreCell>PyList_GetItem(array, i)
                    x = xc.score
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif (hi - lo) == 1:
                mid = i
                best_index = mid
                best_error = abs(err)
                i = mid - 1
                while i >= 0:
                    xc = <ScoreCell>PyList_GetItem(array, i)
                    x = xc.score
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    xc = <ScoreCell>PyList_GetItem(array, i)
                    x = xc.score
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif x < value:
                lo = i
            elif x > value:
                hi = i

    cpdef get_pair(self, double key):
        return self.items[self._find_closest_item(key) + 1]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return "{s.__class__.__name__}({size})".format(
            s=self, size=len(self))

    def __getitem__(self, key):
        return self._get_one(key)

    def _get_sequence(self, key):
        value = [self._get_one(k) for k in key]
        if isinstance(key, np.ndarray):
            value = np.array(value, dtype=float)
        return value

    cpdef _get_one(self, double key):
        ix = self._find_closest_item(key)
        if ix >= len(self):
            ix = len(self) - 1
        if ix < 0:
            ix = 0
        try:
            pair = self.items[ix]
        except IndexError:
            print("IndexError in %r with index %r and query %r" % (self, ix, key))
            print(self.items)
            raise
        return pair.value