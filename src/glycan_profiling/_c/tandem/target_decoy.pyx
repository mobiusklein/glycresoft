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

    def __eq__(self, other):
        return abs(self.score - other.score) < 1e-12 and abs(self.value - other.value) < 1e-12

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.score
        yield self.value

    def __repr__(self):
        return "{self.__class__.__name__}({self.score}, {self.value})".format(self=self)

    def __reduce__(self):
        return self.__class__, (self.score, self.value)


cpdef __pyx_unpickle_ScoreCell(tp, checksum, state):
    return tp(*state)


cdef class NearestValueLookUp(object):
    '''A mapping-like object which simplifies
    finding the value of a pair whose key is nearest
    to a given query.

    .. note::
        Queries exceeding the maximum key will return
        the maximum key's value.
    '''
    cdef:
        public list items

    def __init__(self, items):
        if isinstance(items, dict):
            items = items.items()
        self.items = self._transform_items(items)

    def __eq__(self, other):
        return self.items == other.items

    def __ne__(self, other):
        return not self == other

    def _transform_items(self, items):
        return sorted([ScoreCell(*x) for x in items if not np.isnan(x[0])], key=lambda x: x[0])

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self.items = self._transform_items(state[0])
        elif isinstance(state, dict):
            self.items = self._transform_items(state['items'])
        else:
            raise

    def __getstate__(self):
        return {
            "items": [tuple(i) for i in self.items]
        }

    cpdef max_key(self):
        cdef:
            Py_ssize_t n
            ScoreCell cell
        n = len(self.items)
        if n == 0:
            return 0
        cell = <ScoreCell>PyList_GetItem(self.items, n - 1)
        return cell.score

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
        pair = self.items[ix]
        return pair.value