cimport cython

cimport numpy as np

np.import_array()


@cython.freelist(100000)
cdef class SplittingPoint(object):
    cdef:
        public double first_maximum
        public double minimum
        public double second_maximum
        public double minimum_index
        public double total_distance

    @staticmethod
    cdef SplittingPoint _create(double first_maximum, double minimum, double second_maximum, double minimum_index):
        cdef:
            SplittingPoint inst
        inst = SplittingPoint.__new__(SplittingPoint)
        inst.first_maximum = first_maximum
        inst.minimum = minimum
        inst.second_maximum = second_maximum
        inst.minimum_index = minimum_index
        inst.total_distance = inst.compute_distance()
        return inst

    def __init__(self, first_maximum, minimum, second_maximum, minimum_index):
        self.first_maximum = first_maximum
        self.minimum = minimum
        self.second_maximum = second_maximum
        self.minimum_index = minimum_index
        self.total_distance = self.compute_distance()

    def __reduce__(self):
        return self.__class__, (self.first_maximum, self.minimum, self.second_maximum, self.minimum_index)

    cpdef double compute_distance(self):
        return (self.first_maximum - self.minimum) + (self.second_maximum - self.minimum)

    def __repr__(self):
        return "SplittingPoint(%0.4f, %0.4f, %0.4f, %0.2f, %0.3e)" % (
            self.first_maximum, self.minimum, self.second_maximum, self.minimum_index, self.total_distance)


@cython.binding
@cython.boundscheck(False)
def locate_extrema(self, np.ndarray[double, ndim=1, mode='c'] xs=None, np.ndarray[double, ndim=1, mode='c']ys=None):
    cdef:
        np.ndarray[np.int64_t, ndim=1, mode='c'] maxima_indices
        np.ndarray[np.int64_t, ndim=1, mode='c'] minima_indices
        list candidates
        size_t i, j, k, max_i, max_j, min_k, n_maxima, n_minima
        double y_i, y_j, y_k
        SplittingPoint point

    if xs is None:
        xs = self.xs
    if ys is None:
        ys = self.ys

    maxima_indices, minima_indices = self._extreme_indices(ys)
    candidates = []

    n_maxima = len(maxima_indices)
    n_minima = len(minima_indices)

    for i in range(n_maxima):
        max_i = maxima_indices[i]
        for j in range(i + 1, n_maxima):
            max_j = maxima_indices[j]
            for k in range(n_minima):
                min_k = minima_indices[k]
                y_i = ys[max_i]
                y_j = ys[max_j]
                y_k = ys[min_k]
                if max_i < min_k < max_j and (y_i - y_k) > (y_i * 0.01) and (
                        y_j - y_k) > (y_j * 0.01):
                    point = SplittingPoint._create(y_i, y_k, y_j, xs[min_k])
                    candidates.append(point)

    if candidates:
        best_point = max(candidates, key=lambda x: x.total_distance)
        self.partition_sites.append(best_point)

    return candidates
