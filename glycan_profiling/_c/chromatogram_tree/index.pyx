cimport cython
from cpython.list cimport PyList_Append, PyList_Size, PyList_GetItem


@cython.freelist(1000000)
cdef class MassObject(object):

    def __init__(self, obj, mass):
        self.obj = obj
        self.mass = mass

    def __repr__(self):
        return "MassObject(%r, %r)" % (self.obj, self.mass)

    @classmethod
    def from_chromatogram(cls, obj):
        return cls(obj, obj.weighted_neutral_mass)


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef (ssize_t, bint) binary_search_with_flag(list array, double mass, double error_tolerance=1e-5):
    """Binary search an ordered array of objects with :attr:`neutral_mass`
    using a PPM error tolerance of `error_toler

    Parameters
    ----------
    array : list
        An list of objects, sorted over :attr:`neutral_mass` in increasing order
    mass : float
        The mass to search for
    error_tolerance : float, optional
        The PPM error tolerance to use when deciding whether a match has been found

    Returns
    -------
    int:
        The index in `array` of the best match
    bool:
        Whether or not a match was actually found, used to
        signal behavior to the caller.
    """
    cdef:
        ssize_t lo, hi, n, mid, best_index, i
        double err, best_error
        MassObject x
    lo = 0
    n = hi = len(array)
    while hi != lo:
        mid = (hi + lo) // 2
        x = <MassObject>PyList_GetItem(array, mid)
        err = (x.mass - mass) / mass
        if abs(err) <= error_tolerance:
            best_index = mid
            best_error = abs(err)
            i = mid - 1
            while i >= 0:
                x = <MassObject>PyList_GetItem(array, i)
                err = abs((x.mass - mass) / mass)
                if err < best_error:
                    best_error = err
                    best_index = i
                elif err > error_tolerance:
                    break
                i -= 1

            i = mid + 1
            while i < n:
                x = <MassObject>PyList_GetItem(array, i)
                err = abs((x.mass - mass) / mass)
                if err < best_error:
                    best_error = err
                    best_index = i
                elif err > error_tolerance:
                    break
                i += 1
            return best_index, True
        elif (hi - lo) == 1:
            return mid, False
        elif err > 0:
            hi = mid
        elif err < 0:
            lo = mid
    return 0, False


cdef class ChromatogramIndex(object):

    def __init__(self, chromatograms, sort=True):
        if sort:
            self._sort(chromatograms)
        else:
            self._chromatograms = list(map(MassObject.from_chromatogram, chromatograms))

    def _sort(self, iterable):
        self._chromatograms = [
            MassObject.from_chromatogram(c) for c in sorted(
                [c for c in iterable if len(c)], key=lambda x: (x.neutral_mass, x.start_time))]

    def add(self, chromatogram, sort=True):
        self._chromatograms.append(MassObject.from_chromatogram(chromatogram))
        if sort:
            self._sort(self.chromatograms)

    def __len__(self):
        return self.get_size()

    def __getitem__(self, i):
        cdef:
            MassObject obj
            slice si
            size_t j, n
            list slice_out, slice_in
        if isinstance(i, slice):
            si = <slice>i
            slice_out = []
            slice_in = self._chromatograms[si]
            n = PyList_Size(slice_in)
            for j in range(n):
                PyList_Append(slice_out, (<MassObject>PyList_GetItem(slice_in, j)).obj)
            return slice_out
        obj = <MassObject>PyList_GetItem(self._chromatograms, i)
        return obj.obj

    cdef inline MassObject get_index(self, size_t i):
        obj = <MassObject>PyList_GetItem(self._chromatograms, i)
        return obj

    cdef inline size_t get_size(self):
        return PyList_Size(self._chromatograms)

    def __iter__(self):
        cdef:
            size_t i, n
            MassObject obj
        n = self.get_size()
        for i in range(n):
            obj = <MassObject>PyList_GetItem(self._chromatograms, i)
            yield obj.obj

    @property
    def chromatograms(self):
        return self._chromatograms

    @chromatograms.setter
    def chromatograms(self, value):
        self._chromatograms = list(map(MassObject.from_chromatogram, value))

    cpdef (ssize_t, bint) _binary_search(self, double mass, double error_tolerance=1e-5):
        return binary_search_with_flag(self._chromatograms, mass, error_tolerance)

    cpdef object find_mass(self, double mass, double ppm_error_tolerance=1e-5):
        index, flag = self._binary_search(mass, ppm_error_tolerance)
        if flag:
            return self[index]
        else:
            return None

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef object find_all_by_mass(self, double mass, double ppm_error_tolerance=1e-5):
        cdef:
            ssize_t i, n, center_index, low_index, high_index
            bint flag
            list items
            MassObject x

        n = self.get_size()
        if n == 0:
            return self.__class__([], sort=False)
        center_index, flag = self._binary_search(mass, ppm_error_tolerance)
        low_index = center_index
        while low_index > 0:
            x = self.get_index(low_index - 1)
            if abs((mass - x.mass) / x.mass) > ppm_error_tolerance:
                break
            low_index -= 1

        high_index = center_index - 1
        while high_index < n - 1:
            x = self.get_index(high_index + 1)
            if abs((mass - x.mass) / x.mass) > ppm_error_tolerance:
                break
            high_index += 1

        if low_index == high_index == center_index:
            x = self.get_index(center_index)
            if abs((mass - x.mass) / x.mass) > ppm_error_tolerance:
                return self.__class__([], sort=False)

        items = []
        for i in range(low_index, high_index + 1):
            x = self.get_index(i)
            if (abs(x.mass - mass) / mass) < ppm_error_tolerance:
                PyList_Append(items, x.obj)

        return self.__class__(items, sort=False)

    cpdef object mass_between(self, double low, double high):
        cdef:
            size_t n
            ssize_t low_index, high_index
        n = self.get_size()
        if n == 0:
            return self.__class__([])
        low_index, flag = self._binary_search(low, 1e-5)
        low_index = max(0, min(low_index, n - 1))
        if self.get_index(low_index).mass  < low:
            low_index += 1
        high_index, flag = self._binary_search(high, 1e-5)
        high_index += 2
        # high_index = min(n - 1, high_index)
        if (high_index < n) and (self.get_index(high_index).mass > high):
            high_index -= 1
        items = self[low_index:high_index]
        items = [c for c in items if low <= c.neutral_mass <= high]
        return self.__class__(items, sort=False)

    def __repr__(self):
        return repr(list(self))

    def _repr_pretty_(self, p, cycle):
        return p.pretty(list(self))

    def __str__(self):
        return str(list(self))
