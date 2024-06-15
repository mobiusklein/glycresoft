# cython: embedsignature=True

import operator

cimport cython
from cpython.list cimport PyList_Append, PyList_Size, PyList_GetItem


@cython.freelist(1000000)
cdef class MassObject(object):

    def __init__(self, obj, mass):
        self.obj = obj
        self.mass = mass

    def __repr__(self):
        return "MassObject(%r, %r)" % (self.obj, self.mass)

    def __lt__(self, other):
        return self.mass < other.mass

    def __gt__(self, other):
        return self.mass > other.mass

    def __eq__(self, other):
        return abs(self.mass - other.mass) < 1e-3

    def __ne__(self, other):
        return abs(self.mass - other.mass) >= 1e-3


def identity(x):
    return x


cdef class NeutralMassDatabaseImpl(object):

    def __init__(self, structures, mass_getter=operator.attrgetter("calculated_mass"), sort=True):
        self.mass_getter = mass_getter
        self.structures = self._prepare(structures, sort=sort)

    cdef list _prepare(self, list structures, bint sort=True):
        cdef:
            size_t i, n
            list translated
            MassObject mo
            object o

        translated = []
        n = PyList_Size(structures)
        for i in range(n):
            o = structures[i]
            mo = MassObject(o, self.mass_getter(o))
            translated.append(mo)
        if sort:
            translated.sort()
        return translated

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef size_t search_binary(self, double mass, double error_tolerance=1e-6):
        """Search within :attr:`structures` for the index of a structure
        with a mass nearest to `mass`, within `error_tolerance`

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The approximate error tolerance to accept

        Returns
        -------
        int
            The index of the structure with the nearest mass
        """
        cdef:
            ssize_t lo, hi, n, mid, best_index, i
            double err, best_error
            MassObject x
            list array
        array = self.structures
        lo = 0
        n = hi = PyList_Size(array)
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
                return best_index
            elif (hi - lo) == 1:
                best_index = mid
                best_error = err
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
                return best_index
            elif err > 0:
                hi = mid
            elif err < 0:
                lo = mid
        return 0

    def __len__(self):
        return self.get_size()

    def __iter__(self):
        for obj in self.structures:
            yield obj.obj

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [mo.obj for mo in self.structures[i]]

        if i < 0:
            i = self.get_size() + i
        return self.get_item(i)

    def __reduce__(self):
        return self.__class__, ([], identity, False), self.__getstate__()

    @property
    def lowest_mass(self):
        return self.structures[0].mass

    @property
    def highest_mass(self):
        return self.structures[-1].mass

    cdef size_t get_size(self):
        return PyList_Size(self.structures)

    cdef object get_item(self, size_t i):
        cdef:
            MassObject mo
        mo = <MassObject>PyList_GetItem(self.structures, i)
        return mo.obj

    cpdef list search_mass(self, double mass, double error_tolerance=0.1):
        """Search for the set of all items in :attr:`structures` within `error_tolerance` Da
        of the queried `mass`.

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The range of mass errors (in Daltons) to allow

        Returns
        -------
        list
            The list of instances which meet the criterion
        """
        cdef:
            double lo_mass, hi_mass
            size_t lo, hi, i
            list result
            MassObject mo

        if self.get_size() == 0:
            return []
        lo_mass = mass - error_tolerance
        hi_mass = mass + error_tolerance
        lo = self.search_binary(lo_mass)
        hi = self.search_binary(hi_mass) + 1

        result = []
        for i in range(lo, hi):
            mo = <MassObject>PyList_GetItem(self.structures, i)
            if lo_mass <= mo.mass <= hi_mass:
                result.append(mo.obj)
        return result

    cpdef object search_between(self, double lower, double higher):
        cdef:
            size_t lo, hi, i
            list result
            MassObject mo
            NeutralMassDatabaseImpl inst

        if self.get_size() == 0:
            return []
        lo = self.search_binary(lower)
        hi = self.search_binary(higher) + 1

        result = []
        for i in range(lo, hi):
            mo = <MassObject>PyList_GetItem(self.structures, i)
            if lower <= mo.mass <= higher:
                result.append(mo)
        inst = self.__class__(
            [],
            self.mass_getter, sort=False)
        inst.structures = result
        return inst
