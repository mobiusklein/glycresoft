from cpython cimport PyTuple_GET_ITEM, PyTuple_GET_SIZE, PyInt_AsLong

from glypy._c.structure.glycan_composition cimport _CompositionBase

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue
from glycopeptidepy.structure.glycan import GlycanCompositionProxy as _GlycanCompositionProxy

from_iupac_lite = FrozenMonosaccharideResidue.from_iupac_lite

cdef object GlycanCompositionProxy = _GlycanCompositionProxy

cdef class SignatureSpecification(object):
    cdef:
        public tuple components
        public tuple masses
        public bint _is_compound
        public Py_hash_t _hash

    def __init__(self, components, masses):
        self.components = tuple(from_iupac_lite(k) for k in components)
        self.masses = tuple(masses)
        self._hash = hash(self.components)
        self._is_compound = len(self.masses) > 1

    def __getitem__(self, i):
        return self.components[i]

    def __iter__(self):
        return iter(self.components)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.components == other.components

    def __repr__(self):
        return "{self.__class__.__name__}({self.components}, {self.masses})".format(self=self)

    cpdef bint is_expected(self, glycan_composition):
        cdef:
            size_t i, n
            bint is_expected
            long count
            _CompositionBase composition

        if isinstance(glycan_composition, GlycanCompositionProxy):
            composition = <_CompositionBase>glycan_composition.obj
        elif isinstance(glycan_composition, _CompositionBase):
            composition = <_CompositionBase>glycan_composition
        else:
            raise TypeError("Requires a _CompositionBase or GlycanCompositionProxy")

        n = PyTuple_GET_SIZE(self.components)
        for i in range(n):
            component = <object>PyTuple_GET_ITEM(self.components, i)
            count = PyInt_AsLong(composition._getitem_fast(component))
            if count == 0:
                return False
        return True

    cpdef int count_of(self, glycan_composition):
        cdef:
            size_t i, n
            bint is_expected
            long count, limit
            _CompositionBase composition

        if isinstance(glycan_composition, GlycanCompositionProxy):
            composition = <_CompositionBase>glycan_composition.obj
        elif isinstance(glycan_composition, _CompositionBase):
            composition = <_CompositionBase>glycan_composition
        else:
            raise TypeError("Requires a _CompositionBase or GlycanCompositionProxy")

        n = PyTuple_GET_SIZE(self.components)
        limit = 100000
        for i in range(n):
            component = <object>PyTuple_GET_ITEM(self.components, i)
            count = PyInt_AsLong(composition._getitem_fast(component))
            if count < limit:
                limit = count
        return limit