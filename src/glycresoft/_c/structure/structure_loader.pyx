cimport cython
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM, PyTuple_SET_ITEM, PyTuple_New
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM, PyList_SET_ITEM, PyList_New

from numpy cimport npy_uint32 as uint32_t, npy_uint64 as uint64_t, npy_float64 as float64_t
cimport numpy as np

np.import_array()

from glypy.utils.cenum cimport EnumMeta, EnumValue
from glycopeptidepy._c.structure.fragment cimport StubFragment


@cython.freelist(1000000)
cdef class GlycopeptideDatabaseRecord(object):
    def __init__(self, id, calculated_mass, glycopeptide_sequence, protein_id,
                 start_position, end_position, peptide_mass, hypothesis_id):
        self.id = id
        self.calculated_mass = calculated_mass
        self.glycopeptide_sequence = glycopeptide_sequence
        self.protein_id = protein_id
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_mass = peptide_mass
        self.hypothesis_id = hypothesis_id

    def __repr__(self):
        template = (
            "{self.__class__.__name__}(id={self.id}, calculated_mass={self.calculated_mass}, "
            "glycopeptide_sequence={self.glycopeptide_sequence}, protein_id={self.protein_id}, "
            "start_position={self.start_position}, end_position={self.end_position}, "
            "peptide_mass={self.peptide_mass}, hypothesis_id={self.hypothesis_id}, ")
        return template.format(self=self)

    def __reduce__(self):
        return self.__class__, (self.id, self.calculated_mass, self.glycopeptide_sequence, self.protein_id,
                                self.start_position, self.end_position, self.peptide_mass, self.hypothesis_id)


@cython.freelist(10000000)
cdef class glycopeptide_key(object):
    def __init__(self, start_position, end_position, peptide_id, protein_id, hypothesis_id,
                 glycan_combination_id, structure_type, site_combination_index):
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_id = peptide_id
        self.protein_id = protein_id
        self.hypothesis_id = hypothesis_id
        self.glycan_combination_id = glycan_combination_id
        self.structure_type = structure_type
        self.site_combination_index = site_combination_index
        self._hash = hash((
            start_position, end_position, peptide_id, protein_id, hypothesis_id,
            glycan_combination_id, structure_type, site_combination_index))

    @staticmethod
    cdef glycopeptide_key _create(uint32_t start_position, uint32_t end_position, uint64_t peptide_id,
                                  uint32_t protein_id, uint32_t hypothesis_id, uint64_t glycan_combination_id,
                                  object structure_type, uint64_t site_combination_index, Py_hash_t _hash=0):
        cdef glycopeptide_key self = glycopeptide_key.__new__(glycopeptide_key)
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_id = peptide_id
        self.protein_id = protein_id
        self.hypothesis_id = hypothesis_id
        self.glycan_combination_id = glycan_combination_id
        self.structure_type = structure_type
        self.site_combination_index = site_combination_index
        if _hash == 0:
            self._hash = hash((
                start_position, end_position, peptide_id, protein_id, hypothesis_id,
                glycan_combination_id, structure_type, site_combination_index))
        else:
            self._hash = _hash
        return self

    cdef void _rehash(self):
        self._hash = hash((
            self.start_position, self.end_position, self.peptide_id, self.protein_id,
            self.hypothesis_id, self.glycan_combination_id, self.structure_type,
            self.site_combination_index))

    cpdef glycopeptide_key copy(self, object structure_type=None):
        cdef:
            Py_hash_t _hash
        if structure_type is None:
            structure_type = self.structure_type
        return self.__class__(
            self.start_position, self.end_position, self.peptide_id, self.protein_id,
            self.hypothesis_id, self.glycan_combination_id, structure_type,
            self.site_combination_index)

    def _replace(self, **kwargs):
        cdef glycopeptide_key dup = self.copy()
        for key, val in kwargs.items():
            setattr(dup, key, val)
        dup._rehash()
        return dup

    def __iter__(self):
        yield self.start_position
        yield self.end_position
        yield self.peptide_id
        yield self.protein_id
        yield self.hypothesis_id
        yield self.glycan_combination_id
        yield self.structure_type
        yield self.site_combination_index

    def __reduce__(self):
        return self.__class__, tuple(self)

    def __getitem__(self, i):
        if i == 0:
            return self.start_position
        elif i == 1:
            return self.end_position
        elif i == 2:
            return self.peptide_id
        elif i == 3:
            return self.protein_id
        elif i == 4:
            return self.hypothesis_id
        elif i == 5:
            return self.glycan_combination_id
        elif i == 6:
            return self.structure_type
        elif i == 7:
            return self.site_combination_index
        else:
            raise IndexError(i)

    cpdef dict as_dict(self, bint stringify=False):
        cdef:
            dict d

        d = {}
        if stringify:
            d['start_position'] = str(self.start_position)
            d['end_position'] = str(self.end_position)
            d['peptide_id'] = str(self.peptide_id)
            d['protein_id'] = str(self.protein_id)
            d['hypothesis_id'] = str(self.hypothesis_id)
            d['glycan_combination_id'] = str(self.glycan_combination_id)
            d['structure_type'] = str(self.structure_type)
            d['site_combination_index'] = str(self.site_combination_index)
        else:
            d['start_position'] = self.start_position
            d['end_position'] = self.end_position
            d['peptide_id'] = self.peptide_id
            d['protein_id'] = self.protein_id
            d['hypothesis_id'] = self.hypothesis_id
            d['glycan_combination_id'] = self.glycan_combination_id
            d['structure_type'] = self.structure_type
            d['site_combination_index'] = self.site_combination_index
        return d


@cython.binding(True)
cpdef tuple peptide_backbone_fragment_key(self, target, args, dict kwargs):
    key = ("get_fragments", args, frozenset(kwargs.items()))
    return key


cdef class PeptideDatabaseRecordBase(object):

    def __hash__(self):
        return hash(self.modified_peptide_sequence)

    def __eq__(self, PeptideDatabaseRecordBase other):
        if other is None:
            return False
        if self.id != other.id:
            return False
        if self.protein_id != other.protein_id:
            return False
        if abs(self.calculated_mass - other.calculated_mass) > 1e-3:
            return False
        if self.start_position != other.start_position:
            return False
        if self.end_position != other.end_position:
            return False
        if self.hypothesis_id != other.hypothesis_id:
            return False
        if self.n_glycosylation_sites != other.n_glycosylation_sites:
            return False
        if self.o_glycosylation_sites != other.o_glycosylation_sites:
            return False
        if self.gagylation_sites != other.gagylation_sites:
            return False
        return True

    def __ne__(self, other):
        return not (self == other)

    cpdef bint has_glycosylation_sites(self):
        if len(self.n_glycosylation_sites) > 0:
            return True
        elif len(self.o_glycosylation_sites) > 0:
            return True
        elif len(self.gagylation_sites) > 0:
            return True
        return False

    @classmethod
    def from_record(cls, record):
        return cls(**record)


cdef class GlycopeptideFragmentCachingContext(object):
    def __init__(self, store=None):
        if store is None:
            store = {}
        self.store = store

    cpdef peptide_backbone_fragment_key(self, target, args, dict kwargs):
        key = ("get_fragments", args, frozenset(kwargs.items()))
        return key

    cpdef stub_fragment_key(self, target, args, dict kwargs):
        key = ('stub_fragments', args, frozenset(kwargs.items()), )
        return key

    def __contains__(self, key):
        return PyDict_GetItem(self.store, key) != NULL

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        PyDict_SetItem(self.store, key, value)

    def __reduce__(self):
        return self.__class__, (self.store, )

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def clear(self):
        self.store.clear()

    cpdef bind(self, target):
        target.fragment_caches = self
        return target

    cpdef unbind(self, target):
        target.fragment_caches = self.__class__()
        return target

    def __call__(self, target):
        return self.bind(target)

    cpdef _make_target_key(self, tuple key):
        return None


from glycresoft.structure.enums import SpectrumMatchClassification as _StructureClassification

cdef EnumMeta StructureClassification = _StructureClassification


cdef class GlycanAwareGlycopeptideFragmentCachingContext(GlycopeptideFragmentCachingContext):
    cpdef stub_fragment_key(self, target, args, dict kwargs):
        cdef tid = target.id
        key = ('stub_fragments', args, frozenset(
            kwargs.items()), tid.glycan_combination_id, tid.structure_type)
        return key

    cpdef _make_target_key(self, tuple key):
        cdef:
            size_t n, i
            tuple new_key
            PyObject* tmp
            object as_target_peptide
            EnumValue value
        n = PyTuple_GET_SIZE(key)
        value = <EnumValue>PyTuple_GET_ITEM(key, n - 1)
        new_key = PyTuple_New(n)
        for i in range(n - 1):
            tmp = PyTuple_GET_ITEM(key, i)
            Py_INCREF(<object>tmp)
            PyTuple_SET_ITEM(new_key, i, <object>tmp)
        as_target_peptide = StructureClassification[value.value ^ 1]
        Py_INCREF(as_target_peptide)
        PyTuple_SET_ITEM(new_key, n - 1, as_target_peptide)
        return new_key


cpdef list clone_stub_fragments(list stubs):
    cdef:
        size_t i, n
        list result
        StubFragment frag

    n = PyList_GET_SIZE(stubs)
    result = PyList_New(n)
    for i in range(n):
        frag = <StubFragment>PyList_GET_ITEM(stubs, i)
        frag = <StubFragment>StubFragment.clone(frag)
        Py_INCREF(frag)
        PyList_SET_ITEM(result, i, frag)
    return result


@cython.boundscheck(False)
cpdef list clone_and_shift_stub_fragments(list stubs, np.ndarray[float64_t, ndim=1, mode='c'] rand_deltas, bint do_clone=True, int min_shift_size=1):
    cdef:
        size_t i, n, j
        list result
        StubFragment frag
        float64_t delta

    n = PyList_GET_SIZE(stubs)
    if do_clone:
        result = PyList_New(n)
    j = 0
    for i in range(n):
        frag = <StubFragment>PyList_GET_ITEM(stubs, i)
        if do_clone:
            frag = <StubFragment>StubFragment.clone(frag)
        if min_shift_size == 0 or StubFragment.get_glycosylation_size(frag) > min_shift_size:
            delta = rand_deltas[j]
            j += 1
            frag.mass += delta
        if do_clone:
            Py_INCREF(frag)
            PyList_SET_ITEM(result, i, frag)
    if do_clone:
        return result
    return stubs