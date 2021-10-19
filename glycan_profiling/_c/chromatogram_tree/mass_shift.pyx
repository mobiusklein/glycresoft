cimport cython
from glypy.composition.ccomposition cimport CComposition

from collections import defaultdict


cdef dict mass_shift_index = dict()


cdef class MassShiftBase(object):
    def __eq__(self, other):
        try:
            return (self.name == other.name and abs(
                self.mass - other.mass) < 1e-10) or self.composition == other.composition
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    def _register_name(self):
        mass_shift_index[self.name] = self.composition

    @classmethod
    def get(cls, name):
        return mass_shift_index[name]

    @classmethod
    def _get_name_registry(cls):
        return mass_shift_index


cdef class MassShift(MassShiftBase):
    def __init__(self, name, composition, tandem_composition=None):
        self.name = name
        self.composition = composition
        self.mass = composition.mass
        if tandem_composition is None:
            tandem_composition = self.composition.copy()
        self.tandem_composition = tandem_composition
        self.tandem_mass = tandem_composition.mass
        self._register_name()

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)

    def __mul__(self, n):
        if not isinstance(self, MassShiftBase):
            if isinstance(n, MassShiftBase):
                self, n = n, self
        if self.composition == {}:
            return self
        if isinstance(n, int):
            if n == 0:
                return Unmodified
            return CompoundMassShift({self: n})
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __add__(self, other):
        if self.composition == {}:
            return other
        elif other.composition == {}:
            return self

        if self == other:
            return self * 2
        if isinstance(other, CompoundMassShift):
            return other + self
        else:
            composite = {
                other: 1,
                self: 1
            }
            return CompoundMassShift(composite)

    def __sub__(self, other):
        if other.composition == {}:
            return self
        if self.composition == {}:
            name = "-(%s)" % other.name
            composition = -other.composition
        if self == other:
            return Unmodified
        if isinstance(other, CompoundMassShift):
            return other - self
        if other.composed_with(self):
            return other - self
        else:
            composite = {
                other: -1,
                self: 1
            }
            return CompoundMassShift(composite)

    def composed_with(self, other):
        return self == other


cdef class CompoundMassShift(MassShiftBase):
    def __init__(self, counts=None):
        if counts is None:
            counts = {}
        self.counts = defaultdict(int, counts)
        self.composition = None
        self.tandem_composition = None
        self.name = None

        self._compute_composition()
        self._compute_name()

    def _compute_composition(self):
        composition = CComposition()
        tandem_composition = CComposition()
        for k, v in self.counts.items():
            composition += k.composition * v
            tandem_composition += k.tandem_composition * v
        self.composition = composition
        self.mass = composition.mass
        self.tandem_composition = tandem_composition
        self.tandem_mass = tandem_composition.mass

    def _compute_name(self):
        parts = []
        for k, v in self.counts.items():
            if v == 0:
                continue
            elif v == 1:
                parts.append(k.name)
            else:
                parts.append("%s * %d" % (k.name, v))
        self.name = " + ".join(sorted(parts))

    def composed_with(self, other):
        if isinstance(other, MassShift):
            return self.counts.get(other, 0) >= 1
        elif isinstance(other, CompoundMassShift):
            for key, count in other.counts.items():
                if self.counts.get(key, 0) != count:
                    return False
            return True

    def __add__(self, other):
        if other == Unmodified:
            return self
        elif self == Unmodified:
            return other

        if isinstance(other, MassShift):
            counts = defaultdict(int, self.counts)
            counts[other] += 1
            if counts[other] == 0:
                counts.pop(other)
            if counts:
                return self.__class__(counts)
            return Unmodified
        elif isinstance(other, CompoundMassShift):
            counts = defaultdict(int, self.counts)
            for k, v in other.counts.items():
                if v != 0:
                    counts[k] += v
                if counts[k] == 0:
                    counts.pop(k)
            if counts:
                return self.__class__(counts)
            return Unmodified
        else:
            return NotImplemented

    def __sub__(self, other):
        if other == Unmodified:
            return self
        if not self.composed_with(other):
            raise ValueError("Cannot subtract %r from %r, not part of the compound" % (other, self))

        if isinstance(other, MassShift):
            counts = defaultdict(int, self.counts)
            counts[other] -= 1
            if counts[other] == 0:
                counts.pop(other)
            if counts:
                return self.__class__(counts)
            return Unmodified
        elif isinstance(other, CompoundMassShift):
            counts = defaultdict(int, self.counts)
            for k, v in other.counts.items():
                counts[k] -= v
                if counts[k] == 0:
                    counts.pop(k)
            if counts:
                return self.__class__(counts)
            return Unmodified
        else:
            return NotImplemented

    def __mul__(self, i):
        if not isinstance(self, MassShiftBase):
            if isinstance(i, MassShiftBase):
                self, i = i, self
        if self.composition == {}:
            return self
        if isinstance(i, int):
            counts = defaultdict(int, self.counts)
            for k in counts:
                if k == Unmodified:
                    continue
                counts[k] *= i
            return self.__class__(counts)
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)



cdef MassShiftBase Unmodified = MassShift("Unmodified", CComposition())
