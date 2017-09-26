from collections import defaultdict
from glypy import Composition

mass_shift_index = dict()


class MassShiftBase(object):
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


class MassShift(MassShiftBase):
    def __init__(self, name, composition):
        self.name = name
        self.composition = composition
        self.mass = composition.mass
        self._register_name()

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)

    def __mul__(self, n):
        if self.composition == {}:
            return self
        if isinstance(n, int):
            return CompoundMassShift({self: n})
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __add__(self, other):
        if self.composition == {}:
            return other
        elif other.composition == {}:
            return self
        name = "(%s) + (%s)" % (self.name, other.name)
        composition = self.composition + other.composition
        return self.__class__(name, composition)

    def composed_with(self, other):
        return self == other


class CompoundMassShift(MassShiftBase):
    def __init__(self, counts=None):
        if counts is None:
            counts = {}
        self.counts = defaultdict(int, counts)
        self.composition = None
        self.name = None
        self.mass = None

        self._compute_composition()
        self._compute_name()

    def _compute_composition(self):
        composition = Composition()
        for k, v in self.counts.items():
            composition += k.composition * v
        self.composition = composition
        self.mass = composition.mass

    def _compute_name(self):
        parts = []
        for k, v in self.counts.items():
            if v == 1:
                parts.append(k.name)
            else:
                parts.append("%s * %d" % (k.name, v))
        self.name = " + ".join(sorted(parts))

    def composed_with(self, other):
        if isinstance(other, MassShift):
            return self.counts.get(other, 0) == 1
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
            return self.__class__(counts)
        elif isinstance(other, CompoundMassShift):
            counts = defaultdict(int, self.counts)
            for k, v in other.counts.items():
                counts[k] += v
            return self.__class__(counts)
        else:
            return NotImplemented

    def __mul__(self, i):
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

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)


Unmodified = MassShift("Unmodified", Composition())
Formate = MassShift("Formate", Composition('HCOOH'))
Ammonium = MassShift("Ammonium", Composition("NH3"))
Sodium = MassShift("Sodium", Composition("Na"))
Potassium = MassShift("Potassium", Composition("K"))
