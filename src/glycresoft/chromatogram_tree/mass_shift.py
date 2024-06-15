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

try:
    from glycresoft._c.chromatogram_tree.mass_shift import MassShiftBase
    mass_shift_index = MassShiftBase._get_name_registry()
except ImportError:
    pass


class MassShift(MassShiftBase):
    def __init__(self, name, composition, tandem_composition=None, charge_carrier=0):
        self.name = name
        self.composition = composition
        self.mass = composition.mass
        self.charge_carrier = charge_carrier
        if tandem_composition is None:
            tandem_composition = self.composition.copy()
        self.tandem_composition = tandem_composition
        self.tandem_mass = self.tandem_composition.mass
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

    def __rmul__(self, n):
        if self.composition == {}:
            return self
        if isinstance(n, int):
            return CompoundMassShift({self: n})
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __getstate__(self):
        return {
            "tandem_composition": self.tandem_composition,
            "name": self.name,
            "mass": self.mass,
            "charge_carrier": self.charge_carrier,
            "tandem_mass": self.tandem_mass,
            "composition": self.composition
        }

    def __setstate__(self, state):
        if isinstance(state, tuple):
            # We're receiving an older Cython pickled state
            self._hash = state[0]
            self.composition = state[1]
            self.mass = state[2]
            self.name = state[3]
            self.tandem_composition = state[4]
            self.tandem_mass = state[5]
            self.charge_carrier = state[6].get('charge_carrier', 0)

        else:
            self.name = state['name']
            self.charge_carrier = state.get('charge_carrier', 0)
            self.mass = state['mass']
            self.composition = state['composition']
            self.tandem_mass = state['tandem_mass']
            self.tandem_composition = state['tandem_composition']
            self._hash = hash(self.name)

    def __add__(self, other):
        if self.composition == {}:
            return other
        elif other.composition == {}:
            return self

        if self == other:
            return self * 2
        if isinstance(other, CompoundMassShift):
            return other + self
        return CompoundMassShift({
            self: 1,
            other: 1
        })

    def __sub__(self, other):
        if other.composition == {}:
            return self
        # if self.composition == {}:
        #     name = "-(%s)" % other.name
        #     composition = -other.composition
        #     tandem_composition = -other.tandem_composition
        #     charge_carrier = -other.charge_carrier
        if self == other:
            return Unmodified
        if isinstance(other, CompoundMassShift):
            return other - self
        else:
            return CompoundMassShift({
                self: 1,
                other: -1
            })

    def composed_with(self, other):
        return self == other


class CompoundMassShift(MassShiftBase):
    def __init__(self, counts=None):
        if counts is None:
            counts = {}
        self.counts = defaultdict(int, counts)
        self.composition = None
        self.tandem_composition = None
        self.name = None
        self.mass = 0
        self.tandem_mass = 0
        self.charge_carrier = 0
        self._compute_composition()
        self._compute_name()

    def __reduce__(self):
        return self.__class__, (self.counts, )

    def _compute_composition(self):
        composition = Composition()
        tandem_composition = Composition()
        charge_carrier = 0
        for k, v in self.counts.items():
            composition += k.composition * v
            tandem_composition += k.tandem_composition * v
            charge_carrier += k.charge_carrier * v
        self.composition = composition
        self.mass = composition.mass
        self.tandem_composition = tandem_composition
        self.tandem_mass = tandem_composition.mass
        self.charge_carrier = charge_carrier

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
            raise ValueError(
                "Cannot subtract %r from %r, not part of the compound" % (other, self))

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


Unmodified = MassShift("Unmodified", Composition())
Formate = MassShift("Formate", Composition('HCOOH'), charge_carrier=1)
Ammonium = MassShift("Ammonium", Composition("NH3"), Composition())
Sodium = MassShift("Sodium", Composition("Na1H-1"), charge_carrier=1)
Potassium = MassShift("Potassium", Composition("K1H-1"), charge_carrier=1)
Deoxy = MassShift("Deoxy", Composition("O-1"), Composition())


class MassShiftCollection(object):
    def __init__(self, mass_shifts):
        self.mass_shifts = list(mass_shifts)
        self.mass_shift_map = {}
        self._invalidate()

    def _invalidate(self):
        self.mass_shift_map = {
            mass_shift.name: mass_shift for mass_shift in self.mass_shifts
        }

    def append(self, mass_shift):
        self.mass_shifts.append(mass_shift)
        self._invalidate()

    def __getitem__(self, i):
        try:
            return self.mass_shifts[i]
        except (IndexError, TypeError):
            return self.mass_shift_map[i]

    def __iter__(self):
        return iter(self.mass_shifts)

    def __len__(self):
        return len(self.mass_shifts)