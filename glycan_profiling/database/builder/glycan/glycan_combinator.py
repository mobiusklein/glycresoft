import itertools
from collections import Counter

from glypy.composition.glycan_composition import FrozenGlycanComposition
from glypy.composition import formula, Composition

from glycan_profiling.serialize.hypothesis.glycan import (
    GlycanComposition, GlycanCombination, GlycanCombinationGlycanComposition)

from glycan_profiling.serialize import DatabaseBoundOperation
from glycan_profiling.task import TaskBase


class GlycanCompositionRecord(object):
    def __init__(self, db_glycan_composition):
        self.id = db_glycan_composition.id
        self.composition = db_glycan_composition.convert()
        self._str = str(self.composition)
        self._hash = hash(self._str)
        self.mass = self.composition.mass()
        self.elemental_composition = self.composition.total_composition()

    def __repr__(self):
        return "GlycanCompositionRecord(%d, %s)" % (self.id, self.composition)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._str == other._str


def merge_compositions_frozen(composition_list):
    """Given a list of monosaccharide packed strings,
    sum the monosaccharide counts across lists and return
    the merged string

    Parameters
    ----------
    composition_list : list of GlycanCompositionRecord

    Returns
    -------
    FrozenGlycanComposition
    """
    first = FrozenGlycanComposition()
    for comp_rec in composition_list:
        first += comp_rec.composition
    return first


class GlycanCombinationBuilder(object):
    def __init__(self, glycan_compositions, max_size=1):
        self.glycan_compositions = list(map(GlycanCompositionRecord, glycan_compositions))
        self.max_size = max_size

    def combinate(self, n=1):
        j = 0
        for comb_compositions in itertools.combinations_with_replacement(self.glycan_compositions, n):
            j += 1
            counts = Counter(comb_compositions)
            merged = merge_compositions_frozen(comb_compositions)
            composition = str(merged)
            mass = sum(c.mass for c in comb_compositions)
            elemental_composition = Composition()
            for c in comb_compositions:
                elemental_composition += c.elemental_composition
            inst = GlycanCombination(
                count=n,
                calculated_mass=mass,
                composition=composition,
                formula=formula(elemental_composition))
            yield inst, counts

    def combinate_all(self):
        for i in range(1, self.max_size + 1):
            for data in self.combinate(i):
                yield data


class GlycanCombinationSerializer(DatabaseBoundOperation, TaskBase):
    def __init__(self, connection, source_hypothesis_id, target_hypothesis_id, max_size=1):
        DatabaseBoundOperation.__init__(self, connection)
        self.source_hypothesis_id = source_hypothesis_id
        self.target_hypothesis_id = target_hypothesis_id
        self.max_size = max_size
        self.total_count = 0

    def load_compositions(self):
        return self.query(GlycanComposition).filter(
            GlycanComposition.hypothesis_id == self.source_hypothesis_id).all()

    def generate(self):
        compositions = self.load_compositions()
        combinator = GlycanCombinationBuilder(compositions, self.max_size)

        relation_acc = []
        i = 0
        j = 0
        hypothesis_id = self.target_hypothesis_id
        self.log("... Building combinations for Hypothesis %d" % hypothesis_id)
        for comb, counts in combinator.combinate_all():
            comb.hypothesis_id = hypothesis_id
            self.session.add(comb)
            self.session.flush()

            for key, count in counts.items():
                relation_acc.append({
                    "glycan_id": key.id,
                    "count": count,
                    "combination_id": comb.id
                })

            i += 1
            j += 1
            self.total_count += 1
            if i > 50000:
                self.log("%d combinations created" % j)
                self.session.execute(
                    GlycanCombinationGlycanComposition.insert(), relation_acc)
                i = 0
                relation_acc = []
        if relation_acc:
            self.session.execute(
                GlycanCombinationGlycanComposition.insert(), relation_acc)
        self.session.commit()

    def run(self):
        self.generate()
        total = self.session.query(
            GlycanCombination).filter(
            GlycanCombination.hypothesis_id == self.target_hypothesis_id
        ).count()
        self.log("%d Glycan Combinations Constructed." % (total,))
