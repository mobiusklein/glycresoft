from array import array
from collections import defaultdict

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenMonosaccharideResidue
from glycan_profiling.chromatogram_tree import mass_shift

from glycan_profiling.chromatogram_tree.mass_shift import Unmodified, Ammonium


class RevisionRule(object):
    def __init__(self, delta_glycan, mass_shift_rule=None, priority=0):
        self.delta_glycan = HashableGlycanComposition.parse(delta_glycan)
        self.mass_shift_rule = mass_shift_rule
        self.priority = priority

    def clone(self):
        return self.__class__(
            self.delta_glycan.clone(), self.mass_shift_rule.clone() if self.mass_shift_rule else None, self.priority)

    def valid(self, record):
        new_record = self(record)
        valid = not any(v < 0 for v in new_record.glycan_composition.values())
        if valid:
            if self.mass_shift_rule:
                return self.mass_shift_rule.valid(record)
        return valid

    def __call__(self, record):
        result = record.shift_glycan_composition(self.delta_glycan)
        if self.mass_shift_rule is not None:
            return self.mass_shift_rule(result)
        return result

    def revert(self, record):
        return record.shift_glycan_composition(-self.delta_glycan)

    def invert_rule(self):
        return self.__class__(
            -self.delta_glycan, self.mass_shift_rule.invert() if self.mass_shift_rule else None, self.priority)

    def __eq__(self, other):
        try:
            return self.delta_glycan == other.delta_glycan
        except AttributeError:
            return self.delta_glycan == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.delta_glycan)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.delta_glycan})"
        return template.format(self=self)


class ValidatingRevisionRule(RevisionRule):
    def __init__(self, delta_glycan, validator, mass_shift_rule=None, priority=0):
        super(ValidatingRevisionRule, self).__init__(
            delta_glycan, mass_shift_rule=mass_shift_rule, priority=priority)
        self.validator = validator

    def clone(self):
        return self.__class__(
            self.delta_glycan.clone(), self.validator, self.mass_shift_rule.clone() if self.mass_shift_rule else None, self.priority)

    def valid(self, record):
        if super(ValidatingRevisionRule, self).valid(record):
            return self.validator(record)
        return False


class MassShiftRule(object):
    def __init__(self, mass_shift, multiplicity):
        self.mass_shift = mass_shift
        self.sign = int(multiplicity / abs(multiplicity))
        self.multiplicity = abs(multiplicity)
        self.single = abs(multiplicity) == 1

    def invert_rule(self):
        return self.__class__(self.mass_shift, -self.sign * self.multiplicity)

    def clone(self):
        return self.__class__(self.mass_shift, self.multiplicity * self.sign)

    def valid(self, record):
        assert isinstance(self.mass_shift, mass_shift.MassShift)
        if self.sign < 0:
            # Can only lose a mass shift if all the mass shifts of the analyte are able to lose self.mass_shift without going
            # negative
            for shift in record.mass_shifts:
                # The current mass shift is a compound mass shift, potentially having multiple copies of self.mass_shift
                if isinstance(shift, mass_shift.CompoundMassShift) and shift.counts.get(self.mass_shift, 0) < self.multiplicity:
                    return False
                # The current mass shift is a single simple mass shift and it isn't a match for self.mass_shift or self.multiplicity > 1
                elif isinstance(shift, mass_shift.MassShift) and (shift != self.mass_shift) or (shift == self.mass_shift and not self.single):
                    return False
            return True
        else:
            # Can always gain a mass shift
            return True

    def apply(self, record):
        new = record.copy()
        new.mass_shifts = [
            m + (self.mass_shift * self.sign * self.multiplicity) for m in new.mass_shifts]
        return new

    def __call__(self, record):
        return self.apply(record)

    def __eq__(self, other):
        if other is None:
            return False
        return self.mass_shift == other.mass_shift and self.multiplicity == other.multiplicity

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.mass_shift)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.mass_shift}, {self.multiplicity})"
        return template.format(self=self)


def modify_rules(rules, symbol_map):
    rules = list(rules)
    for sym_from, sym_to in symbol_map.items():
        editted = []
        for rule in rules:
            if sym_from in rule.delta_glycan:
                rule = rule.clone()
                count = rule.delta_glycan.pop(sym_from)
                rule.delta_glycan[sym_to] = count
            editted.append(rule)
        rules = editted
    return rules


class PeptideYUtilizationPreservingRevisionValidator(object):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def validate(self, revised, original):
        source = revised.source
        if source is None:
            # Can't validate without a source to read the spectrum match metadata
            return True

        try:
            revised_gpsm = source.best_match_for(revised.structure)
        except KeyError:
            # Can't find a spectrum match to the revised form, assume we're allowed to
            # revise.
            return True

        try:
            original_gpsm = source.best_match_for(original.structure)
        except KeyError:
            # Can't find a spectrum match to the original, assume we're allowed to
            # revise.
            return True
        original_utilization = original_gpsm.score_set.stub_glycopeptide_intensity_utilization
        if not original_utilization:
            # Anything is better than or equal to zero
            return True
        utilization_ratio = revised_gpsm.score_set.stub_glycopeptide_intensity_utilization / original_utilization
        return utilization_ratio > self.threshold

    def __call__(self, revised, original):
        return self.validate(revised, original)



class ModelReviser(object):
    def __init__(self, model, rules, chromatograms=None, valid_glycans=None, revision_validator=None):
        if chromatograms is None:
            chromatograms = model.chromatograms
        self.model = model
        self.chromatograms = chromatograms
        self.original_scores = array('d')
        self.original_times = array('d')
        self.rules = list(rules)
        self.alternative_records = defaultdict(list)
        self.alternative_scores = defaultdict(lambda: array('d'))
        self.alternative_times = defaultdict(lambda: array('d'))
        self.valid_glycans = valid_glycans
        self.revision_validator = revision_validator

    def rescore(self, case):
        return self.model.score(case)

    def modify_rules(self, symbol_map):
        self.rules = modify_rules(self.rules, symbol_map)

    def propose_revisions(self, case):
        propositions = {
            None: (case, self.rescore(case)),
        }
        for rule in self.rules:
            if rule.valid(case):
                rec = rule(case)
                if self.valid_glycans and rec.structure.glycan_composition not in self.valid_glycans:
                    continue
                propositions[rule] = (rec, self.rescore(rec))
        return propositions

    def process_rule(self, rule):
        alts = []
        alt_scores = array('d')
        alt_times = array('d')
        for case in self.chromatograms:
            if rule.valid(case):
                rec = rule(case)
                if self.valid_glycans and rec.glycan_composition not in self.valid_glycans:
                    alts.append(None)
                    alt_scores.append(0.0)
                    alt_times.append(0.0)
                    continue
                alts.append(rec)
                alt_times.append(self.model.predict(rec))
                alt_scores.append(self.rescore(rec))
            else:
                alts.append(None)
                alt_scores.append(0.0)
                alt_times.append(0.0)
        self.alternative_records[rule] = alts
        self.alternative_scores[rule] = alt_scores
        self.alternative_times[rule] = alt_times

    def process_model(self):
        scores = array('d')
        times = array('d')
        for case in self.chromatograms:
            scores.append(self.rescore(case))
            times.append(self.model.predict(case))
        self.original_scores = scores
        self.original_times = times

    def evaluate(self):
        self.process_model()
        for rule in self.rules:
            self.process_rule(rule)

    def revise(self, threshold=0.2, delta_threshold=0.2, minimum_time_difference=0.5):
        chromatograms = self.chromatograms
        original_scores = self.original_scores
        next_round = []
        for i in range(len(chromatograms)):
            best_score = original_scores[i]
            best_rule = None
            delta_best_score = float('inf')
            original_solution = best_record = chromatograms[i]

            for rule in self.rules:
                a = self.alternative_scores[rule][i]
                t = self.alternative_times[rule][i]
                if a > best_score and not np.isclose(a, 0.0) and abs(t - self.original_times[i]) > minimum_time_difference:
                    if self.revision_validator and not self.revision_validator(self.alternative_records[rule][i], original_solution):
                        continue
                    delta_best_score = a - original_scores[i]
                    best_score = a
                    best_rule = rule
                    best_record = self.alternative_records[rule][i]

            if best_score > threshold and delta_best_score > delta_threshold:
                if best_rule is not None:
                    best_record.revised_from = (best_rule, chromatograms[i])
                next_round.append(best_record)
            else:
                next_round.append(chromatograms[i])
        return next_round


dhex = FrozenMonosaccharideResidue.from_iupac_lite("d-Hex")
fuc = FrozenMonosaccharideResidue.from_iupac_lite("Fuc")
neuac = FrozenMonosaccharideResidue.from_iupac_lite("Neu5Ac")
hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")

AmmoniumMaskedRule = RevisionRule(
    HashableGlycanComposition(Hex=-1, Fuc=-1, Neu5Ac=1), mass_shift_rule=MassShiftRule(Ammonium, 1), priority=1)
AmmoniumUnmaskedRule = RevisionRule(
    HashableGlycanComposition(Hex=1, Fuc=1, Neu5Ac=-1), mass_shift_rule=MassShiftRule(Ammonium, -1), priority=1)

IsotopeRule = RevisionRule(HashableGlycanComposition(Fuc=-2, Neu5Ac=1))
IsotopeRule2 = RevisionRule(HashableGlycanComposition(Fuc=-4, Neu5Ac=2))

HexNAc2NeuAc2ToHex6AmmoniumRule = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-6, HexNAc=2, Neu5Ac=2),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and x.glycan_composition[hexnac] == 2)

HexNAc2Fuc1NeuAc2ToHex7 = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-7, HexNAc=2, Neu5Ac=2, Fuc=1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and x.glycan_composition[hexnac] == 2)

NeuAc1Hex1ToNeuGc1Fuc1Rule = RevisionRule(
    HashableGlycanComposition(Neu5Ac=1, Hex=1, Neu5Gc=-1, Fuc=-1))
NeuGc1Fuc1ToNeuAc1Hex1Rule = RevisionRule(
    HashableGlycanComposition(Neu5Ac=-1, Hex=-1, Neu5Gc=1, Fuc=1))

Sulfate1HexNAc2ToHex3Rule = RevisionRule(
    HashableGlycanComposition(HexNAc=2, sulfate=1, Hex=-3))

Hex3ToSulfate1HexNAc2Rule = Sulfate1HexNAc2ToHex3Rule.invert_rule()

SulfateToPhosphateRule = RevisionRule(HashableGlycanComposition(sulfate=-1, phosphate=1))
PhosphateToSulfateRule = SulfateToPhosphateRule.invert_rule()


class IntervalModelReviser(ModelReviser):
    alpha = 0.01

    def rescore(self, case):
        return self.model.score_interval(case, alpha=0.01)


# In reviser, combine the score from the absolute coordinate with the
# score relative to each other glycoform in the group? This might look
# something like the old recalibrator code, but adapted for the interval_score
# method
