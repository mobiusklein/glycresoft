import logging

from array import array
from collections import defaultdict

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenMonosaccharideResidue
from glycan_profiling.chromatogram_tree import mass_shift

from glycan_profiling.chromatogram_tree.mass_shift import Unmodified, Ammonium, Deoxy

from glycan_profiling.task import LoggingMixin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RevisionRule(object):
    def __init__(self, delta_glycan, mass_shift_rule=None, priority=0, name=None):
        self.delta_glycan = HashableGlycanComposition.parse(delta_glycan)
        self.mass_shift_rule = mass_shift_rule
        self.priority = priority
        self.name = name

    def clone(self):
        return self.__class__(
            self.delta_glycan.clone(), self.mass_shift_rule.clone() if self.mass_shift_rule else None, self.priority, self.name)

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
            return self.delta_glycan == other.delta_glycan and self.mass_shift_rule == other.mass_shift_rule
        except AttributeError:
            return self.delta_glycan == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.delta_glycan)

    def __repr__(self):
        if self.mass_shift_rule:
            template = "{self.__class__.__name__}({self.delta_glycan}, {self.mass_shift_rule}, {self.name})"
        else:
            template = "{self.__class__.__name__}({self.delta_glycan}, {self.name})"
        return template.format(self=self)


class ValidatingRevisionRule(RevisionRule):
    def __init__(self, delta_glycan, validator, mass_shift_rule=None, priority=0, name=None):
        super(ValidatingRevisionRule, self).__init__(
            delta_glycan, mass_shift_rule=mass_shift_rule, priority=priority, name=name)
        self.validator = validator

    def clone(self):
        return self.__class__(
            self.delta_glycan.clone(), self.validator, self.mass_shift_rule.clone() if self.mass_shift_rule else None, self.priority, self.name)

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


def always(x):
    return True


def get_mass_error(record):
    observed_mass = record.weighted_neutral_mass
    theoretical_mass = record.structure.total_mass


class RevisionValidatorBase(LoggingMixin):
    def __init__(self, spectrum_match_builder, threshold_fn=always):
        self.spectrum_match_builder = spectrum_match_builder
        self.threshold_fn = threshold_fn

    def find_revision_gpsm(self, chromatogram, revised):
        found_revision = False
        revised_gpsm = None
        try:
            revised_gpsm = chromatogram.best_match_for(revised.structure)
            found_revision = True
        except KeyError:
            if self.spectrum_match_builder is not None:
                self.spectrum_match_builder.get_spectrum_solution_sets(
                    revised, chromatogram)
                try:
                    revised_gpsm = chromatogram.best_match_for(revised.structure)
                    found_revision = True
                except KeyError:
                    pass
        return revised_gpsm, found_revision

    def find_gpsms(self, source, revised, original):
        original_gpsm = None
        revised_gpsm = None

        if source is None:
            # Can't validate without a source to read the spectrum match metadata
            return original_gpsm, revised_gpsm, True

        revised_gpsm, found_revision = self.find_revision_gpsm(source, revised)

        if not found_revision:
            # Can't find a spectrum match to the revised form, assume we're allowed to
            # revise.
            self.debug(
                "...... Permitting revision for %s (%0.3f) from %s to %s because revision not evaluated" %
                (revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition))
            return original_gpsm, revised_gpsm, True

        try:
            original_gpsm = source.best_match_for(original.structure)
        except KeyError:
            # Can't find a spectrum match to the original, assume we're allowed to
            # revise.
            self.debug(
                "...... Permitting revision for %s (%0.3f) from %s to %s because original not evaluated" %
                (revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition))
            return original_gpsm, revised_gpsm, True

        return original_gpsm, revised_gpsm, False

    def validate(self, revised, original):
        raise NotImplementedError()

    def __call__(self, revised, original):
        return self.validate(revised, original)


class PeptideYUtilizationPreservingRevisionValidator(RevisionValidatorBase):
    def __init__(self, threshold=0.85, spectrum_match_builder=None, threshold_fn=always):
        super(PeptideYUtilizationPreservingRevisionValidator, self).__init__(
            spectrum_match_builder, threshold_fn)
        self.threshold = threshold

    def validate(self, revised, original):
        source = revised.source
        if source is None:
            # Can't validate without a source to read the spectrum match metadata
            return True

        original_gpsm, revised_gpsm, skip = self.find_gpsms(
            source, revised, original)
        if skip:
            return True

        original_utilization = original_gpsm.score_set.stub_glycopeptide_intensity_utilization
        if not original_utilization:
            # Anything is better than or equal to zero
            return True

        revised_utilization = revised_gpsm.score_set.stub_glycopeptide_intensity_utilization
        utilization_ratio = revised_utilization / original_utilization
        valid = utilization_ratio > self.threshold

        self.debug(
            "...... Checking revision by peptide+Y ions for %s (%0.3f) from %s to %s: %0.3f / %0.3f: %r" %
            (revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition,
             revised_utilization, original_utilization, valid))
        return valid


class OxoniumIonRequiringUtilizationRevisionValidator(RevisionValidatorBase):
    def __init__(self, scale=0.9, spectrum_match_builder=None, threshold_fn=always):
        super(OxoniumIonRequiringUtilizationRevisionValidator,
              self).__init__(spectrum_match_builder, threshold_fn)
        self.scale = scale

    def validate(self, revised, original):
        source = revised.source
        if source is None:
            # Can't validate without a source to read the spectrum match metadata
            return True

        original_gpsm, revised_gpsm, skip = self.find_gpsms(
            source, revised, original)
        if skip:
            return True

        original_utilization = original_gpsm.score_set.oxonium_ion_intensity_utilization
        revised_utilization = revised_gpsm.score_set.oxonium_ion_intensity_utilization

        threshold = original_utilization * self.scale
        valid = (revised_utilization - threshold) >= 0
        self.debug(
            "...... Checking revision by oxonium ions for %s (%0.3f) from %s to %s: %0.3f / %0.3f: %r" %
            (revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition,
            revised_utilization, original_utilization, valid))
        return valid


class CompoundRevisionValidator(object):
    def __init__(self, validators=None):
        if validators is None:
            validators = []
        self.validators = list(validators)

    def validate(self, revised, original):
        for rule in self.validators:
            if not rule(revised, original):
                return False
        return True

    def __call__(self, revised, original):
        return self.validate(revised, original)



def _new_array():
    return array('d')


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
        self.alternative_scores = defaultdict(_new_array)
        self.alternative_times = defaultdict(_new_array)
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
        original_times = self.original_times
        next_round = []
        for i in range(len(chromatograms)):
            original_score = best_score = original_scores[i]
            best_rule = None
            delta_best_score = float('inf')
            original_solution = best_record = chromatograms[i]
            original_time = original_times[i]

            for rule in self.rules:
                alt_rec = self.alternative_records[rule][i]
                alt_score = self.alternative_scores[rule][i]
                alt_time = self.alternative_times[rule][i]
                if alt_score > best_score and not np.isclose(alt_score, 0.0) and abs(alt_time - original_time) > minimum_time_difference:
                    if self.revision_validator and not self.revision_validator(alt_rec, original_solution):
                        continue
                    delta_best_score = alt_score - original_score
                    best_score = alt_score
                    best_rule = rule
                    best_record = alt_rec

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

# The correct glycan was incorrectly assigned as the unadducted form of
# another glycan.
#
# Gain an ammonium adduct and a NeuAc, lose a Hex and Fuc
AmmoniumMaskedRule = RevisionRule(
    HashableGlycanComposition(Hex=-1, Fuc=-1, Neu5Ac=1),
    mass_shift_rule=MassShiftRule(Ammonium, 1), priority=1, name="Ammonium Masked")

AmmoniumMaskedNeuGcRule = RevisionRule(
    HashableGlycanComposition(Hex=-2, Neu5Gc=1),
    mass_shift_rule=MassShiftRule(Ammonium, 1), priority=1, name="Ammonium Masked NeuGc")

# The correct glycan was incorrectly assigned as the ammonium adducted form of another
# glycan
AmmoniumUnmaskedRule = RevisionRule(
    HashableGlycanComposition(Hex=1, Fuc=1, Neu5Ac=-1),
    mass_shift_rule=MassShiftRule(Ammonium, -1), priority=1, name="Ammonium Unmasked")

AmmoniumUnmaskedNeuGcRule = RevisionRule(
    HashableGlycanComposition(Hex=2, Neu5Gc=-1),
    mass_shift_rule=MassShiftRule(Ammonium, -1), priority=1, name="Ammonium Unmasked NeuGc")

# The correct glycan was incorrectly assigned identified because of an error in monoisotopic peak
# assignment.
IsotopeRule = RevisionRule(HashableGlycanComposition(Fuc=-2, Neu5Ac=1), name="Isotope Error")
IsotopeRule2 = RevisionRule(HashableGlycanComposition(Fuc=-4, Neu5Ac=2), name="Isotope Error 2")

HexNAc2NeuAc2ToHex6Deoxy = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-6, HexNAc=2, Neu5Ac=2),
    mass_shift_rule=MassShiftRule(Deoxy, 1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and x.glycan_composition[hexnac] == 2 and x.mass_shifts == [Unmodified],
    name="Rare precursor deoxidation followed by large mass ambiguity")

HexNAc2NeuAc2ToHex6AmmoniumRule = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-6, HexNAc=2, Neu5Ac=2),
    mass_shift_rule=MassShiftRule(Ammonium, 1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and x.glycan_composition[hexnac] == 2,
    name="Ammonium Masked followed by Large Mass Ambiguity")

HexNAc2Fuc1NeuAc2ToHex7 = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-7, HexNAc=2, Neu5Ac=2, Fuc=1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and x.glycan_composition[hexnac] == 2,
    name="Large Mass Ambiguity")

# Moving a hydroxyl group from another monosaccharide onto a Sialic acid or vis-versa obscures their mass identity
NeuGc1Fuc1ToNeuAc1Hex1Rule = RevisionRule(
    HashableGlycanComposition(Neu5Ac=1, Hex=1, Neu5Gc=-1, Fuc=-1), name="NeuAc Masked By NeuGc")
NeuAc1Hex1ToNeuGc1Fuc1Rule = RevisionRule(
    HashableGlycanComposition(Neu5Ac=-1, Hex=-1, Neu5Gc=1, Fuc=1), name="NeuGc Masked By NeuAc")

Sulfate1HexNAc2ToHex3Rule = RevisionRule(
    HashableGlycanComposition(HexNAc=2, sulfate=1, Hex=-3), name="Sulfate + 2 HexNAc Masked By 3 Hex")
Hex3ToSulfate1HexNAc2Rule = Sulfate1HexNAc2ToHex3Rule.invert_rule()
Hex3ToSulfate1HexNAc2Rule.name = "3 Hex Masked By Sulfate + 2 HexNAc"

SulfateToPhosphateRule = RevisionRule(HashableGlycanComposition(sulfate=-1, phosphate=1), name="Phosphate Masked By Sulfate")
PhosphateToSulfateRule = SulfateToPhosphateRule.invert_rule()
PhosphateToSulfateRule.name = "Sulfate Masked By Phosphate"


class RevisionRuleList(object):
    def __init__(self, rules):
        self.rules = list(rules)
        self.by_name = {
            rule.name: rule for rule in self.rules
            if rule.name
        }

    def __getitem__(self, i):
        if i in self.by_name:
            return self.by_name[i]
        return self.rules[i]

    def __len__(self):
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __repr__(self):
        return "{self.__class__.__name__}({self.rules})".format(self=self)


class IntervalModelReviser(ModelReviser):
    alpha = 0.01

    def rescore(self, case):
        return self.model.score_interval(case, alpha=self.alpha)


# In reviser, combine the score from the absolute coordinate with the
# score relative to each other glycoform in the group? This might look
# something like the old recalibrator code, but adapted for the interval_score
# method


class RuleBasedFDREstimator(object):
    def __init__(self, rule, chromatograms, rt_model, valid_glycans=None):
        self.rule = rule
        self.chromatograms = chromatograms
        self.decoy_chromatograms = []
        self.rt_model = rt_model
        self.target_scores = []
        self.decoy_scores = []
        self.decoy_is_valid = []
        self.valid_glycans = valid_glycans
        self.estimator = None

        self.prepare()

    @property
    def name(self):
        return self.rule.name

    def prepare(self):
        self.target_scores = np.array([self.rt_model.score_interval(p, 0.01) for p in self.chromatograms])
        self.target_times = np.array([p.apex_time for p in self.chromatograms])
        self.target_predictions = np.array([
            self.rt_model.predict(p) for p in self.chromatograms
        ])
        self.target_residuals = self.target_times - self.target_predictions
        decoy_scores = []
        decoy_is_valid = []
        for p in self.chromatograms:
            p = self.rule(p)
            self.decoy_chromatograms.append(p)
            decoy_is_valid.append(p.glycan_composition in self.valid_glycans if self.valid_glycans else True)
            if self.rule.valid(p):
                decoy_scores.append(self.rt_model.score_interval(p, 0.01))
            else:
                decoy_scores.append(np.nan)

        self.decoy_times = np.array([p.apex_time for p in self.decoy_chromatograms])
        self.decoy_predictions = np.array([
            self.rt_model.predict(p) for p in self.decoy_chromatograms
        ])
        self.decoy_residuals = self.decoy_times - self.decoy_predictions


        self.decoy_scores = np.array(decoy_scores)
        self.decoy_is_valid = np.array(decoy_is_valid)

        self.target_scores[np.isnan(self.target_scores)] = 0.0
        self.decoy_scores[np.isnan(self.decoy_scores)] = 0.0
        self.chromatograms = np.array(self.chromatograms)
        self.decoy_chromatograms = np.array(self.decoy_chromatograms)

        self.estimate_fdr()

    def estimate_fdr(self):
        from glycan_profiling.tandem.target_decoy import TargetDecoyAnalyzer
        self.estimator = TargetDecoyAnalyzer(
            self.target_scores.view([('score', np.float64)]).view(np.recarray),
            self.decoy_scores[self.decoy_is_valid].view([('score', np.float64)]).view(np.recarray))

    @property
    def q_value_map(self):
        return self.estimator.q_value_map

    def score_for_fdr(self, fdr_value):
        return self.estimator.score_for_fdr(fdr_value)

    def plot(self):
        ax = self.estimator.plot()
        ax.set_xlim(-0.01, 1)
        return ax

    def __repr__(self):
        template = "{self.__class__.__name__}({self.rule})"
        return template.format(self=self)

    def get_interval_masks(self):
        spans = [mod for mod in self.rt_model.models.values()]
        centers = [s.centroid for s in spans]
        masks = [list(map(span.contains, self.target_times)) for span in spans]
        return masks, centers

    def __getstate__(self):
        state = {
            "estimator": self.estimator,
            "target_scores": self.target_scores,
            "target_predictions": self.target_predictions,
            "target_times": self.target_times,
            "decoy_scores": self.decoy_scores,
            "decoy_predictions": self.decoy_predictions,
            "decoy_times": self.decoy_times,
            "decoy_is_valid": self.decoy_is_valid,
            "rule": self.rule
        }
        return state

    def __setstate__(self, state):
        self.estimator = state['estimator']
        self.target_scores = state['target_scores']
        self.decoy_scores = state['decoy_scores']
        self.decoy_is_valid = state['decoy_is_valid']
        self.rule = state['rule']

        self.target_times = state.get("target_times", np.array([]))
        self.target_predictions = state.get("target_predictions", np.array([]))

        self.decoy_times = state.get("decoy_times", np.array([]))
        self.decoy_predictions = state.get("decoy_predictions", np.array([]))

        self.target_residuals = self.target_times - self.target_predictions
        self.decoy_residuals = self.decoy_times - self.decoy_predictions

        self.rt_model = None
        self.chromatograms = None
        self.decoy_chromatograms = None
        self.valid_glycans = None


def make_normalized_monotonic_bell(X, Y):
    center = np.abs(X).argmin()
    Ynew = np.zeros_like(Y)
    last_y = None
    for i in range(center, X.size):
        if last_y is None:
            last_y = Y[i]
        if Y[i] > last_y:
            last_y = Y[i]
        Ynew[i] = last_y
    Ynew[center:][Ynew[center:] == Ynew[center:].max()] /= Ynew[center:].max()

    last_y = None
    for i in range(center, -1, -1):
        if last_y is None:
            last_y = Y[i]
        if Y[i] > last_y:
            last_y = Y[i]
        Ynew[i] = last_y
    Ynew[:center][Ynew[:center] == Ynew[:center].max()] /= Ynew[:center].max()
    return Ynew
