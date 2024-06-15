import logging
import operator

from typing import (
    Callable, DefaultDict, Dict,
    List, Mapping, Optional, Set,
    TYPE_CHECKING, Tuple, Union,
    OrderedDict, Iterable, NamedTuple)
from array import array
from dataclasses import dataclass

from matplotlib import pyplot as plt

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenMonosaccharideResidue
from glycresoft.chromatogram_tree import mass_shift

from glycresoft.chromatogram_tree.mass_shift import Unmodified, Ammonium, Deoxy
from glycresoft.scoring.elution_time_grouping.structure import GlycopeptideChromatogramProxy
from glycresoft.structure.structure_loader import GlycanCompositionDeltaCache

from glycresoft._c.composition_network.graph import CachingGlycanCompositionVectorContext, GlycanCompositionVector

from glycresoft.task import LoggingMixin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


if TYPE_CHECKING:
    from glycresoft.scoring.elution_time_grouping.model import ElutionTimeFitter
    from glycresoft.tandem.chromatogram_mapping import SpectrumMatchUpdater, TandemAnnotatedChromatogram
    from glycresoft.tandem.target_decoy import NearestValueLookUp, TargetDecoyAnalyzer
    from glycresoft.tandem.spectrum_match import SpectrumMatch, MultiScoreSpectrumMatch


def _invert_fn(fn):
    def wrapper(*args, **kwargs):
        value = fn(*args, **kwargs)
        return not value
    return wrapper


class RevisionRule(object):
    delta_glycan: HashableGlycanComposition
    mass_shift_rule: Optional['MassShiftRule']
    priority: int
    name: Optional[str]
    delta_cache: Optional[GlycanCompositionDeltaCache]

    def __init__(self, delta_glycan, mass_shift_rule=None, priority=0, name=None):
        self.delta_glycan = HashableGlycanComposition.parse(delta_glycan)
        self.mass_shift_rule = mass_shift_rule
        self.priority = priority
        self.name = name
        self.delta_cache = None

    def monosaccharides(self) -> Set[FrozenMonosaccharideResidue]:
        return set(self.delta_glycan)

    def clone(self):
        return self.__class__(
            self.delta_glycan.clone(),
            self.mass_shift_rule.clone() if self.mass_shift_rule else None,
            self.priority,
            self.name)

    def is_valid_revision(self, record: GlycopeptideChromatogramProxy,
                          new_record: GlycopeptideChromatogramProxy) -> bool:
        valid = not any(v < 0 for v in new_record.glycan_composition.values())
        if valid:
            if self.mass_shift_rule:
                return self.mass_shift_rule.valid(record)
        return valid

    def valid(self, record: GlycopeptideChromatogramProxy) -> bool:
        new_record = self(record)
        valid = self.is_valid_revision(record, new_record)
        return valid

    def with_cache(self):
        new = self.clone()
        new.delta_cache = GlycanCompositionDeltaCache(op=operator.add)
        return new

    def without_cache(self):
        return self.clone()

    def _apply(self, record: 'GlycopeptideChromatogramProxy') -> 'GlycopeptideChromatogramProxy':
        if self.delta_cache:
            new_gc = self.delta_cache(record.glycan_composition, self.delta_glycan)
            new_record = record.copy()
            new_record.update_glycan_composition(new_gc)
            return new_record
        else:
            # Re-compute the new glycan every time
            return record.shift_glycan_composition(self.delta_glycan)

    def __call__(self, record):
        result = self._apply(record)
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
            self.delta_glycan.clone(), self.validator, self.mass_shift_rule.clone() if self.mass_shift_rule else None,
            self.priority, self.name)

    def is_valid_revision(self, record: GlycopeptideChromatogramProxy,
                          new_record: GlycopeptideChromatogramProxy) -> bool:
        if super().is_valid_revision(record, new_record):
            return self.validator(record)
        return False

    def invert_rule(self, validator: Optional[Callable[['GlycopeptideChromatogramProxy'], bool]]=None):
        if validator is None:
            validator = _invert_fn(self.validator)
        return self.__class__(
            -self.delta_glycan, validator, self.mass_shift_rule.invert() if self.mass_shift_rule else None,
            self.priority)


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
            # Can only lose a mass shift if all the mass shifts of the analyte are able to lose
            # self.mass_shift without going negative
            for shift in record.mass_shifts:
                # The current mass shift is a compound mass shift, potentially having multiple copies of self.mass_shift
                if isinstance(shift, mass_shift.CompoundMassShift) and \
                        shift.counts.get(self.mass_shift, 0) < self.multiplicity:
                    return False
                # The current mass shift is a single simple mass shift and it isn't a match for self.mass_shift
                # or self.multiplicity > 1
                elif isinstance(shift, mass_shift.MassShift) and (shift != self.mass_shift) or \
                        (shift == self.mass_shift and not self.single):
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


def modify_rules(rules: Iterable[RevisionRule], symbol_map: Mapping[str, str]):
    rules = list(rules)
    for sym_from, sym_to in symbol_map.items():
        editted: List[RevisionRule] = []
        for rule in rules:
            if sym_from in rule.delta_glycan:
                if rule.delta_cache is not None:
                    rule = rule.with_cache()
                else:
                    rule = rule.clone()
                count = rule.delta_glycan.pop(sym_from)
                rule.delta_glycan[sym_to] = count
            editted.append(rule)
        rules = editted
    return rules


def always(x):
    return True


SpectrumMatchType = Union['SpectrumMatch', 'MultiScoreSpectrumMatch']

class RevisionQueryResult(NamedTuple):
    spectrum_match: Optional[SpectrumMatchType]
    found: bool


class OriginalRevisionQueryResult(NamedTuple):
    original_spectrum_match: Optional[SpectrumMatchType]
    revision_spectrum_match: Optional[SpectrumMatchType]
    skip: bool


class RevisionValidatorBase(LoggingMixin):
    spectrum_match_builder: 'SpectrumMatchUpdater'

    def __init__(self, spectrum_match_builder, threshold_fn=always):
        self.spectrum_match_builder = spectrum_match_builder
        self.threshold_fn = threshold_fn

    def find_revision_gpsm(self, chromatogram: 'TandemAnnotatedChromatogram', revised) -> RevisionQueryResult:
        found_revision = False
        revised_gpsm = None
        try:
            revised_gpsm = chromatogram.best_match_for(revised.structure)
            found_revision = True
        except KeyError:
            if self.spectrum_match_builder is not None:
                self.spectrum_match_builder.get_spectrum_solution_sets(
                    revised,
                    chromatogram,
                    invalidate_reference=False
                )
                try:
                    revised_gpsm = chromatogram.best_match_for(revised.structure)
                    found_revision = True
                except KeyError:
                    logger.debug(
                        "Failed to find revised spectrum match for %s in %r",
                        revised.structure,
                        chromatogram)
        return RevisionQueryResult(revised_gpsm, found_revision)

    def find_gpsms(self, source: 'TandemAnnotatedChromatogram',
                   revised: GlycopeptideChromatogramProxy,
                   original: GlycopeptideChromatogramProxy) -> OriginalRevisionQueryResult:
        '''Find spectrum matches in ``source`` for the pair of alternative interpretations
        for this feature.

        Parameters
        ----------
        source : :class:`~.TandemAnnotatedChromatogram`
            The chromatogram with MS2 spectra to find spectrum matches from
        revised : :class:`~.GlycopeptideChromatogramProxy`
            The revised structure
        original : :class:`~.GlycopeptideChromatogramProxy`
            The original structure

        Returns
        -------
        original_gpsm : :class:`~.SpectrumMatchType`
            The best spectrum match for the original structure
        revised_gpsm : :class:`~.SpectrumMatchType`
            The best spectrum match for the revised structure
        skip : :class:`bool`
            Whether there were any failures to find a revised solution, telling
            the caller to skip over this one.
        '''
        original_gpsm = None
        revised_gpsm = None
        skip = False
        if source is None:
            # Can't validate without a source to read the spectrum match metadata
            skip = True
            return OriginalRevisionQueryResult(original_gpsm, revised_gpsm, skip)

        revised_gpsm, found_revision = self.find_revision_gpsm(source, revised)

        if not found_revision:
            # Can't find a spectrum match to the revised form, assume we're allowed to
            # revise.
            self.debug(
                "...... Permitting revision for %s (%0.3f) from %s to %s because revision not evaluated",
                revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition)
            skip = True
            return OriginalRevisionQueryResult(original_gpsm, revised_gpsm, skip)

        try:
            original_gpsm = source.best_match_for(original.structure)
        except KeyError:
            # Can't find a spectrum match to the original, assume we're allowed to
            # revise.
            self.debug(
                "...... Permitting revision for %s (%0.3f) from %s to %s because original not evaluated" %
                (revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition))
            skip = True
            return OriginalRevisionQueryResult(original_gpsm, revised_gpsm, skip)
        skip = revised_gpsm is None or original_gpsm is None
        return OriginalRevisionQueryResult(original_gpsm, revised_gpsm, skip)

    def validate(self, revised: GlycopeptideChromatogramProxy, original: GlycopeptideChromatogramProxy) -> bool:
        raise NotImplementedError()

    def __call__(self, revised: GlycopeptideChromatogramProxy, original: GlycopeptideChromatogramProxy) -> bool:
        return self.validate(revised, original)

    def msn_score_for(self, source: 'TandemAnnotatedChromatogram', solution: GlycopeptideChromatogramProxy) -> float:
        gpsm, found = self.find_revision_gpsm(source, solution)
        if not found or gpsm is None:
            return 0
        return gpsm.score


class PeptideYUtilizationPreservingRevisionValidator(RevisionValidatorBase):
    '''Prevent a revision that leads to substantial loss of peptide+Y ion-explainable
    signal.

    This does not use peptide backbone fragments because they are by definition the
    revision algorithm only updates the glycan composition, not the peptide backbone.
    '''
    threshold: float

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
        if hasattr(original_gpsm, 'score_set'):
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
        else:
            # TODO: Make this check explicit by directly calculating the metric here.
            valid = True
        return valid


class OxoniumIonRequiringUtilizationRevisionValidator(RevisionValidatorBase):
    scale: float

    def __init__(self, scale=1.0, spectrum_match_builder=None, threshold_fn=always):
        super(OxoniumIonRequiringUtilizationRevisionValidator,
              self).__init__(spectrum_match_builder, threshold_fn)
        self.scale = scale

    def validate(self, revised: GlycopeptideChromatogramProxy, original: GlycopeptideChromatogramProxy) -> bool:
        source = revised.source
        if source is None:
            # Can't validate without a source to read the spectrum match metadata
            return True

        original_gpsm, revised_gpsm, skip = self.find_gpsms(
            source, revised, original)
        if skip:
            return True
        if hasattr(original_gpsm, 'score_set'):
            original_utilization: float = original_gpsm.score_set.oxonium_ion_intensity_utilization
            revised_utilization: float = revised_gpsm.score_set.oxonium_ion_intensity_utilization

            threshold = original_utilization * self.scale
            valid = (revised_utilization - threshold) >= 0
            self.debug(
                "...... Checking revision by oxonium ions for %s (%0.3f) from %s to %s: %0.3f / %0.3f: %r" %
                (revised.tag, revised.apex_time, original.glycan_composition, revised.glycan_composition,
                revised_utilization, original_utilization, valid))
        else:
            # TODO: Make this check explicit by directly calculating the metric here.
            valid = True
        return valid


class CompoundRevisionValidator(object):
    validators: List[RevisionValidatorBase]

    def __init__(self, validators=None):
        if validators is None:
            validators = []
        self.validators = list(validators)

    @property
    def spectrum_match_builder(self):
        return self.validators[0].spectrum_match_builder

    def validate(self, revised, original):
        for rule in self.validators:
            if not rule(revised, original):
                return False
        return True

    def __call__(self, revised, original):
        return self.validate(revised, original)

    def msn_score_for(self, source, solution: GlycopeptideChromatogramProxy) -> float:
        return self.validators[0].msn_score_for(source, solution)


def _new_array():
    return array('d')


def display_table(rows: List[str], headers: List[str]) -> str:
    rows = [headers] + rows
    widths = list(map(lambda col: max(map(len, col)), zip(*rows)))

    buffer = []
    for i, row in enumerate(rows):
        buffer.append(' | '.join(col.center(w) for col, w in zip(row, widths)))
        if i == 0:
            buffer.append(
                '|'.join('-' * (w + 2 if j > 0 else w + 1)
                         for j, (col, w) in enumerate(zip(row, widths)))
            )

    return '\n'.join(buffer)


@dataclass
class RevisionEvent:
    rt_score: float
    chromatogram_record: GlycopeptideChromatogramProxy
    rule: Optional[RevisionRule]
    msn_score: float


class ValidatedGlycome:
    valid_glycans: Set[HashableGlycanComposition]
    monosaccharides: Set[FrozenMonosaccharideResidue]

    def __init__(self, valid_glycans):
        self.valid_glycans = valid_glycans
        self.monosaccharides = self._collect_monosaccharides()

    def _collect_monosaccharides(self):
        keys = set()
        for gc in self.valid_glycans:
            keys.update(gc)
        return keys

    def __contains__(self, glycan_composition: HashableGlycanComposition) -> bool:
        return glycan_composition in self.valid_glycans

    def __bool__(self):
        return bool(self.valid_glycans)

    def __nonzero__(self):
        return bool(self.valid_glycans)

    def __iter__(self):
        return iter(self.valid_glycans)

    def __len__(self):
        return len(self.valid_glycans)

    def check(self, record: GlycopeptideChromatogramProxy) -> bool:
        return record.glycan_composition in self.valid_glycans


class IndexedValidatedGlycome(ValidatedGlycome):
    indexer: CachingGlycanCompositionVectorContext
    indexed_valid_glycans: Set[GlycanCompositionVector]

    def __init__(self, valid_glycans):
        super().__init__(valid_glycans)
        self.indexer = CachingGlycanCompositionVectorContext(self.monosaccharides)
        self.indexed_valid_glycans = {
            self.indexer.encode(gc) for gc in self.valid_glycans
        }

    def encode(self, record: GlycopeptideChromatogramProxy):
        if "_indexed_glycan_composition" in record.kwargs:
            return record.kwargs["_indexed_glycan_composition"]
        vec = self.indexer.encode(record.glycan_composition)
        record.kwargs["_indexed_glycan_composition"] = vec
        return vec

    def check(self, record: GlycopeptideChromatogramProxy) -> bool:
        return self.encode(record) in self.indexed_valid_glycans



class ModelReviser(LoggingMixin):
    valid_glycans: Optional[ValidatedGlycome]
    revision_validator: Optional[RevisionValidatorBase]

    model: 'ElutionTimeFitter'
    rules: 'RevisionRuleList'
    chromatograms: List[GlycopeptideChromatogramProxy]

    original_scores: array
    original_times: array

    alternative_records: DefaultDict[RevisionRule, List[GlycopeptideChromatogramProxy]]
    alternative_scores: DefaultDict[RevisionRule, array]
    alternative_times: DefaultDict[RevisionRule, array]

    def __init__(self, model, rules, chromatograms=None, valid_glycans=None, revision_validator=None):
        if chromatograms is None:
            chromatograms = model.chromatograms
        self.model = model
        self.chromatograms = chromatograms
        self.original_scores = array('d')
        self.original_times = array('d')
        self.rules = list(rules)
        self.alternative_records = DefaultDict(list)
        self.alternative_scores = DefaultDict(_new_array)
        self.alternative_times = DefaultDict(_new_array)
        if valid_glycans is not None:
            if isinstance(valid_glycans, set):
                valid_glycans = ValidatedGlycome(valid_glycans)
        self.valid_glycans = valid_glycans
        self.revision_validator = revision_validator

    def rescore(self, case):
        return self.model.score(case)

    def modify_rules(self, symbol_map):
        self.rules = self.rules.modify_rules(symbol_map)
        return self

    def propose_revisions(self, case):
        propositions = {
            None: (case, self.rescore(case)),
        }
        for rule in self.rules:
            if rule.valid(case):
                rec = rule(case)
                if self.valid_glycans and not self.valid_glycans.check(rec):
                    continue
                propositions[rule] = (rec, self.rescore(rec))
        return propositions

    def process_rule(self, rule: RevisionRule):
        alts = []
        alt_scores = array('d')
        alt_times = array('d')
        for case in self.chromatograms:
            rec = rule(case)
            if rule.is_valid_revision(case, rec):
                if self.valid_glycans and not self.valid_glycans.check(rec):
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

    def select_revision_atlernative(self, alternatives: List[RevisionEvent],
                                    original: GlycopeptideChromatogramProxy,
                                    minimum_rt_score: float) -> GlycopeptideChromatogramProxy:
        alternatives = [alt for alt in alternatives if alt.rt_score > minimum_rt_score]
        alternatives.sort(key=lambda x: x.msn_score, reverse=True)

        table = [
            (str(alt.chromatogram_record.glycan_composition),
            (','.join(map(lambda x: x.name, alt.chromatogram_record.mass_shifts))),
            (alt.rule.name if alt.rule is not None else '-'),
             '%0.3f' % alt.rt_score, '%0.3f' % alt.msn_score)
            for alt in alternatives
        ]

        if len({alt.chromatogram_record.glycan_composition for alt in alternatives}) > 1:
            self.log('\n' + display_table(table, ['Glycan Composition', 'Mass Shifts',
                                                  'Revision Rule', 'RT Score', 'MSn Score']))

        best_event = alternatives[0]
        best_record = best_event.chromatogram_record
        best_rule = best_event.rule
        if best_rule is not None: # If *somehow* the best solution at the time this is called is still unchanged?
            best_record.revised_from = (best_rule, original)
        return best_record

    def _msn_score_if_available(self, source, solution: GlycopeptideChromatogramProxy) -> float:
        if self.revision_validator is None:
            return 0.0
        return self.revision_validator.msn_score_for(source, solution)

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

            alternative_solutions = [
                RevisionEvent(original_score, original_solution, None,
                              self._msn_score_if_available(
                                original_solution.source,
                                original_solution)
                )
            ]

            for rule in self.rules:
                alt_rec = self.alternative_records[rule][i]
                alt_score = self.alternative_scores[rule][i]
                alt_time = self.alternative_times[rule][i]
                if not np.isclose(alt_score, 0.0) and abs(alt_time - original_time) > minimum_time_difference:
                    if self.revision_validator and not self.revision_validator(alt_rec, original_solution):
                        continue
                    alternative_solutions.append(
                        RevisionEvent(
                            alt_score, alt_rec, rule,
                            self._msn_score_if_available(
                                original_solution.source, alt_rec)))
                    if alt_score > best_score:
                        delta_best_score = alt_score - original_score
                        best_score = alt_score
                        best_rule = rule
                        best_record = alt_rec


            if best_score > threshold and delta_best_score > delta_threshold:
                if best_rule is not None:
                    best_record = self.select_revision_atlernative(
                        alternative_solutions, original_solution, best_score * 0.9)
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

# The glycan was incorrectly assigned identified because of an error in monoisotopic peak
# assignment.
IsotopeRule = RevisionRule(HashableGlycanComposition(Fuc=-2, Neu5Ac=1), name="Isotope Error")
IsotopeRule2 = RevisionRule(HashableGlycanComposition(Fuc=-4, Neu5Ac=2), name="Isotope Error 2")

IsotopeRuleNeuGc = RevisionRule(HashableGlycanComposition(Fuc=-1, Hex=-1, NeuGc=1), name="Isotope Error NeuGc")

HexNAc2NeuAc2ToHex6Deoxy = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-6, HexNAc=2, Neu5Ac=2),
    mass_shift_rule=MassShiftRule(Deoxy, 1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and \
            x.glycan_composition[hexnac] == 2 and x.mass_shifts == [Unmodified],
    name="Rare precursor deoxidation followed by large mass ambiguity")

HexNAc2NeuAc2ToHex6AmmoniumRule = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-6, HexNAc=2, Neu5Ac=2),
    mass_shift_rule=MassShiftRule(Ammonium, 1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and \
            x.glycan_composition[hexnac] == 2,
    name="Ammonium Masked followed by Large Mass Ambiguity")


# Not yet in use, needs to be explored more. In practice, this cannot even be used
# with such a small chromatographic shift (<= 0.5 minutes) on fucose and hexose.
IsotopeAmmoniumFucToHex = RevisionRule(HashableGlycanComposition(Fuc=-1, Hex=1),
                                       mass_shift_rule=MassShiftRule(Ammonium, 1),
                                       name='Ammonium Isotope Error')

# Not yet in use, needs to be explored more
dHex2HexNAc2NeuAc1ToHex6AmmoniumRule = ValidatingRevisionRule(
    HashableGlycanComposition(Fuc=2, Hex=-6, HexNAc=2, Neu5Ac=1),
    mass_shift_rule=MassShiftRule(Ammonium, 1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(
        dhex) == 0 and x.glycan_composition[hexnac] == 1,
    name="Ammonium Masked followed by Large Mass Ambiguity II")


HexNAc2Fuc1NeuAc2ToHex7 = ValidatingRevisionRule(
    HashableGlycanComposition(Hex=-7, HexNAc=2, Neu5Ac=2, Fuc=1),
    validator=lambda x: x.glycan_composition[neuac] == 0 and x.glycan_composition.query(dhex) == 0 and \
            x.glycan_composition[hexnac] == 2,
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

Phosphate1HexNAc2ToHex3Rule = RevisionRule(
    HashableGlycanComposition(HexNAc=2, phosphate=1, Hex=-3), name="Phosphate + 2 HexNAc Masked By 3 Hex")
Hex3ToPhosphate1HexNAc2Rule = Phosphate1HexNAc2ToHex3Rule.invert_rule()
Hex3ToPhosphate1HexNAc2Rule.name = "3 Hex Masked By Phosphate + 2 HexNAc"

SulfateToPhosphateRule = RevisionRule(
    HashableGlycanComposition(sulfate=-1, phosphate=1),
    name="Phosphate Masked By Sulfate")
PhosphateToSulfateRule = SulfateToPhosphateRule.invert_rule()
PhosphateToSulfateRule.name = "Sulfate Masked By Phosphate"


class RevisionRuleList(object):
    rules: List[RevisionRule]
    by_name: Dict[str, RevisionRule]

    def __init__(self, rules):
        self.rules = list(rules)
        self.by_name = {
            rule.name: rule for rule in self.rules
            if rule.name
        }

    def modify_rules(self, symbol_map):
        self.rules = modify_rules(self.rules, symbol_map)
        return self

    def with_cache(self):
        return self.__class__([
            rule.with_cache() for rule in self.rules
        ])

    def filter_defined(self, valid_glycans: ValidatedGlycome):
        acc = []
        for rule in self.rules:
            keys = rule.monosaccharides()
            if keys == (keys & valid_glycans.monosaccharides):
                acc.append(rule)
        self.rules = acc
        return self

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
    alpha: float = 0.01

    def rescore(self, case):
        return self.model.score_interval(case, alpha=self.alpha)


# In reviser, combine the score from the absolute coordinate with the
# score relative to each other glycoform in the group? This might look
# something like the old recalibrator code, but adapted for the interval_score
# method


class FDREstimatorBase(object):
    estimator: 'TargetDecoyAnalyzer'

    def estimate_fdr(self):
        from glycresoft.tandem.target_decoy import TargetDecoyAnalyzer
        self.estimator = TargetDecoyAnalyzer(
            self.target_scores.view([('score', np.float64)]).view(np.recarray),
            self.decoy_scores[self.decoy_is_valid].view([('score', np.float64)]).view(np.recarray))

    @property
    def q_value_map(self):
        return self.estimator.q_value_map

    def score_for_fdr(self, fdr_value: float) -> float:
        return self.estimator.score_for_fdr(fdr_value)

    def plot(self):
        ax = self.estimator.plot()
        ax.set_xlim(-0.01, 1)
        return ax


class RuleBasedFDREstimator(FDREstimatorBase):
    rule: RevisionRule

    chromatograms: Union[List[GlycopeptideChromatogramProxy], np.ndarray]
    decoy_chromatograms: Union[List[GlycopeptideChromatogramProxy], np.ndarray]

    target_scores: np.ndarray
    decoy_scores: np.ndarray

    target_predictions: np.ndarray
    decoy_predictions: np.ndarray
    decoy_residuals: np.ndarray

    decoy_is_valid: np.ndarray
    over_time: OrderedDict

    def __init__(self, rule, chromatograms, rt_model, valid_glycans=None):
        self.rule = rule
        self.chromatograms = chromatograms
        self.decoy_chromatograms = []
        self.rt_model = rt_model
        self.target_scores = np.array([])
        self.decoy_scores = np.array([])
        self.decoy_is_valid = np.array([])
        self.valid_glycans = valid_glycans
        self.estimator = None
        self.over_time = OrderedDict()

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
        self.decoy_chromatograms = []
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

    def __repr__(self):
        template = "{self.__class__.__name__}({self.rule})"
        return template.format(self=self)

    def get_residuals_from_interval(self, span):
        target_mask = [span.contains(i) for i in self.target_times]
        target_residuals = self.target_residuals[target_mask]
        target_residuals = target_residuals[~np.isnan(target_residuals)]

        decoy_mask = [span.contains(i) for i in self.decoy_times]
        decoy_residuals = self.decoy_residuals[decoy_mask & self.decoy_is_valid]
        decoy_residuals = decoy_residuals[~np.isnan(decoy_residuals)]
        return target_residuals, decoy_residuals

    def get_scores_from_interval(self, span):
        target_mask = [span.contains(i) for i in self.target_times]
        target_scores = self.target_scores[target_mask]
        target_scores = target_scores[~np.isnan(target_scores)]

        decoy_mask = [span.contains(i) for i in self.decoy_times]
        decoy_scores = self.decoy_scores[decoy_mask &
                                               self.decoy_is_valid]
        decoy_scores = decoy_scores[~np.isnan(decoy_scores)]
        return target_scores, decoy_scores

    def get_scores_from_mask(self, mask):
        target_scores = self.target_scores[mask]
        target_scores = target_scores[~np.isnan(target_scores)]

        decoy_scores = self.decoy_scores[mask &
                                         self.decoy_is_valid]
        decoy_scores = decoy_scores[~np.isnan(decoy_scores)]
        return target_scores, decoy_scores

    def get_interval_masks(self):
        spans = [mod for mod in self.rt_model.models.values()]
        centers = np.array([s.centroid for s in spans])
        masks = [
            np.fromiter(map(span.contains, self.target_times), bool) for span in spans]
        return masks, centers

    def fit_over_time(self):
        masks, centers = self.get_interval_masks()
        for i, mask in enumerate(masks):
            target_scores, decoy_scores = self.get_scores_from_mask(mask)
            facet = FDRFacet(target_scores, decoy_scores)
            self.over_time[centers[i]] = facet

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
            "rule": self.rule,
            "over_time": self.over_time
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
        self.over_time = state.get("over_time", OrderedDict())

        self.target_residuals = self.target_times - self.target_predictions
        self.decoy_residuals = self.decoy_times - self.decoy_predictions

        self.rt_model = None
        self.chromatograms = None
        self.decoy_chromatograms = None
        self.valid_glycans = None

    def copy(self):
        proto, newargs, state = self.__reduce__()
        dup = proto(*newargs)
        dup.__setstate__(state)

        dup.rt_model = self.rt_model
        dup.chromatograms = self.chromatograms
        dup.decoy_chromatograms = self.decoy_chromatograms
        dup.valid_glycans = self.valid_glycans
        return dup


class FDRFacet(FDREstimatorBase):
    def __init__(self, target_scores, decoy_scores):
        self.target_scores = target_scores
        self.decoy_scores = decoy_scores
        if decoy_scores is None:
            self.decoy_is_valid = np.array([], dtype=bool)
        else:
            self.decoy_is_valid = np.ones_like(decoy_scores).astype(bool)
        if self.target_scores is not None and self.decoy_scores is not None:
            self.estimate_fdr()

    def __getstate__(self):
        state = {
            "estimator": self.estimator
        }
        return state

    def __setstate__(self, state):
        self.estimator = state['estimator']
        self.target_scores = self.estimator.targets['score']
        self.decoy_scores = self.estimator.decoys['score']
        self.decoy_is_valid = np.ones_like(self.decoy_scores).astype(bool)

    def __reduce__(self):
        return self.__class__, (None, None), self.__getstate__()


def make_normalized_monotonic_bell(X: np.ndarray, Y: np.ndarray, symmetric: bool=False) -> np.ndarray:
    center = np.abs(X).argmin()
    Ynew = np.zeros_like(Y)
    last_y = None
    for i in range(center, X.size):
        if last_y is None:
            last_y = Y[i]
        if Y[i] > last_y:
            last_y = Y[i]
        Ynew[i] = last_y

    last_y = None
    for i in range(center, -1, -1):
        if last_y is None:
            last_y = Y[i]
        if Y[i] > last_y:
            last_y = Y[i]
        Ynew[i] = last_y

    if symmetric:
        Ynew = (Ynew + Ynew[::-1]) / 2

    try:
        maximum = Ynew[:center].max()
        Ynew[:center] /= maximum
    except ValueError:
        pass

    try:
        maximum = Ynew[center:].max()
        Ynew[center:] /= maximum
    except ValueError:
        pass

    return Ynew


def dropna(array: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(array)
    return array[mask]


class ResidualFDREstimator(object):
    rules: RevisionRuleList
    residual_mapper: 'PosteriorErrorToScore'

    def __init__(self, rules, rt_model=None, residual_mapper=None):
        self.rules = rules
        self.rt_model = rt_model
        self.residual_mapper = residual_mapper
        if residual_mapper is None:
            self.fit()

    def _extract_scores_from_rules(self) -> Tuple[np.ndarray, np.ndarray]:
        rule = self.rules[0]
        target_scores = rule.target_residuals[
            ~np.isnan(rule.target_residuals)]
        all_decoys = []
        for rule in self.rules:
            all_decoys.extend(rule.decoy_residuals[
                ~np.isnan(rule.decoy_residuals) & rule.decoy_is_valid])
        return target_scores, np.array(all_decoys)

    def _extract_scores_from_model(self, rt_model, seed=1, n_resamples=1) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        decoys = np.array([])
        for i in range(n_resamples):
            decoys = np.concatenate(
                (
                    decoys,
                    dropna(
                        rt_model._summary_statistics['apex_time_array'] - rng.permutation(
                            rt_model._summary_statistics['predicted_apex_time_array'].copy())
                    )
                )
            )

        targets = dropna(rt_model._summary_statistics['residuals_array'])
        return targets, decoys

    def fit(self):
        models: List[PosteriorErrorToScore] = []
        if self.rules:
            try:
                fit_from_rules = self.fit_from_rules()
                models.append(fit_from_rules)
            except Exception:
                logger.error("Failed to fit FDR from delta rules", exc_info=True)

        if self.rt_model:
            try:
                fit_from_rt_model = self.fit_from_rt_model()
                models.append(fit_from_rt_model)
            except Exception as err:
                show_traceback = str(err) != "No non-zero range in the posterior error distribution."
                logger.error(
                    "Failed to fit FDR from permutations: %r", err, exc_info=show_traceback)
        if models:
            models.sort(key=lambda x: x.width_at_half_max())
            model = models[0]
            self.residual_mapper = model
        else:
            raise ValueError("Could not fit any FDR model for retention time!")

    def fit_from_rules(self):
        from glycresoft.tandem.glycopeptide.dynamic_generation.multipart_fdr import (
            FiniteMixtureModelFDREstimatorDecoyGaussian
        )
        target_scores, all_decoy_scores = self._extract_scores_from_rules()
        fmm = FiniteMixtureModelFDREstimatorDecoyGaussian(all_decoy_scores, target_scores)
        fmm.fit(max_target_components=3,
                max_decoy_components=len(self.rules) * 2)
        return PosteriorErrorToScore.from_model(fmm)

    def fit_from_rt_model(self):
        from glycresoft.tandem.glycopeptide.dynamic_generation.multipart_fdr import (
            FiniteMixtureModelFDREstimatorDecoyGaussian
        )
        target_scores, all_decoy_scores = self._extract_scores_from_model(self.rt_model)
        fmm = FiniteMixtureModelFDREstimatorDecoyGaussian(
            all_decoy_scores, target_scores)
        fmm.fit(max_target_components=3)
        return PosteriorErrorToScore.from_model(fmm)

    def bounds_for_probability(self, probability: float) -> np.ndarray:
        return self.residual_mapper.bounds_for_probability(probability)

    def __reduce__(self):
        return self.__class__, (self.rules, None, self.residual_mapper)

    def __getitem__(self, i):
        return self.rules[i]

    def __len__(self):
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def plot(self, ax=None, **kwargs):
        return self.residual_mapper.plot(ax, **kwargs)

    def score(self, chromatogram: GlycopeptideChromatogramProxy, *args, **kwargs) -> float:
        t_pred = self.rt_model.predict(chromatogram, *args, **kwargs)
        residual = chromatogram.apex_time - t_pred
        return self.residual_mapper(residual)

    def __call__(self, chromatogram: GlycopeptideChromatogramProxy, *args, **kwds) -> float:
        return self.score(chromatogram, *args, **kwds)


def _prepare_domain(target_scores, decoy_scores, delta=0.1):
    lo = min(target_scores.min(), decoy_scores.min())
    hi = max(target_scores.max(), decoy_scores.max())
    sign = np.sign(lo)
    lo = sign * max(abs(lo), abs(hi))
    sign = np.sign(hi)
    hi = sign * max(abs(lo), abs(hi))
    domain = np.concatenate(
        (np.arange(lo, 0, delta), np.arange(0.0, hi + delta, delta), ))
    return domain


class PosteriorErrorToScore(object):
    mapper: 'NearestValueLookUp'
    domain: np.ndarray
    normalized_score: np.ndarray

    @classmethod
    def from_model(cls, model, delta=0.1, symmetric=True):
        domain = _prepare_domain(model.target_scores, model.decoy_scores, delta=delta)
        return cls(model, domain, symmetric=symmetric)

    def __init__(self, model, domain, symmetric=True):
        self.model = model
        self.domain = domain
        self.normalized_score = None
        self.mapper = None
        if self.model is not None:
            self._create_normalized(symmetric=symmetric)

    def _create_normalized(self, symmetric=True):
        from glycresoft.tandem.target_decoy import NearestValueLookUp
        Y = self.model.estimate_posterior_error_probability(self.domain)
        self.normalized_score = np.clip(
            1 - make_normalized_monotonic_bell(self.domain, Y, symmetric=symmetric), 0, 1)
        support = np.where(self.normalized_score != 0)[0]
        if len(support) == 0:
            raise ValueError("No non-zero range in the posterior error distribution.")
        lo, hi = support[[0, -1]]
        lo = max(lo - 5, 0)
        hi += 5
        self.domain = self.domain[lo:hi]
        self.normalized_score = self.normalized_score[lo:hi]
        self.mapper = NearestValueLookUp(zip(self.domain, self.normalized_score))

    def __call__(self, value):
        if isinstance(value, (np.ndarray, Iterable)):
            return np.array([self.mapper[v] for v in value])
        return self.mapper[value]

    def __getstate__(self):
        return {
            "mapper": self.mapper,
        }

    def __setstate__(self, state):
        self.mapper = state["mapper"]
        self.domain, self.normalized_score = map(np.array, zip(*self.mapper.items))

    def __reduce__(self):
        return self.__class__, (None, None), self.__getstate__()

    def copy(self):
        proto, newargs, state = self.__reduce__()
        dup = proto(*newargs)
        dup.__setstate__(state)
        dup.model = self.model
        return dup

    def bounds_for_probability(self, probability: float) -> np.ndarray:
        xbounds = np.where(self.normalized_score >= probability)[0]
        if len(xbounds) == 0:
            return np.array([0.0, 0.0])
        lo, hi = xbounds[[0, -1]]
        return self.domain[[lo, hi]]

    def at_half_max(self):
        half_max = self.normalized_score.max() / 2
        return self.bounds_for_probability(half_max)

    def width_at_half_max(self) -> float:
        lo, hi = self.at_half_max()
        return hi - lo

    def plot(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1, 1)
        ax.plot(self.domain, self.normalized_score)
        ax.set_xlabel(r"$t - \hat{t}$")
        ax.set_ylabel("Probability")
        return ax
