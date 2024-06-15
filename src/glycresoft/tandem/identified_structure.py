'''A set of base implementations for summarizing identified structures
with both MS1 and MSn.
'''
from typing import Any, List, Optional, Set
import numpy as np

from glycresoft.chromatogram_tree import get_chromatogram, ArithmeticMapping, ChromatogramInterface
from glycresoft.serialize.chromatogram import ChromatogramSolution
from glycresoft.tandem.spectrum_match.solution_set import SpectrumSolutionSet
from glycresoft.tandem.spectrum_match.spectrum_match import FDRSet, ScoreSet, SpectrumMatch

from .chromatogram_mapping import TandemSolutionsWithoutChromatogram, TargetType


class IdentifiedStructure(object):
    """A base class for summarizing structures identified by tandem MS
    and matched to MS1 signal.

    Attributes
    ----------
    structure: object
        The structure object identified.
    spectrum_matches: :class:`list` of :class:`~.SolutionSet`
        The MSn spectra that were matched. This is a list of solution sets,
        so other structures may be present, and in the case of multi-score
        matches, they may even appear to score higher.
    chromatogram: :class:`~.ChromatogramSolution`
        The MS1 signal over time matched.
    ms2_score: float
        The best MSn identification score. Usually the highest score amongst
        all spectra matched.
    q_value: float
        The best FDR estimate for this structure. Always the lowest amongst
        all spectra matched.
    ms1_score: float
        The score of :attr:`chromatogram`, or :const:`0` if it is :const:`None`.
    total_signal: float
        The total abundance of :attr:`chromatogram`.
    shared_with: list
        Other :class:`IdentifiedStructure` instances which match the same signal.
    """

    structure: TargetType

    chromatogram: ChromatogramSolution

    spectrum_matches: List[SpectrumSolutionSet]
    _best_spectrum_match: Optional[SpectrumMatch]

    ms1_score: float
    ms2_score: float
    q_value: float

    charge_states: Set[int]
    shared_with: List

    def __init__(self, structure, spectrum_matches, chromatogram, shared_with=None):
        if shared_with is None:
            shared_with = []
        self.structure = structure
        self.spectrum_matches = spectrum_matches
        self.chromatogram = chromatogram
        self._best_spectrum_match = None
        self.ms2_score = None
        self.q_value = None
        self._find_best_spectrum_match()
        self.ms1_score = chromatogram.score if chromatogram is not None else 0
        self.charge_states = chromatogram.charge_states if chromatogram is not None else {
            psm.scan.precursor_information.charge for psm in spectrum_matches
        }
        self.shared_with = shared_with

    def clone(self):
        dup = self.__class__(
            self.structure,
            [sset.clone() for sset in self.spectrum_matches],
            self.chromatogram.clone(),
            self.shared_with[:])
        return dup

    def is_multiscore(self):
        """Check whether this match has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        try:
            return self.spectrum_matches[0].is_multiscore()
        except IndexError:
            return False

    @property
    def score_set(self) -> ScoreSet:
        """The :class:`~.ScoreSet` of the best MS/MS match

        Returns
        -------
        :class:`~.ScoreSet`
        """
        if not self.is_multiscore():
            return None
        best_match = self._best_spectrum_match
        if best_match is None:
            return None
        return best_match.score_set

    @property
    def best_spectrum_match(self):
        return self._best_spectrum_match

    @property
    def q_value_set(self) -> FDRSet:
        """The :class:`~.FDRSet` of the best MS/MS match

        Returns
        -------
        :class:`~.FDRSet`
        """
        if not self.is_multiscore():
            return None
        best_match = self._best_spectrum_match
        if best_match is None:
            return None
        return best_match.q_value_set

    @property
    def key(self):
        return self.chromatogram.key

    def _find_best_spectrum_match(self):
        is_multiscore = self.is_multiscore()
        best_match = None
        if is_multiscore:
            best_score = -float('inf')
            best_q_value = float('inf')
        else:
            best_score = -float('inf')
        for solution_set in self.spectrum_matches:
            try:
                match = solution_set.solution_for(self.structure)
                if is_multiscore:
                    q_value = match.q_value
                    if q_value <= best_q_value:
                        q_delta = abs(best_q_value - q_value)
                        best_q_value = q_value
                        if q_delta > 0.001:
                            best_score = match.score
                            best_match = match
                        else:
                            score = match.score
                            if score > best_score:
                                best_score = score
                                best_match = match
                else:
                    score = match.score
                    if score > best_score:
                        best_score = score
                        best_match = match
            except KeyError:
                continue
        if best_match is None:
            self._best_spectrum_match = None
            self.ms2_score = 0
            self.q_value = 1.0
        else:
            self._best_spectrum_match = best_match
            self.ms2_score = best_match.score
            self.q_value = best_match.q_value
        return best_match

    @property
    def neutral_mass(self):
        return self.observed_neutral_mass

    @property
    def observed_neutral_mass(self):
        try:
            return self.chromatogram.neutral_mass
        except AttributeError:
            return self.spectrum_matches[0].scan.precursor_information.neutral_mass

    @property
    def weighted_neutral_mass(self):
        try:
            return self.chromatogram.weighted_neutral_mass
        except AttributeError:
            return self.observed_neutral_mass

    @property
    def start_time(self) -> float:
        try:
            return self.chromatogram.start_time
        except AttributeError:
            return None

    @property
    def end_time(self) -> float:
        try:
            return self.chromatogram.end_time
        except AttributeError:
            return None

    @property
    def apex_time(self) -> float:
        try:
            return self.chromatogram.apex_time
        except AttributeError:
            return None

    def as_arrays(self):
        try:
            return self.chromatogram.as_arrays()
        except AttributeError:
            return np.array([]), np.array([])

    @property
    def integrated_abundance(self):
        if self.chromatogram is None:
            return 0
        return self.chromatogram.integrated_abundance

    def integrate(self):
        if self.chromatogram is None:
            return 0
        return self.chromatogram.integrate()

    @property
    def total_signal(self):
        if self.chromatogram is None:
            return 0
        return self.chromatogram.total_signal

    @property
    def mass_shifts(self):
        try:
            return self.chromatogram.mass_shifts
        except AttributeError:
            shifts = set()
            for solution in self.tandem_solutions:
                try:
                    sm = solution.solution_for(self.structure)
                    shifts.add(sm.mass_shift)
                except KeyError:
                    continue
            return list(shifts)

    def mass_shift_signal_fractions(self):
        try:
            return self.chromatogram.mass_shift_signal_fractions()
        except AttributeError:
            return ArithmeticMapping()

    def __repr__(self):
        return "IdentifiedStructure(%s, %0.3f, %0.3f, %0.3e)" % (
            self.structure, self.ms2_score, self.ms1_score, self.total_signal)

    def get_chromatogram(self):
        return self.chromatogram

    def is_distinct(self, other):
        return get_chromatogram(self).is_distinct(get_chromatogram(other))

    @property
    def tandem_solutions(self):
        return self.spectrum_matches

    @tandem_solutions.setter
    def tandem_solutions(self, value):
        self.spectrum_matches = value

    def mass_shift_tandem_solutions(self):
        mapping = ArithmeticMapping()
        for sm in self.tandem_solutions:
            try:
                mapping[sm.solution_for(self.structure).mass_shift] += 1
            except KeyError:
                continue
        return mapping

    @property
    def glycan_composition(self):
        return self.structure.glycan_composition

    def __eq__(self, other):
        try:
            structure_eq = self.structure == other.structure
            if structure_eq:
                chromatogram_eq = self.chromatogram == other.chromatogram
                if chromatogram_eq:
                    spectrum_matches_eq = self.spectrum_matches == other.spectrum_matches
                    return spectrum_matches_eq
            return False
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.structure, self.chromatogram))

    def __iter__(self):
        if self.chromatogram is None:
            return iter(())
        return iter(self.chromatogram)

    def __len__(self):
        if self.chromatogram is None:
            return 0
        return len(self.chromatogram)

    def __getitem__(self, i):
        if self.chromatogram is None:
            raise IndexError(i)
        return self.chromatogram[i]

    def overlaps_in_time(self, x):
        if self.chromatogram is None:
            return False
        return self.chromatogram.overlaps_in_time(x)

    @property
    def composition(self):
        return self.chromatogram.composition

    def has_scan(self, scan_id: str) -> bool:
        return any([sset.scan_id == scan_id for sset in self.tandem_solutions])

    def get_scan(self, scan_id: str) -> SpectrumMatch:
        for sset in self.tandem_solutions:
            if sset.scan_id == scan_id:
                return sset
        raise KeyError(scan_id)

    def has_chromatogram(self):
        return self.chromatogram is not None

ChromatogramInterface.register(IdentifiedStructure)


def extract_identified_structures(tandem_annotated_chromatograms, threshold_fn, result_type=IdentifiedStructure):
    identified_structures = []
    unassigned = []

    for chroma in tandem_annotated_chromatograms:
        if chroma.composition is not None:
            if hasattr(chroma, 'entity'):
                try:
                    representers = chroma.representative_solutions
                    if representers is None:
                        representers = chroma.most_representative_solutions(threshold_fn)
                except AttributeError:
                    representers = chroma.most_representative_solutions(threshold_fn)
                bunch = []
                if isinstance(chroma, TandemSolutionsWithoutChromatogram):
                    chromatogram_entry = None
                else:
                    chromatogram_entry = chroma
                for representer in representers:
                    ident = result_type(representer.solution, chroma.tandem_solutions,
                                        chromatogram_entry, [])
                    bunch.append(ident)
                for i, ident in enumerate(bunch):
                    ident.shared_with = bunch[:i] + bunch[i + 1:]
                identified_structures.extend(bunch)
            else:
                unassigned.append(chroma)
        else:
            unassigned.append(chroma)
    return identified_structures, unassigned
