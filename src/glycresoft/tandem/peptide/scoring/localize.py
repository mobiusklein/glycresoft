# -*- coding: utf-8 -*-
import itertools
import math
import array

from collections import defaultdict
from decimal import Decimal
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union, NamedTuple, Deque

import numpy as np
from scipy.special import comb

from glycopeptidepy import PeptideSequence, Modification, ModificationRule, IonSeries, PeptideFragment
from glycopeptidepy.utils.memoize import memoize
from glycopeptidepy.algorithm import PeptidoformGenerator, ModificationSiteAssignmentCombinator

from ms_deisotope.data_source import ProcessedScan
from ms_deisotope.peak_set import window_peak_set, DeconvolutedPeak
from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from glycresoft.structure import FragmentCachingGlycopeptide
from glycresoft.structure.probability import PredictorBase

from glycresoft.tandem.spectrum_match.spectrum_match import LocalizationScore, ScanMatchManagingMixin


MAX_MISSING_A_SCORE = 1e3


@memoize(100000000000)
def binomial_pmf(n, i, p):
    try:
        return comb(n, i, exact=True) * (p ** i) * ((1 - p) ** (n - i))
    except OverflowError:
        dn = Decimal(n)
        di = Decimal(i)
        dp = Decimal(p)
        x = math.factorial(dn) / (math.factorial(di) * math.factorial(dn - di))
        return float(x * dp ** di * ((1 - dp) ** (dn - di)))


class PeakWindow(SpanningMixin):
    peaks: List[DeconvolutedPeak]
    max_mass: float

    def __init__(self, peaks):
        self.peaks = list(peaks)
        self.max_mass = 0
        self.start = 0
        self.end = 0
        self._calculate()

    def __iter__(self):
        return iter(self.peaks)

    def __getitem__(self, i):
        return self.peaks[i]

    def __len__(self):
        return len(self.peaks)

    def _calculate(self):
        self.peaks.sort(key=lambda x: x.intensity, reverse=True)
        max_mass = 0
        min_mass = float('inf')
        for peak in self.peaks:
            if peak.neutral_mass > max_mass:
                max_mass = peak.neutral_mass
            elif peak.neutral_mass < min_mass:
                min_mass = peak.neutral_mass
        self.end = self.max_mass = max_mass
        self.start = min_mass

    def __repr__(self):
        template = "{self.__class__.__name__}({self.max_mass}, {size})"
        return template.format(self=self, size=len(self))



class BlindPeptidoformGenerator(PeptidoformGenerator):
    """A sub-class of the :class:`~.PeptidoformGenerator` type that ignores
    site-specificities.
    """
    def modification_sites_for(self, sequence, rules: List[ModificationRule], include_empty: bool=True):
        variable_sites = {
            mod: set(range(len(sequence))) for mod in rules}
        modification_sites = ModificationSiteAssignmentCombinator(
            variable_sites, include_empty=include_empty)
        return modification_sites


class ModificationAssignment(NamedTuple):
    site: int
    modification: Modification

    @property
    def is_ambiguous(self):
        try:
            return len(self.site) > 1
        except TypeError:
            return False

    def itersites(self):
        if self.is_ambiguous:
            for i in self.site:
                yield i
        else:
            yield self.site


class ProbableSitePair(NamedTuple):
    top_solution1: PeptideSequence
    top_solution2: PeptideSequence
    modifications: List[Modification]
    peak_depth: int


class AScoreSpec(NamedTuple):
    site: ModificationAssignment
    score: float
    site_determining_ions: np.array
    alternative_site_ions: np.array


class LocalizationCandidate(object):
    peptide: PeptideSequence
    modifications: List[Union[ModificationRule, Modification]]
    fragments: List[PeptideFragment]

    def __init__(self, peptide, modifications, fragments=None):
        self.peptide = peptide
        self.modifications = modifications
        self.fragments = fragments

    def __hash__(self):
        return hash(self.peptide)

    def __eq__(self, other):
        return self.peptide == other.peptide and self.modifications == other.modifications

    def make_solution(self, a_score: AScoreSpec, permutations: Optional[np.ndarray] = None) -> 'AScoreSolution':
        return AScoreSolution(self.peptide, a_score, self.modifications, permutations, self.fragments)

    def __repr__(self):
        template = "{self.__class__.__name__}({d})"

        def formatvalue(v):
            if isinstance(v, float):
                return "%0.4f" % v
            else:
                return str(v)
        d = [
            "%s=%s" % (k, formatvalue(v)) if v is not self else "(...)" for k, v in sorted(
                self.__dict__.items(), key=lambda x: x[0])
            if (not k.startswith("_") and not callable(v))
            and not (v is None) and k != "fragments"]

        return template.format(self=self, d=', '.join(d))


class AScoreSolution(LocalizationCandidate):
    a_score: List[AScoreSpec]
    permutations: np.ndarray

    def __init__(self, peptide: PeptideSequence, a_score, modifications: List[Modification], permutations,
                 fragments: Optional[List[PeptideFragment]]=None):
        super().__init__(peptide, modifications, fragments)
        self.a_score = a_score
        self.permutations = permutations


class PeptidoformPermuter(object):
    peptide: PeptideSequence
    modification_rule: ModificationRule
    modification_count: int
    respect_specificity: bool

    def __init__(self, peptide: PeptideSequence, modification_rule: ModificationRule,
                 modification_count: int=1, respect_specificity: bool=True):
        self.peptide = peptide
        self.modification_rule = modification_rule
        self.modification_count = modification_count
        self.respect_specificity = respect_specificity

    def find_existing(self, modification_rule: ModificationRule):
        '''Find existing modifications derived from this rule

        Parameters
        ----------
        modification_rule: :class:`~.ModificationRule`
            The modification rule to search for

        Returns
        -------
        indices: list
            The indices of :attr:`peptide` where modifications were found
        '''
        indices = []
        for i, position in enumerate(self.peptide):
            if position.modifications and modification_rule in position.modifications:
                indices.append(i)
        return indices

    def generate_base_peptides(self, modification_rule: ModificationRule) -> List[PeptideSequence]:
        """Generate peptides from :attr:`peptide` which have had combinations of
        modification sites removed.

        Parameters
        ----------
        modification_rule : :class:`~.ModificationRule`
            The modification rule to remove

        Returns
        -------
        list
        """
        existing_indices = self.find_existing(modification_rule)
        base_peptides: List[Union[PeptideSequence, FragmentCachingGlycopeptide]] = []
        for indices in itertools.combinations(existing_indices, self.modification_count):
            base_peptide = self.peptide.clone()
            for i in indices:
                base_peptide.drop_modification(i, modification_rule)
            base_peptides.append(base_peptide)
        # The target modification was not present, so the unaltered peptide must be the base
        if not base_peptides:
            base_peptides = [self.peptide.clone()]
        if isinstance(base_peptides[0], FragmentCachingGlycopeptide):
            for bp in base_peptides:
                bp.clear_caches()
        return base_peptides

    def generate_peptidoforms(self, modification_rule: ModificationRule, base_peptides: Optional[PeptideSequence]=None):
        if base_peptides is None:
            base_peptides = self.generate_base_peptides(modification_rule)
        if self.respect_specificity:
            PeptidoformGeneratorType = PeptidoformGenerator
        else:
            PeptidoformGeneratorType = BlindPeptidoformGenerator
        pepgen = PeptidoformGeneratorType(
            [], [modification_rule], self.modification_count)
        peptidoforms = defaultdict(set)
        for base_peptide in base_peptides:
            is_caching = isinstance(base_peptide, FragmentCachingGlycopeptide)
            mod_combos = pepgen.modification_sites_for(base_peptide, pepgen.variable_modifications)
            for mod_combo in mod_combos:
                if len(mod_combo) != self.modification_count:
                    continue
                mod_combo = [ModificationAssignment(*mc) for mc in mod_combo]
                peptidoform, _n_mods = pepgen.apply_variable_modifications(
                    base_peptide, mod_combo, None, None)
                if is_caching:
                    peptidoform.clear_caches()
                peptidoforms[peptidoform].update(tuple(mod_combo))
        return [LocalizationCandidate(peptide, sorted(mods), self._generate_fragments(peptide))
                for peptide, mods in peptidoforms.items()]


class LocalizationScorerBase(PeptidoformPermuter, ScanMatchManagingMixin):

    scan: ProcessedScan
    _fragment_cache: Dict[Union[PeptideSequence, Any], List[PeptideFragment]]

    def __init__(self, scan, peptide: PeptideSequence, modification_rule: ModificationRule,
                 modification_count: int = 1, respect_specificity: bool = True):
        super().__init__(peptide, modification_rule, modification_count, respect_specificity)
        self.scan = scan
        self._fragment_cache = {}

    def _generate_fragments(self, peptidoform: PeptideSequence) -> List[PeptideFragment]:
        key = str(peptidoform)
        if key in self._fragment_cache:
            return self._fragment_cache[key]

        ion_series = []

        if self.is_hcd():
            ion_series.append(IonSeries.b)
            ion_series.append(IonSeries.y)
        if self.is_exd():
            ion_series.append(IonSeries.c)
            ion_series.append(IonSeries.z)

        frags = itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                map(peptidoform.get_fragments, ion_series)
            )
        )
        frags = sorted(frags, key=lambda x: x.mass)
        self._fragment_cache[key] = frags
        return frags

    def clear_cache(self):
        self._fragment_cache.clear()


class AScoreEvaluator(LocalizationScorerBase):
    '''
    Calculate a localization statistic for given peptidoform and modification rule.

    The original probabilistic model is described in [1]. Implementation based heavily
    on the OpenMS implementation [2].

    References
    ----------
    [1] Beausoleil, S. a, Villén, J., Gerber, S. a, Rush, J., & Gygi, S. P. (2006).
        A probability-based approach for high-throughput protein phosphorylation analysis
        and site localization. Nature Biotechnology, 24(10), 1285–1292. https://doi.org/10.1038/nbt1240
    [2] Rost, H. L., Sachsenberg, T., Aiche, S., Bielow, C., Weisser, H., Aicheler, F., … Kohlbacher, O. (2016).
        OpenMS: a flexible open-source software platform for mass spectrometry data analysis. Nat Meth, 13(9),
        741–748. https://doi.org/10.1038/nmeth.3959
    '''
    peak_windows: List[PeakWindow]

    def __init__(self, scan, peptide, modification_rule, modification_count=1, respect_specificity=True):
        self._scan = None
        super().__init__(
            scan, peptide, modification_rule, modification_count, respect_specificity)
        self.peptidoforms = self.generate_peptidoforms(self.modification_rule)

    @property
    def scan(self):
        return self._scan

    @scan.setter
    def scan(self, value):
        self._scan = value
        if value is None:
            self.peak_windows = []
        else:
            self.peak_windows = list(map(PeakWindow, window_peak_set(value.deconvoluted_peak_set)))


    def match_ions(self, fragments: List[PeptideFragment], depth: int=10, error_tolerance: float=1e-5) -> Tuple[int, List[float]]:
        '''Match fragments against the windowed peak set at a given
        peak depth.

        Parameters
        ----------
        fragments: list
            A list of peptide fragments, sorted by mass
        depth: int
            The peak depth to search to, the `i`th most intense peak in
            each window
        error_tolerance: float
            The PPM error tolerance to use when matching peaks.

        Returns
        -------
        int:
            The number of fragments matched
        '''
        n = 0
        window_i = 0
        window_n = len(self.peak_windows)
        current_window = self.peak_windows[window_i]
        intensities: array.ArrayType[float] = array.array('f')
        for frag in fragments:
            while not current_window or (frag.mass >= (current_window.max_mass + 1)):
                window_i += 1
                if window_i == window_n:
                    return n
                current_window = self.peak_windows[window_i]
            for peak in current_window[:depth]:
                if abs(peak.neutral_mass - frag.mass) / frag.mass < error_tolerance:
                    intensities.append(peak.intensity)
                    n += 1
        return n, np.frombuffer(intensities, dtype=np.float32)

    def permutation_score(self, peptidoform: LocalizationCandidate, error_tolerance: float=1e-5) -> Tuple[np.ndarray,
                                                                                                    List[List[float]]]:
        '''Calculate the binomial statistic for this peptidoform
        using the top 1 to 10 peaks.

        Parameters
        ----------
        peptidoform: :class:`~.PeptideSequence`
            The peptidoform to score
        error_tolerance: float
            The PPM error tolerance to use when matching peaks.

        Returns
        -------
        :class:`numpy.ndarray`:
            The binomial score at peak depth `i + 1`

        See Also
        --------
        :meth:`_score_at_window_depth`
        :meth:`match_ions`
        '''
        fragments = peptidoform.fragments
        N = len(fragments)
        site_scores = np.zeros(10)
        intensities = [None for i in range(10)]
        for i in range(1, 11):
            site_scores[i - 1], intensities[i - 1] = self._score_at_window_depth(
                fragments, N, i, error_tolerance)
        return site_scores, intensities

    def _score_at_window_depth(self, fragments: List[PeptideFragment], N: int, i: int, error_tolerance: float=1e-5) -> Tuple[float, List[float]]:
        '''Score a fragment collection at a given peak depth, and
        calculate the binomial score based upon the probability mass
        function.

        Parameters
        ----------
        fragments: list
            A list of peptide fragments, sorted by mass
        N: int
            The maximum number of theoretical fragments
        i: int
            The peak depth to search through
        error_tolerance: float
            The PPM error tolerance to use when matching peaks.

        Returns
        -------
        score: float
        intensities: list[float]
        '''
        n, intensities = self.match_ions(fragments, i, error_tolerance=error_tolerance)
        p = i / 100.0
        # If a fragment matches twice, this count can exceed the theoretical maximum.
        if n > N:
            n = N
        cumulative_score = binomial_pmf(N, n, p)
        if cumulative_score == 0.0:
            return MAX_MISSING_A_SCORE, intensities
        return abs(-10.0 * math.log10(cumulative_score)), intensities

    def rank_permutations(self, permutation_scores):
        """Rank generated peptidoforms by weighted sum of permutation scores

        Parameters
        ----------
        permutation_scores : :class:`list` of :class:`list` of :class:`float`
            The raw output of :meth:`permutation_score` for each peak depth for
            each peptidoform.

        Returns
        -------
        :class:`list`
            A list of :class:`tuple` instances of (weighted score, peptidoform index)
        """
        ranking = []
        for i, perm_scores in enumerate(permutation_scores):
            ranking.append((self._weighted_score(perm_scores), i))
        ranking.sort(reverse=True)
        return ranking

    # Taken directly from reference [1]
    _weight_vector = np.array([
        0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, .25, .25
    ])

    def _weighted_score(self, scores):
        """Calculate the weighted sum score over the peak-depth permuted
        binomial score vector.

        Parameters
        ----------
        scores : :class:`list`
            The binomial score at each peak depth

        Returns
        -------
        float
        """
        return self._weight_vector.dot(scores) / 10.0

    def score_solutions(self, error_tolerance: float=1e-5, peptidoforms: Optional[List[LocalizationCandidate]]=None) -> List[AScoreSolution]:
        if peptidoforms is None:
            peptidoforms = self.peptidoforms
        scores, _intensities = zip(*[self.permutation_score(candidate, error_tolerance=error_tolerance)
                  for candidate in peptidoforms])
        ranked = self.rank_permutations(scores)
        solutions = [peptidoforms[i].make_solution(score, scores[i])
                     for score, i in ranked]
        return solutions

    def score_localizations(self, solutions: List[AScoreSolution], error_tolerance=1e-5):
        """Find pairs of sequence solutions which differ in the localization
        of individual modifications w.r.t. to the best match to compute the final
        per-modification A-score.

        The first solution in `solutions` is the highest ranked solution, and subsequent
        solutions are searched for the next case where one of the modification of interest
        is located at a different position, forming a pair for that modification site by
        :meth:`find_highest_scoring_permutations`. For each pair, the sequences are re-scored
        using only site-determining ions, and the difference between those scores is the A-score
        for that pair's modification site, as calculated by :meth:`calculate_delta`.

        If there are no alternative sites for a given modification, that modification will be
        given the A-score given by :const:`MAX_MISSING_A_SCORE`. If there is another
        localization which scores equally well, the A-score will be 0 by definition of
        the delta step.

        Parameters
        ----------
        solutions : list
            The list of :class:`AScoreSolution` objects, ranked by total score
        error_tolerance : float, optional
            The mass error tolerance to use when matching site-determining ions (the default is 1e-5)

        Returns
        -------
        :class:`AScoreSolution`
        """
        delta_scores = []
        pairs = self.find_highest_scoring_permutations(solutions)
        top_solution = solutions[0]
        if not pairs:
            for mod in top_solution.modifications:
                delta_scores.append(
                    AScoreSpec(
                        mod,
                        MAX_MISSING_A_SCORE,
                        np.array([], dtype=np.float32),
                        np.array([], dtype=np.float32)
                    )
                )
            top_solution.a_score = delta_scores
            return top_solution
        for pair in pairs:
            delta_score, intensities1, intensities2 = self.calculate_delta(pair, error_tolerance=error_tolerance)
            pair.top_solution1.a_score = delta_score
            delta_scores.append(
                AScoreSpec(
                    pair.modifications,
                    delta_score,
                    intensities1,
                    intensities2
                )
            )
        top_solution.a_score = delta_scores
        return top_solution

    def score(self, error_tolerance: float=1e-5) -> AScoreSolution:
        solutions = self.score_solutions(error_tolerance)
        top_solution = self.score_localizations(solutions, error_tolerance)
        return top_solution

    def find_highest_scoring_permutations(self, solutions: List[AScoreSolution],
                                          best_solution: Optional[AScoreSolution]=None,
                                          offset: Optional[int] = None) -> List[ProbableSitePair]:
        if best_solution is None:
            best_solution = solutions[0]
            offset = 1
        else:
            if offset is None:
                for i, sol in enumerate(solutions):
                    if sol == solutions:
                        offset = i + 1
                        break
                else:
                    raise ValueError("Best solution %r not in solution set")
        permutation_pairs = []
        # for each modification under permutation, find the next best solution which
        # does not have this modification in its set of permuted modifications, and
        # package the pair into a :class:`ProbableSitePair`.
        for site in best_solution.modifications:
            for alt_solution in solutions[offset:]:
                if site not in alt_solution.modifications:
                    peak_depth = np.argmax(best_solution.permutations - alt_solution.permutations) + 1
                    permutation_pairs.append(ProbableSitePair(best_solution, alt_solution, site, peak_depth))
                    break
        return permutation_pairs

    def site_determining_ions(self, solutions: List[LocalizationCandidate]) -> List[List[PeptideFragment]]:
        frag_sets = [set(sol.fragments) for sol in solutions]
        counts = defaultdict(int)
        for frag_set in frag_sets:
            for frag in frag_set:
                counts[frag] += 1
        site_determining = []
        for frag_set in frag_sets:
            acc = []
            for frag in frag_set:
                if counts[frag] == 1:
                    acc.append(frag)
            site_determining.append(sorted(acc, key=lambda x: x.mass))
        return site_determining

    def calculate_delta(self, candidate_pair: ProbableSitePair, error_tolerance=1e-5):
        if candidate_pair.top_solution1 == candidate_pair.top_solution2:
            return 0.0
        site_frags = self.site_determining_ions(
            [candidate_pair.top_solution1, candidate_pair.top_solution2])
        site_frags1, site_frags2 = site_frags[0], site_frags[1]
        N1 = len(site_frags1)
        N2 = len(site_frags2)
        peak_depth = candidate_pair.peak_depth
        score1, intensities1 = self._score_at_window_depth(
            site_frags1, N1, peak_depth, error_tolerance=error_tolerance)
        score2, intensities2 = self._score_at_window_depth(
            site_frags2, N2, peak_depth, error_tolerance=error_tolerance)
        return score1 - score2, intensities1, intensities2


class _PeakSet:
    peaks: Dict[int, float]
    total: float

    def __init__(self, peaks=None, total=None):
        if not peaks:
            peaks = {}
            total = 0
        self.peaks = peaks
        self.total = total

    def add(self, peak: DeconvolutedPeak):
        self.peaks[peak.index.neutral_mass] = peak.intensity
        self.total += peak.intensity

    def shared_intensity(self, other: '_PeakSet'):
        common = self.peaks.keys() & other.peaks.keys()
        acc = 0
        for k in common:
            acc += self.peaks[k]
        return acc

    def count_partitions(self, other: '_PeakSet'):
        common = self.peaks.keys() & other.peaks.keys()
        m = len(common)
        return len(self.peaks) - m, len(other.peaks) - m, m



class _PTMProphetMatch(NamedTuple):
    occupied: _PeakSet
    unoccupied: _PeakSet
    normalizer: float

    def o_score(self) -> float:
        common = self.occupied.shared_intensity(self.unoccupied)
        occupied_diff = (self.occupied.total - common) / self.normalizer
        unoccupied_diff = (self.unoccupied.total - common) / self.normalizer
        return occupied_diff / (occupied_diff + unoccupied_diff + 1e-6)

    def m_score(self) -> float:
        m_occupied, m_unoccupied, _shared = self.occupied.count_partitions(self.unoccupied)
        return m_occupied / (m_occupied + m_unoccupied + 1e-6)

    def to_solution(self, occupied_index: int, unoccupied_index: int) -> 'PTMProphetSolution':
        return PTMProphetSolution(self.o_score(), self.m_score(), occupied_index, unoccupied_index)


class PTMProphetSolution(NamedTuple):
    o_score: float
    m_score: float
    occupied_peptidoform_index: int
    unoccupied_peptidoform_index: int


class ScoredIsoform(NamedTuple):
    isoform: LocalizationCandidate
    localizations: List[LocalizationScore]
    score: float


class PTMProphetEvaluator(LocalizationScorerBase):
    raw_matches: List[_PeakSet]
    occupied_site_index: DefaultDict[int, Set[int]]
    solution_for_site: Dict[int, PTMProphetSolution]

    def __init__(self, scan, peptide, modification_rule, modification_count=1, respect_specificity=True):
        super().__init__(
            scan, peptide, modification_rule, modification_count, respect_specificity)
        self.peptidoforms = self.generate_peptidoforms(self.modification_rule)
        self.raw_matches = []
        self.occupied_site_index = DefaultDict(set)
        self.solution_for_site = {}

    def get_normalizer(self) -> float:
        normalizer = float('inf')
        for peak in self.scan.deconvoluted_peak_set:
            if peak.intensity < normalizer:
                normalizer = peak.intensity
        return normalizer

    def score_arrangements(self, error_tolerance: float=1e-5):
        raw_matches = []
        occupied_site_index = DefaultDict[int, Set[int]](set)

        normalizer = self.get_normalizer()

        for i, peptidoform in enumerate(self.peptidoforms):
            acc = _PeakSet()
            for fragment in peptidoform.fragments:
                for peak in self.scan.deconvoluted_peak_set.all_peaks_for(fragment.mass, error_tolerance):
                    acc.add(peak)
            raw_matches.append(acc)
            for j, pos in enumerate(peptidoform.peptide):
                if pos.has_modification(self.modification_rule):
                    occupied_site_index[j].add(i)

        self.raw_matches = raw_matches
        self.occupied_site_index = occupied_site_index

        for site, occupied_in in self.occupied_site_index.items():
            best_occupied_match = _PeakSet()
            best_occupied_index = None
            best_unoccupied_match = _PeakSet()
            best_unoccupied_index = None
            for i in range(len(self.peptidoforms)):
                if i in occupied_in:
                    if best_occupied_match.total < self.raw_matches[i].total:
                        best_occupied_match = self.raw_matches[i]
                        best_occupied_index = i
                else:
                    if best_unoccupied_match.total < self.raw_matches[i].total:
                        best_unoccupied_match = self.raw_matches[i]
                        best_unoccupied_index = i

            if best_occupied_index is None:
                continue
            self.solution_for_site[site] = _PTMProphetMatch(
                best_occupied_match, best_unoccupied_match, normalizer).to_solution(
                    best_occupied_index, best_unoccupied_index)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.scan}, {self.peptide}, {self.modification_rule}, {self.solution_for_site})"

    def entropy(self, prophet: Optional[PredictorBase]) -> float:
        m = self.modification_count
        s = len(self.solution_for_site)
        log_base = np.log(s / m)
        acc = 0.0
        for sol in self.solution_for_site.values():
            prob = prophet.predict([sol.o_score, sol.m_score])[0]
            acc += prob * np.log(prob) / log_base
        return acc

    def site_probabilities(self, prophet: Optional[PredictorBase]) -> Dict[int, float]:
        position_probs = {}
        acc = 0.0
        for position, sol in self.solution_for_site.items():
            if prophet is not None:
                prob = prophet.predict([sol.o_score, sol.m_score])[0]
            else:
                prob = sol.o_score
            position_probs[position] = prob
            acc += prob
        acc /= self.modification_count
        factor = 1 / acc if acc > 0 else 0
        return {k: min(v * factor, 1.0) for k, v in position_probs.items()}

    def score_isoforms(self, prophet: Optional[PredictorBase]) -> List[ScoredIsoform]:
        weights: List[float] = []
        localizations: List[List[LocalizationScore]] = []
        sites = self.site_probabilities(prophet)

        for candidate in self.peptidoforms:
            acc = 0.0
            parts: List[LocalizationScore] = []
            for mod_a in candidate.modifications:
                try:
                    score = sites[mod_a.site]
                except KeyError:
                    if not isinstance(mod_a.site, int):
                        # TODO: This needs to handle terminal modifications
                        # which requires a change to the core algorithm.
                        # Ignore them for now.
                        score = 0.0
                    else:
                        # The position was never observed. Ignore it.
                        # May happen when sites are not restricted during
                        # peptidoform generation here but not during the original
                        # search.
                        score = 0.0
                acc += score
                parts.append(
                    LocalizationScore(
                        mod_a.site,
                        mod_a.modification.name,
                        score
                    )
                )
            localizations.append(parts)
            weights.append(acc)

        scored_isoforms = []
        for candidate, weight, loc_scores in zip(self.peptidoforms, weights, localizations):
            scored_isoforms.append(ScoredIsoform(
                candidate, loc_scores, weight))
        return scored_isoforms
