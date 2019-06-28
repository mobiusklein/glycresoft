# -*- coding: utf-8 -*-
import itertools
import math

from collections import namedtuple, defaultdict
from decimal import Decimal

import numpy as np
from scipy.special import comb

from glycopeptidepy.utils.memoize import memoize
from glycopeptidepy.algorithm import PeptidoformGenerator, ModificationSiteAssignmentCombinator

from ms_deisotope.peak_set import window_peak_set


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


class PeakWindow(object):
    def __init__(self, peaks):
        self.peaks = list(peaks)
        self.max_mass = 0
        self._calculate()

    def __iter__(self):
        return iter(self.peaks)

    def __getitem__(self, i):
        return self.peaks[i]

    def __len__(self):
        return len(self.peaks)

    def _calculate(self):
        self.peaks.sort(key=lambda x: x.intensity, reverse=True)
        self.max_mass = 0
        for peak in self.peaks:
            if peak.neutral_mass > self.max_mass:
                self.max_mass = peak.neutral_mass

    def __repr__(self):
        template = "{self.__class__.__name__}({self.max_mass}, {size})"
        return template.format(self=self, size=len(self))



class BlindPeptidoformGenerator(PeptidoformGenerator):
    """A sub-class of the :class:`~.PeptidoformGenerator` type that ignores
    site-specificities.
    """
    def modification_sites(self, sequence):
        variable_sites = {
            mod.name: set(range(len(sequence))) for mod in self.variable_modifications}
        modification_sites = ModificationSiteAssignmentCombinator(
            variable_sites)
        return modification_sites


ProbableSitePair = namedtuple("ProbableSitePair", ['peptide1', 'peptide2', 'modifications', 'peak_depth'])
_ModificationAssignment = namedtuple("ModificationAssignment", ["site", "modification"])


class ModificationAssignment(_ModificationAssignment):
    __slots__ = []

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


class AScoreCandidate(object):
    def __init__(self, peptide, modifications, fragments=None):
        self.peptide = peptide
        self.modifications = modifications
        self.fragments = fragments

    def __hash__(self):
        return hash(self.peptide)

    def __eq__(self, other):
        return self.peptide == other.peptide and self.modifications == other.modifications

    def make_solution(self, a_score, permutations=None):
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


class AScoreSolution(AScoreCandidate):
    def __init__(self, peptide, a_score, modifications, permutations, fragments=None):
        super(AScoreSolution, self).__init__(peptide, modifications, fragments)
        self.a_score = a_score
        self.permutations = permutations


class PeptidoformPermuter(object):
    def __init__(self, peptide, modification_rule, modification_count=1, respect_specificity=True):
        self.peptide = peptide
        self.modification_rule = modification_rule
        self.modification_count = modification_count
        self.respect_specificity = respect_specificity

    def find_existing(self, modification_rule):
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
            if modification_rule in position.modifications:
                indices.append(i)
        return indices

    def generate_base_peptides(self, modification_rule):
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
        base_peptides = []
        for indices in itertools.combinations(existing_indices, self.modification_count):
            base_peptide = self.peptide.clone()
            for i in indices:
                base_peptide.drop_modification(i, modification_rule)
            base_peptides.append(base_peptide)
        # The target modification was not present, so the unaltered peptide must be the base
        if not base_peptides:
            base_peptides = [self.peptide.clone()]
        return base_peptides

    def generate_peptidoforms(self, modification_rule, base_peptides=None):
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
            mod_combos = pepgen.modification_sites(base_peptide)
            for mod_combo in mod_combos:
                if len(mod_combo) != self.modification_count:
                    continue
                mod_combo = [ModificationAssignment(*mc) for mc in mod_combo]
                peptidoform, _n_mods = pepgen.apply_variable_modifications(
                    base_peptide, mod_combo, None, None)
                peptidoforms[peptidoform].update(tuple(mod_combo))
        return [AScoreCandidate(peptide, sorted(mods), self._generate_fragments(peptide))
                for peptide, mods in peptidoforms.items()]


class AScoreEvaluator(PeptidoformPermuter):
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
    def __init__(self, scan, peptide, modification_rule, modification_count=1, respect_specificity=True):
        self._scan = None
        self.peak_windows = []

        PeptidoformPermuter.__init__(
            self, peptide, modification_rule, modification_count, respect_specificity)
        self.scan = scan
        self.peptidoforms = self.generate_peptidoforms(self.modification_rule)
        self._fragment_cache = {}

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

    def _generate_fragments(self, peptidoform):
        frags = itertools.chain.from_iterable(
            itertools.chain(
                peptidoform.get_fragments("y"),
                peptidoform.get_fragments("b")))
        frags = list(frags)
        frags.sort(key=lambda x: x.mass)
        return frags

    def match_ions(self, fragments, depth=10, error_tolerance=1e-5):
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
        for frag in fragments:
            while not current_window or (frag.mass >= (current_window.max_mass + 1)):
                window_i += 1
                if window_i == window_n:
                    return n
                current_window = self.peak_windows[window_i]
            for peak in current_window[:depth]:
                if abs(peak.neutral_mass - frag.mass) / frag.mass < error_tolerance:
                    n += 1
        return n

    def permutation_score(self, peptidoform, error_tolerance=1e-5):
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
        for i in range(1, 11):
            site_scores[i - 1] = self._score_at_window_depth(
                fragments, N, i, error_tolerance)
        return site_scores

    def _score_at_window_depth(self, fragments, N, i, error_tolerance=1e-5):
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
        float
        '''
        n = self.match_ions(fragments, i, error_tolerance=error_tolerance)
        p = i / 100.0
        # If a fragment matches twice, this count can exceed the theoretical maximum.
        if n > N:
            n = N
        cumulative_score = binomial_pmf(N, n, p)
        if cumulative_score == 0.0:
            return 1e3
        return (abs(-10.0 * math.log10(cumulative_score)))

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

    def score_solutions(self, error_tolerance=1e-5, peptidoforms=None):
        if peptidoforms is None:
            peptidoforms = self.peptidoforms
        scores = [self.permutation_score(candidate, error_tolerance=error_tolerance)
                  for candidate in peptidoforms]
        ranked = self.rank_permutations(scores)
        solutions = [peptidoforms[i].make_solution(score, scores[i])
                     for score, i in ranked]
        return solutions

    def score_localizations(self, solutions, error_tolerance=1e-5):
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
        peptide = solutions[0]
        if not pairs:
            for mod in peptide.modifications:
                delta_scores.append((mod, MAX_MISSING_A_SCORE))
            peptide.a_score = delta_scores
            return peptide
        for pair in pairs:
            delta_score = self.calculate_delta(pair, error_tolerance=error_tolerance)
            pair.peptide1.a_score = delta_score
            delta_scores.append((pair.modifications, delta_score))
        peptide.a_score = delta_scores
        return peptide

    def score(self, error_tolerance=1e-5):
        solutions = self.score_solutions(error_tolerance)
        peptide = self.score_localizations(solutions, error_tolerance)
        return peptide

    def find_highest_scoring_permutations(self, solutions, best_solution=None, offset=None):
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

    def site_determining_ions(self, solutions):
        frag_sets = [set(sol.fragments) for sol in solutions]
        common = set.intersection(*frag_sets)
        n = len(solutions)
        site_determining = []
        for i, _solution in enumerate(solutions):
            cur_frags = frag_sets[i]
            if i == n - 1:
                diff = cur_frags - common
                site_determining.append(sorted(diff, key=lambda x: x.mass))
            else:
                diff = cur_frags - common - frag_sets[i + 1]
                site_determining.append(sorted(diff, key=lambda x: x.mass))
        return site_determining

    def calculate_delta(self, candidate_pair, error_tolerance=1e-5):
        if candidate_pair.peptide1 == candidate_pair.peptide2:
            return 0.0
        site_frags = self.site_determining_ions(
            [candidate_pair.peptide1, candidate_pair.peptide2])
        site_frags1, site_frags2 = site_frags[0], site_frags[1]
        N1 = len(site_frags1)
        N2 = len(site_frags2)
        peak_depth = candidate_pair.peak_depth
        P1 = self._score_at_window_depth(
            site_frags1, N1, peak_depth, error_tolerance=error_tolerance)
        P2 = self._score_at_window_depth(
            site_frags2, N2, peak_depth, error_tolerance=error_tolerance)
        return P1 - P2
