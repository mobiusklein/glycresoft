from ms_deisotope import DeconvolutedPeakSet, isotopic_shift

from glycan_profiling.structure import (
    ScanWrapperBase)

from glycan_profiling.chromatogram_tree import Unmodified

from .ref import TargetReference, SpectrumReference


neutron_offset = isotopic_shift()


class SpectrumMatchBase(ScanWrapperBase):
    __slots__ = ['scan', 'target', "_mass_shift"]

    def __init__(self, scan, target, mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        self.scan = scan
        self.target = target
        self._mass_shift = None
        self.mass_shift = mass_shift

    @property
    def mass_shift(self):
        return self._mass_shift

    @mass_shift.setter
    def mass_shift(self, value):
        self._mass_shift = value

    @staticmethod
    def threshold_peaks(deconvoluted_peak_set, threshold_fn=lambda peak: True):
        deconvoluted_peak_set = DeconvolutedPeakSet([
            p for p in deconvoluted_peak_set
            if threshold_fn(p)
        ])
        deconvoluted_peak_set._reindex()
        return deconvoluted_peak_set

    def precursor_mass_accuracy(self, offset=0):
        observed = self.precursor_ion_mass
        theoretical = self.target.total_composition().mass + (
            offset * neutron_offset) + self.mass_shift.mass
        return (observed - theoretical) / theoretical

    def determine_precursor_offset(self, probing_range=3):
        best_offset = 0
        best_error = float('inf')
        for i in range(probing_range + 1):
            error = abs(self.precursor_mass_accuracy(i))
            if error < best_error:
                best_error = error
                best_offset = i
        return best_offset

    def __reduce__(self):
        return self.__class__, (self.scan, self.target)

    def get_top_solutions(self):
        return [self]

    def __eq__(self, other):
        try:
            target_id = self.target.id
        except AttributeError:
            target_id = None
        try:
            other_target_id = self.target.id
        except AttributeError:
            other_target_id = None
        return (self.scan == other.scan) and (self.target == other.target) and (
            target_id == other_target_id)

    def __hash__(self):
        try:
            target_id = self.target.id
        except AttributeError:
            target_id = None
        return hash((self.scan.id, self.target, target_id))


class SpectrumMatcherBase(SpectrumMatchBase):
    __slots__ = ["spectrum", "_score"]

    def __init__(self, scan, target, mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        self.scan = scan
        self.spectrum = scan.deconvoluted_peak_set
        self.target = target
        self._score = 0
        self._mass_shift = None
        self.mass_shift = mass_shift

    @property
    def score(self):
        return self._score

    def match(self, *args, **kwargs):
        raise NotImplementedError()

    def calculate_score(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, scan, target, *args, **kwargs):
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        inst = cls(scan, target, mass_shift=mass_shift)
        inst.match(*args, **kwargs)
        inst.calculate_score(*args, **kwargs)
        return inst

    def __getstate__(self):
        return (self.score,)

    def __setstate__(self, state):
        self.score = state[0]

    def __reduce__(self):
        return self.__class__, (self.scan, self.target,)

    @staticmethod
    def load_peaks(scan):
        try:
            return scan.convert(fitted=False, deconvoluted=True)
        except AttributeError:
            return scan

    def __repr__(self):
        return "{self.__class__.__name__}({self.spectrum}, {self.target}, {self.score})".format(
            self=self)

    def plot(self, ax=None, **kwargs):
        from glycan_profiling.plotting import spectral_annotation
        art = spectral_annotation.SpectrumMatchAnnotator(self, ax=ax)
        art.draw(**kwargs)
        return art


class DeconvolutingSpectrumMatcherBase(SpectrumMatcherBase):

    @staticmethod
    def load_peaks(scan):
        try:
            return scan.convert(fitted=True, deconvoluted=False)
        except AttributeError:
            return scan

    def __init__(self, scan, target):
        super(DeconvolutingSpectrumMatcherBase, self).__init__(scan, target)
        self.spectrum = scan.peak_set


class SpectrumMatch(SpectrumMatchBase):

    __slots__ = ['score', 'best_match', 'data_bundle', "q_value", 'id']

    def __init__(self, scan, target, score, best_match=False, data_bundle=None,
                 q_value=None, id=None, mass_shift=None):
        if data_bundle is None:
            data_bundle = dict()

        super(SpectrumMatch, self).__init__(scan, target, mass_shift)

        self.score = score
        self.best_match = best_match
        self.data_bundle = data_bundle
        self.q_value = q_value
        self.id = id

    def clear_caches(self):
        try:
            self.target.clear_caches()
        except AttributeError:
            pass

    def __reduce__(self):
        return self.__class__, (self.scan, self.target, self.score, self.best_match,
                                self.data_bundle, self.q_value, self.id, self.mass_shift)

    def evaluate(self, scorer_type, *args, **kwargs):
        if isinstance(self.scan, SpectrumReference):
            raise TypeError("Cannot evaluate a spectrum reference")
        elif isinstance(self.target, TargetReference):
            raise TypeError("Cannot evaluate a target reference")
        return scorer_type.evaluate(self.scan, self.target, *args, **kwargs)

    def __repr__(self):
        return "SpectrumMatch(%s, %s, %0.4f)" % (self.scan, self.target, self.score)

    @classmethod
    def from_match_solution(cls, match):
        return cls(match.scan, match.target, match.score, mass_shift=match.mass_shift)


class SpectrumSolutionSet(ScanWrapperBase):

    def __init__(self, scan, solutions):
        self.scan = scan
        self.solutions = solutions
        self.mean = self._score_mean()
        self.variance = self._score_variance()
        self._is_simplified = False
        self._is_top_only = False
        self._target_map = None

    def _invalidate(self):
        self._target_map = None

    @property
    def score(self):
        return self.best_solution().score

    def _make_target_map(self):
        self._target_map = {
            sol.target: sol for sol in self
        }

    def solution_for(self, target):
        if self._target_map is None:
            self._make_target_map()
        return self._target_map[target]

    def precursor_mass_accuracy(self):
        return self.best_solution().precursor_mass_accuracy()

    def best_solution(self):
        return self.solutions[0]

    def _score_mean(self):
        i = 0
        total = 0
        for match in self:
            total += match.score
            i += 1.
        if i > 0:
            return total / i
        else:
            return 0

    def _score_variance(self):
        total = 0.
        i = 0.
        mean = self.mean
        for match in self:
            total += (match.score - mean) ** 2
            i += 1.
        if i < 3:
            return 0
        return total / (i - 2.)

    def __repr__(self):
        if len(self) == 0:
            return "SpectrumSolutionSet(%s, [])" % (self.scan,)
        return "SpectrumSolutionSet(%s, %s, %f)" % (
            self.scan, self.best_solution().target, self.best_solution().score)

    def __getitem__(self, i):
        return self.solutions[i]

    def __iter__(self):
        return iter(self.solutions)

    def __len__(self):
        return len(self.solutions)

    def threshold(self):
        if len(self) == 0:
            return self
        thresh = min(self.mean / 2., self.score / 2.)
        self.solutions = [
            x for x in self if x.score >= thresh
        ]
        self._invalidate()
        return self

    def simplify(self):
        if self._is_simplified:
            return
        self.scan = SpectrumReference(
            self.scan.id, self.scan.precursor_information)
        solutions = []
        best_score = self.best_solution().score
        for sol in self.solutions:
            sm = SpectrumMatch.from_match_solution(sol)
            if abs(sm.score - best_score) < 1e-6:
                sm.best_match = True
            sm.scan = self.scan
            solutions.append(sm)
        self.solutions = solutions
        self._is_simplified = True
        self._invalidate()

    def get_top_solutions(self):
        score = self.best_solution().score
        return [x for x in self.solutions if abs(x.score - score) < 1e-6]

    def select_top(self):
        if self._is_top_only:
            return
        self.solutions = self.get_top_solutions()
        self._is_top_only = True
        self._invalidate()
