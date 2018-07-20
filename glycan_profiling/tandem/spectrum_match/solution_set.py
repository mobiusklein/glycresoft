from .spectrum_match import SpectrumMatch, SpectrumReference, ScanWrapperBase


class SpectrumSolutionSet(ScanWrapperBase):
    spectrum_match_type = SpectrumMatch

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
            sm = self.spectrum_match_type.from_match_solution(sol)
            if abs(sm.score - best_score) < 1e-6:
                sm.best_match = True
            sm.scan = self.scan
            solutions.append(sm)
        self.solutions = solutions
        self._is_simplified = True
        self._invalidate()

    def get_top_solutions(self, d=3):
        best_score = self.best_solution().score
        return [x for x in self.solutions if (best_score - x.score) < d]

    def select_top(self, method=None):
        if method is None:
            method = default_selection_method
        if self._is_top_only:
            return
        best_solution = self.best_solution()
        self.solutions = method(self)
        if len(self) == 0:
            self.solutions = [best_solution]
        self._is_top_only = True
        self._invalidate()

    def merge(self, other):
        self.solutions.extend(other)
        self.solutions.sort(key=lambda x: x.score, reverse=True)
        if self._is_top_only:
            self._is_top_only = False
            self.select_top()


class SpectrumMatchRetentionStrategyBase(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter_matches(self, solution_set):
        raise NotImplementedError()

    def __call__(self, solution_set):
        return self.filter_matches(solution_set)

    def __repr__(self):
        return "{self.__class__.__name__}({self.threshold})".format(self=self)


class MinimumScoreRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    def filter_matches(self, solution_set):
        retain = []
        for match in solution_set:
            if match.score > self.threshold:
                retain.append(match)
        return retain


class MaximumSolutionCountRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    def filter_matches(self, solution_set):
        return solution_set[:self.threshold]


class TopScoringSolutionsRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    def filter_matches(self, solution_set):
        if len(solution_set) == 0:
            return solution_set
        best_score = solution_set[0].score
        retain = []
        for solution in solution_set:
            if solution.score - best_score < self.threshold:
                retain.append(solution)
        return retain


class SpectrumMatchRetentionMethod(SpectrumMatchRetentionStrategyBase):
    def __init__(self, strategies=None):
        if strategies is None:
            strategies = []
        self.strategies = strategies

    def filter_matches(self, solution_set):
        retained = list(solution_set)
        for strategy in self.strategies:
            retained = strategy(retained)
        return retained


default_selection_method = SpectrumMatchRetentionMethod([
    MinimumScoreRetentionStrategy(4.),
    TopScoringSolutionsRetentionStrategy(3.),
    MaximumSolutionCountRetentionStrategy(100)
])
