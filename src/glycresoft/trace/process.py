from glycresoft.task import TaskBase

from .match import (
    GlycanChromatogramMatcher,
    GlycopeptideChromatogramMatcher,
    NonSplittingChromatogramMatcher)

from .evaluate import (
    ChromatogramEvaluator,
    LogitSumChromatogramEvaluator,
    LaplacianRegularizedChromatogramEvaluator)


class ChromatogramProcessor(TaskBase):
    matcher_type = GlycanChromatogramMatcher

    def __init__(self, chromatograms, database, mass_shifts=None, mass_error_tolerance=1e-5,
                 scoring_model=None, smooth_overlap_rt=True, acceptance_threshold=0.4,
                 delta_rt=0.25, peak_loader=None):
        if mass_shifts is None:
            mass_shifts = []
        self._chromatograms = chromatograms
        self.database = database
        self.mass_shifts = mass_shifts
        self.mass_error_tolerance = mass_error_tolerance
        self.peak_loader = peak_loader

        self.scoring_model = scoring_model

        self.smooth_overlap_rt = smooth_overlap_rt
        self.acceptance_threshold = acceptance_threshold
        self.delta_rt = delta_rt

        self.solutions = None
        self.accepted_solutions = None

    def make_matcher(self):
        matcher = self.matcher_type(self.database)
        return matcher

    def _match_compositions(self):
        matcher = self.make_matcher()
        matches = matcher.process(
            self._chromatograms, self.mass_shifts, self.mass_error_tolerance,
            delta_rt=(self.delta_rt * 4 if self.smooth_overlap_rt else 0))
        return matches

    def make_evaluator(self):
        evaluator = ChromatogramEvaluator(self.scoring_model)
        return evaluator

    def match_compositions(self):
        self.log("Begin Matching Chromatograms")
        matches = self._match_compositions()
        self.log("End Matching Chromatograms")
        self.log("%d Chromatogram Candidates Found" % (len(matches),))
        return matches

    def evaluate_chromatograms(self, matches):
        self.log("Begin Evaluating Chromatograms")
        self.evaluator = self.make_evaluator()
        self.evaluator.configure({
            "peak_loader": self.peak_loader,
            "mass_shifts": self.mass_shifts,
            "delta_rt": self.delta_rt,
            "mass_error_tolerance": self.mass_error_tolerance,
            "matches": matches
        })
        self.solutions = self.evaluator.score(
            matches, smooth_overlap_rt=self.smooth_overlap_rt,
            mass_shifts=self.mass_shifts, delta_rt=self.delta_rt)
        self.accepted_solutions = self.evaluator.acceptance_filter(self.solutions)
        self.log("End Evaluating Chromatograms")
        return self.accepted_solutions

    def run(self):
        matches = self.match_compositions()
        result = self.evaluate_chromatograms(matches)
        return result

    def __iter__(self):
        if self.accepted_solutions is None:
            self.run()
        return iter(self.accepted_solutions)


class LogitSumChromatogramProcessor(ChromatogramProcessor):
    def make_evaluator(self):
        evaluator = LogitSumChromatogramEvaluator(self.scoring_model)
        return evaluator


class LaplacianRegularizedChromatogramProcessor(LogitSumChromatogramProcessor):
    GRID_SEARCH = 'grid'

    def __init__(self, chromatograms, database, network=None, mass_shifts=None, mass_error_tolerance=1e-5,
                 scoring_model=None, smooth_overlap_rt=True, acceptance_threshold=0.4,
                 delta_rt=0.25, peak_loader=None, smoothing_factor=0.2, grid_smoothing_max=1.0,
                 regularization_model=None):
        super(LaplacianRegularizedChromatogramProcessor, self).__init__(
            chromatograms, database, mass_shifts, mass_error_tolerance,
            scoring_model, smooth_overlap_rt, acceptance_threshold,
            delta_rt, peak_loader)
        if grid_smoothing_max is None:
            grid_smoothing_max = 1.0
        if network is None:
            network = database.glycan_composition_network
        self.network = network
        self.smoothing_factor = self.prepare_regularization_parameters(smoothing_factor)
        self.grid_smoothing_max = grid_smoothing_max
        self.regularization_model = regularization_model

    def prepare_regularization_parameters(self, smoothing_factor):
        if isinstance(smoothing_factor, tuple):
            if smoothing_factor[0] == self.GRID_SEARCH:
                smoothing_factor = (None, smoothing_factor[1])
        elif self.GRID_SEARCH == smoothing_factor:
            smoothing_factor = None
        return smoothing_factor

    def make_evaluator(self):
        evaluator = LaplacianRegularizedChromatogramEvaluator(
            self.scoring_model,
            self.network,
            smoothing_factor=self.smoothing_factor,
            grid_smoothing_max=self.grid_smoothing_max,
            regularization_model=self.regularization_model)
        return evaluator


class GlycopeptideChromatogramProcessor(ChromatogramProcessor):
    matcher_type = GlycopeptideChromatogramMatcher


class NonSplittingChromatogramProcessor(ChromatogramProcessor):
    matcher_type = NonSplittingChromatogramMatcher
