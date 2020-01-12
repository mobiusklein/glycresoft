from collections import defaultdict

from glycan_profiling.task import TaskBase

from .cross_run import ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter
from .model import AbundanceWeightedPeptideFactorElutionTimeFitter

class GlycopeptideElutionTimeModeler(TaskBase):
    def __init__(self, glycopeptide_chromatograms, factors=None, refit_filter=0.01):
        self.glycopeptide_chromatograms = glycopeptide_chromatograms
        self.factors = factors
        if self.factors is None:
            self.factors = self._infer_factors()
        self.joint_model = None
        self.refit_filter = refit_filter
        self.by_peptide = defaultdict(list)
        self.peptide_specific_models = dict()
        self._partition_by_sequence()

    def _partition_by_sequence(self):
        for record in self.glycopeptide_chromatograms:
            key = str(record.clone().deglycosylate())
            self.by_peptide[key].append(record)

    def _infer_factors(self):
        keys = set()
        for record in self.glycopeptide_chromatograms:
            keys.update(record.glycan_composition)
        keys = sorted(map(str, keys))
        return keys

    def fit_model(self, glycopeptide_chromatograms):
        model = ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter(
            glycopeptide_chromatograms, self.factors)
        model.fit()
        return model

    def fit(self):
        self.log("Fitting Joint Model")
        model = self.fit_model(self.glycopeptide_chromatograms)
        self.log("R^2: %0.3f" % (model.R2(), ))
        if self.refit_filter != 0.0:
            self.log("Filtering Training Data")
            filtered_cases = [
                case for case in self.glycopeptide_chromatograms
                if model.score(case) > self.refit_filter
            ]
            self.log("Re-fitting After Filtering")
            model = self.fit_model(filtered_cases)
            self.log("R^2: %0.3f" % (model.R2(), ))
        self.joint_model = model
