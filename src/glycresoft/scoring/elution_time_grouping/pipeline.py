import os

from collections import defaultdict

import dill as pickle

import numpy as np

from scipy.stats import gaussian_kde

import glycopeptidepy

from glycresoft.task import TaskBase
from glycresoft.output.csv_format import csv_stream

from .structure import GlycopeptideChromatogramProxy
from .cross_run import ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter


# Deprecated. Adapt to model.GlycopeptideElutionTimeModelBuildingPipeline
class GlycopeptideElutionTimeModeler(TaskBase):
    _model_class = ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter

    def __init__(self, glycopeptide_chromatograms, factors=None, refit_filter=0.01, replicate_key_attr=None,
                 test_chromatograms=None, use_retention_time_normalization=False, prefer_joint_model=False,
                 minimum_observations_for_specific_model=20):
        if replicate_key_attr is None:
            replicate_key_attr = 'analysis_name'
        if test_chromatograms is not None:
            if not isinstance(test_chromatograms[0], GlycopeptideChromatogramProxy):
                test_chromatograms = [
                    GlycopeptideChromatogramProxy.from_chromatogram(i) for i in test_chromatograms]

        self.replicate_key_attr = replicate_key_attr
        if not isinstance(glycopeptide_chromatograms[0], GlycopeptideChromatogramProxy):
            glycopeptide_chromatograms = [
                GlycopeptideChromatogramProxy.from_chromatogram(i) for i in glycopeptide_chromatograms]
        self.glycopeptide_chromatograms = glycopeptide_chromatograms
        self.test_chromatograms = test_chromatograms
        self.prefer_joint_model = prefer_joint_model
        self.minimum_observations_for_specific_model = minimum_observations_for_specific_model
        self.factors = factors
        if self.factors is None:
            self.factors = self._infer_factors()
        self.use_retention_time_normalization = use_retention_time_normalization
        self.joint_model = None
        self.refit_filter = refit_filter
        self.by_peptide = defaultdict(list)
        self.peptide_specific_models = dict()
        self.delta_by_factor = dict()
        self._partition_by_sequence()

    def _partition_by_sequence(self):
        for record in self.glycopeptide_chromatograms:
            key = glycopeptidepy.parse(str(record.structure)).deglycosylate()
            self.by_peptide[key].append(record)

    def _deltas_for(self, monosaccharide):
        deltas = []
        for _backbone, cases in self.by_peptide.items():
            for target in cases:
                gc = target.glycan_composition.clone()
                gc[monosaccharide] += 1
                key = self.joint_model._get_replicate_key(target)
                for case in cases:
                    if case.glycan_composition == gc and self.joint_model._get_replicate_key(case) == key:
                        deltas.append(case.apex_time - target.apex_time)
        return np.array(deltas)

    def _infer_factors(self):
        keys = set()
        for record in self.glycopeptide_chromatograms:
            keys.update(record.glycan_composition)
        keys = sorted(map(str, keys))
        return keys

    def fit_model(self, glycopeptide_chromatograms):
        model = self._model_class(
            glycopeptide_chromatograms, self.factors,
            replicate_key_attr=self.replicate_key_attr,
            use_retention_time_normalization=self.use_retention_time_normalization)
        model.fit()
        return model

    def fit(self):
        self.log("Fitting Joint Model")
        model = self.fit_model(self.glycopeptide_chromatograms)
        self.log("R^2: %0.3f, MSE: %0.3f" % (model.R2(), model.mse))
        if self.refit_filter != 0.0:
            self.log("Filtering Training Data")
            filtered_cases = [
                case for case in self.glycopeptide_chromatograms
                if model.score(case) > self.refit_filter
            ]
            self.log("Re-fitting After Filtering")
            model = self.fit_model(filtered_cases)
            self.log("R^2: %0.3f, MSE: %0.3f" % (model.R2(), model.mse))
        self.log('\n' + model.summary())
        self.joint_model = model
        factors = sorted(self.factors)
        self.log("Measuring Single Monosaccharide Deltas, Median and MAD")
        for key in factors:
            deltas = self._deltas_for(key)
            self.delta_by_factor[key] = deltas
            self.log("%s:   %0.3f   %0.3f" %
                     (key,
                      np.median(deltas),
                      np.median(np.abs(deltas - np.median(deltas)), )))
        for key, members in self.by_peptide.items():
            distinct_members = set(str(m.structure) for m in members)
            self.log("Fitting Model For %s (%d observations, %d distinct)" % (key, len(members), len(distinct_members)))
            if len(distinct_members) <= max(len(self.factors), self.minimum_observations_for_specific_model):
                self.log("Too few distinct observations for %s" % (key, ))
                continue
            model = self.fit_model(members)
            self.log("R^2: %0.3f, MSE: %0.3f" % (model.R2(), model.mse))
            if self.refit_filter != 0.0:
                self.log("Filtering Training Data")
                filtered_cases = [
                    case for case in members
                    if model.score(case) > self.refit_filter
                ]
                self.log("Re-fitting After Filtering")
                model = self.fit_model(filtered_cases)
                self.log("R^2: %0.3f, MSE: %0.3f" % (model.R2(), model.mse))
            self.log('\n' + model.summary())
            self.peptide_specific_models[key] = model
            joint_perf = np.mean(map(self.joint_model.score, members))
            spec_perf = np.mean(map(model.score, members))
            self.log("Mean Peptide Model Score: %0.3f" % (spec_perf, ))
            self.log("Mean Joint Model Score:   %0.3f" % (joint_perf, ))

    def evaluate_training(self):
        for key, group in self.by_peptide.items():
            self.log("Evaluating %s" % key)
            for obs in group:
                model = self._model_for(obs)
                score = model.score(obs)
                pred = model.predict(obs)
                delta = model._get_apex_time(obs) - pred
                obs.annotations['score'] = score
                obs.annotations['predicted_apex_time'] = pred
                obs.annotations['delta_apex_time'] = delta

    def evaluate(self, observations):
        for obs in observations:
            model = self._model_for(obs)
            score = model.score(obs)
            pred = model.predict(obs)
            delta = model._get_apex_time(obs) - pred
            obs.annotations['score'] = score
            obs.annotations['predicted_apex_time'] = pred
            obs.annotations['delta_apex_time'] = delta

    def _model_for(self, observation):
        if self.prefer_joint_model:
            return self.joint_model
        key = glycopeptidepy.parse(str(observation.structure)).deglycosylate()
        model = self.peptide_specific_models.get(key, self.joint_model)
        return model

    def predict(self, observation):
        model = self._model_for(observation)
        return model.predict(observation)

    def score(self, observation):
        model = self._model_for(observation)
        return model.score(observation)

    def run(self):
        self.fit()
        self.evaluate_training()
        if self.test_chromatograms:
            self.evaluate(self.test_chromatograms)

    def write(self, path):
        from glycresoft.output.report.base import render_plot
        from glycresoft.plotting.base import figax
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise IOError("Expected a path to a directory, %s is a file!" % (path, ))
        pjoin = os.path.join
        self.log("Writing scored chromatograms")
        with csv_stream(open(pjoin(path, "scored_chromatograms.csv"), 'wb')) as fh:
            GlycopeptideChromatogramProxy.to_csv(self.glycopeptide_chromatograms, fh)
        if self.test_chromatograms:
            with csv_stream(open(pjoin(path, "test_chromatograms.csv"), 'wb')) as fh:
                GlycopeptideChromatogramProxy.to_csv(self.test_chromatograms, fh)
        self.log("Writing joint model descriptors")
        with csv_stream(open(pjoin(path, "joint_model_parameters.csv"), 'wb')) as fh:
            self.joint_model.to_csv(fh)
        with open(pjoin(path, "joint_model_predplot.png"), 'wb') as fh:
            ax = figax()
            self.joint_model.prediction_plot(ax=ax)
            fh.write(render_plot(ax, dpi=160.0).getvalue())

        if self.use_retention_time_normalization:
            with open(pjoin(path, "retention_time_normalization.png"), 'wb') as fh:
                ax = figax()
                self.joint_model.run_normalizer.plot(ax=ax)
                ax.set_title("Cross-Run RT Correction")
                fh.write(render_plot(ax, dpi=160.0).getvalue())

        for key, model in self.peptide_specific_models.items():
            self.log("Writing %s model descriptors" % (key, ))
            with csv_stream(open(pjoin(path, "%s_model_parameters.csv" % (key, )), 'wb')) as fh:
                model.to_csv(fh)
            with open(pjoin(path, "%s_model_predplot.png" % (key, )), 'wb') as fh:
                ax = figax()
                model.prediction_plot(ax=ax)
                ax.set_title(key)
                fh.write(render_plot(ax, dpi=160.0).getvalue())

        for factor, deltas in self.delta_by_factor.items():
            with open(pjoin(path, '%s_delta_hist.png' % (factor.replace("@", ""), )), 'wb') as fh:
                ax = figax()

                ax.hist(deltas, bins='auto', density=False, ec='black', alpha=0.5)
                ax.set_title(factor)
                ax.figure.text(0.75, 0.8, 'Median: %0.3f' % np.median(deltas), ha='center')
                ax.set_xlabel("RT Shift (Min)")
                ax.set_ylabel("Count")
                ax2 = ax.twinx()
                if len(deltas) > 1:
                    x = np.linspace(deltas.min() + -0.1, deltas.max() + 0.1)
                    m = gaussian_kde(deltas)
                    ax2.fill_between(x, 0, m.pdf(x), alpha=0.5, color='green')
                    ax2.set_ylabel("Density")
                    ax2.set_ylim(0, ax2.get_ylim()[1])
                else:
                    m = None
                fh.write(render_plot(ax, dpi=160.0).getvalue())

        self.log("Saving models")
        with open(pjoin(path, "model.pkl"), 'wb') as fh:
            pickle.dump(self, fh, -1)
