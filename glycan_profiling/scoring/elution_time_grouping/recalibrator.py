from collections import namedtuple

import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from scipy import stats

from .linear_regression import SMALL_ERROR


CalibrationPoint = namedtuple("CalibrationPoint", (
    "reference_index", "reference_point_rt", "reference_point_rt_pred",
    # rrt = Relative Retention Time, prrt = Predicted Relative Retention Time
    "rrt", "prrt", "residuals", 'weight'))


class RecalibratingPredictor(object):
    def __init__(self, training_examples, testing_examples, model, scale=1.0, dilation=1.0, weighted=True):
        if training_examples is None:
            training_examples = []
        self.training_examples = np.array(training_examples)
        self.testing_examples = np.array(testing_examples)
        self.model = model
        self.scale = scale
        self.dilation = dilation
        self.configurations = dict()
        self._fit()
        self.apex_time_array = np.array(
            [self.model._get_apex_time(c) for c in testing_examples])
        if weighted:
            self.weight_array = np.array(
                [np.log10(c.total_signal) for c in testing_examples])
            self.weight_array /= self.weight_array.max()
        else:
            self.weight_array = np.ones_like(self.apex_time_array)
        self.weighted = weighted
        self.weight_array /= self.weight_array.max()
        self.predicted_apex_time_array = self._predict()
        self.score_array = self._score()

    def _adapt_dilate_fit(self, reference_point, dilation):
        parameters = np.hstack(
            [self.model.parameters[0], dilation * self.model.parameters[1:]])
        predicted_reference_point_rt = self.model._prepare_data_vector(
            reference_point).dot(parameters)
        reference_point_rt = self.model._get_apex_time(reference_point)
        resid = []
        relative_retention_time = []
        predicted_relative_retention_time = []
        for ex in self.testing_examples:
            rrt = self.model._get_apex_time(ex) - reference_point_rt
            prrt = self.model._prepare_data_vector(ex).dot(
                parameters) - predicted_reference_point_rt
            relative_retention_time.append(rrt)
            predicted_relative_retention_time.append(prrt)
            resid.append(prrt - rrt)
        return (reference_point_rt, predicted_reference_point_rt,
                relative_retention_time, predicted_relative_retention_time, resid)

    def _predict_delta_single(self, test_point, reference_point, dilation=None):
        if dilation is None:
            dilation = self.dilation
        parameters = np.hstack(
            [self.model.parameters[0], dilation * self.model.parameters[1:]])
        predicted_reference_point_rt = self.model._prepare_data_vector(
            reference_point).dot(parameters)
        reference_point_rt = self.model._get_apex_time(reference_point)
        rrt = self.model._get_apex_time(test_point) - reference_point_rt
        prrt = self.model._prepare_data_vector(test_point).dot(
            parameters) - predicted_reference_point_rt
        return prrt - rrt

    def predict_delta_single(self, test_point, dilation=None):
        if dilation is None:
            dilation = self.dilation
        delta = []
        weight = []
        for _i, reference_point in enumerate(self.testing_examples):
            delta.append(self._predict_delta_single(
                test_point, reference_point, dilation))
            weight.append(np.log10(reference_point.total_signal))
        return np.dot(delta, weight) / np.sum(weight), np.std(delta)

    def score_single(self, test_point):
        delta, _sd = (self.predict_delta_single(test_point))
        delta = abs(delta)
        score = stats.t.sf(
            delta,
            df=self._df(), scale=self.scale) * 2
        score -= SMALL_ERROR
        if score < SMALL_ERROR:
            score = SMALL_ERROR
        return score

    def _fit(self):
        for i, training_reference_point in enumerate(self.training_examples):
            reference_point = [
                c for c in self.testing_examples
                if (c.glycan_composition == training_reference_point.glycan_composition)]
            if not reference_point:
                continue
            reference_point = max(
                reference_point, key=lambda x: x.total_signal)
            dilation = self.dilation
            (reference_point_rt, predicted_reference_point_rt,
             relative_retention_time,
             predicted_relative_retention_time, resid) = self._adapt_dilate_fit(reference_point, dilation)
            self.configurations[i] = CalibrationPoint(
                i, reference_point_rt, predicted_reference_point_rt,
                np.array(relative_retention_time), np.array(
                    predicted_relative_retention_time),
                np.array(resid), np.log10(reference_point.total_signal)
            )
        if not self.configurations:
            for i, reference_point in enumerate(self.testing_examples):
                dilation = self.dilation
                (reference_point_rt, predicted_reference_point_rt,
                 relative_retention_time,
                 predicted_relative_retention_time, resid) = self._adapt_dilate_fit(reference_point, dilation)
                self.configurations[-i] = CalibrationPoint(
                    i, reference_point_rt, predicted_reference_point_rt,
                    np.array(relative_retention_time), np.array(
                        predicted_relative_retention_time),
                    np.array(resid), np.log10(reference_point.total_signal)
                )

    def _predict(self):
        configs = self.configurations
        predicted_apex_time_array = []
        weight = []
        for _key, calibration_point in configs.items():
            predicted_apex_time_array.append(
                (calibration_point.prrt + calibration_point.reference_point_rt) * calibration_point.weight)
            weight.append(calibration_point.weight)
        return np.sum(predicted_apex_time_array, axis=0) / np.sum(weight)

    def _df(self):
        return max(len(self.configurations) - len(self.model.parameters), 1)

    def _score(self):
        score = stats.t.sf(
            abs(self.predicted_apex_time_array - self.apex_time_array),
            df=self._df(), scale=self.scale) * 2
        score -= SMALL_ERROR
        score[score < SMALL_ERROR] = SMALL_ERROR
        return score

    def R2(self, adjust=False):
        y = self.apex_time_array
        w = self.weight_array
        yhat = self.predicted_apex_time_array
        residuals = (y - yhat)
        rss = (w * residuals * residuals).sum()
        tss = (y - y.mean())
        tss = (w * tss * tss).sum()
        if adjust:
            n = len(y)
            k = len(self.model.parameters)
            adjustment_factor = (n - 1.0) / float(n - k - 1.0)
        else:
            adjustment_factor = 1.0
        R2 = (1 - adjustment_factor * (rss / tss))
        return R2

    @classmethod
    def predict(cls, training_examples, testing_examples, model, dilation=1.0):
        inst = cls(training_examples, testing_examples, model, dilation)
        return inst.predicted_apex_time_array

    @classmethod
    def score(cls, training_examples, testing_examples, model, dilation=1.0):
        inst = cls(training_examples, testing_examples, model, dilation)
        return inst.score_array

    def plot(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        X = self.apex_time_array
        Y = self.predicted_apex_time_array
        S = np.array([c.total_signal for c in self.testing_examples])
        S /= S.max()
        S *= 100
        ax.scatter(X, Y, s=S, marker='o')
        ax.plot((X.min(), X.max()), (X.min(), X.max()),
                color='black', linestyle='--', lw=0.75)
        ax.set_xlabel('Experimental Apex Time', fontsize=18)
        ax.set_ylabel('Predicted Apex Time', fontsize=18)
        ax.figure.text(0.8, 0.15, "$R^2=%0.4f$" % self.R2(), ha='center')
        return ax

    def filter(self, threshold):
        filtered = self.__class__(
            self.training_examples,
            self.testing_examples[self.score_array > threshold],
            self.model,
            self.scale,
            self.dilation,
            self.weighted)
        return filtered
