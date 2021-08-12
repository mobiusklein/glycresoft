from array import array
from collections import defaultdict

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition

class RevisionRule(object):
    def __init__(self, delta_glycan, mass_shift_rule=None):
        self.delta_glycan = HashableGlycanComposition.parse(delta_glycan)
        self.mass_shift_rule = mass_shift_rule

    def valid(self, record):
        new_record = self(record)
        return not any(v < 0 for v in new_record.glycan_composition.values())

    def __call__(self, record):
        return record.shift_glycan_composition(self.delta_glycan)

    def __eq__(self, other):
        try:
            return self.delta_glycan == other.delta_glycan
        except AttributeError:
            return self.delta_glycan == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.delta_glycan)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.delta_glycan})"
        return template.format(self=self)


class ModelReviser(object):
    def __init__(self, model, rules, chromatograms=None):
        if chromatograms is None:
            chromatograms = model.chromatograms
        self.model = model
        self.chromatograms = chromatograms
        self.original_scores = array('d')
        self.rules = list(rules)
        self.alternative_records = defaultdict(list)
        self.alternative_scores = defaultdict(lambda: array('d'))

    def rescore(self, case):
        return self.model.score(case)

    def process_rule(self, rule):
        alts = []
        alt_scores = []
        for case in self.chromatograms:
            if rule.valid(case):
                rec = rule(case)
                alts.append(rec)
                alt_scores.append(self.rescore(rec))
            else:
                alts.append(None)
                alt_scores.append(0.0)
        self.alternative_records[rule] = alts
        self.alternative_scores[rule] = alt_scores

    def process_model(self):
        scores = []
        for case in self.chromatograms:
            scores.append(self.rescore(case))
        self.original_scores = scores

    def evaluate(self):
        self.process_model()
        for rule in self.rules:
            self.process_rule(rule)

    def revise(self, threshold=0.2, delta_threshold=0.2):
        chromatograms = self.chromatograms
        original_scores = self.original_scores
        next_round = []
        for i in range(len(chromatograms)):
            best_score = original_scores[i]
            delta_best_score = float('inf')
            best_record = chromatograms[i]
            for rule in self.rules:
                a = self.alternative_scores[rule][i]
                if a > best_score and not np.isclose(a, 0.0):
                    delta_best_score = a - original_scores[i]
                    best_score = a
                    best_record = self.alternative_records[rule][i]

            if best_score > threshold and delta_best_score > delta_threshold:
                next_round.append(best_record)
            else:
                next_round.append(chromatograms[i])
        return next_round


AmmoniumMaskedRule = RevisionRule(
    HashableGlycanComposition(Hex=-1, Fuc=-1, Neu5Ac=1))
IsotopeRule = RevisionRule(HashableGlycanComposition(Fuc=-2, Neu5Ac=1))
AmmoniumUnmaskedRule = RevisionRule(
    HashableGlycanComposition(Hex=1, Fuc=1, Neu5Ac=-1))


class IntervalModelReviser(ModelReviser):
    alpha = 0.01

    def rescore(self, case):
        return self.model.score_interval(case, alpha=0.01)
