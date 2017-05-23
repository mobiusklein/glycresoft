from collections import OrderedDict
from matplotlib import pyplot as plt

from .chromatogram_artist import (
    SmoothingChromatogramArtist, AbundantLabeler,
    NGlycanLabelProducer, n_glycan_colorizer)

from .entity_bar_chart import AggregatedAbundanceArtist, BundledGlycanComposition
from .utils import figax


class GlycanChromatographySummaryGraphBuilder(object):
    def __init__(self, solutions):
        self.solutions = solutions

    def chromatograms(self, min_score=0.4, min_signal=0.2, colorizer=None, total_ion_chromatogram=None,
                      base_peak_chromatogram=None):
        monosaccharides = set()

        for sol in self.solutions:
            if sol.glycan_composition:
                monosaccharides.update(map(str, sol.glycan_composition))

        label_abundant = AbundantLabeler(
            NGlycanLabelProducer(monosaccharides),
            max(sol.total_signal for sol in self.solutions if sol.score > min_score) * min_signal)

        if colorizer is None:
            colorizer = n_glycan_colorizer

        results = [sol for sol in self.solutions if sol.score > min_score and not sol.used_as_adduct]
        chrom = SmoothingChromatogramArtist(
            results, ax=figax(), colorizer=colorizer).draw(label_function=label_abundant)

        if total_ion_chromatogram is not None:
            rt, intens = total_ion_chromatogram.as_arrays()
            chrom.draw_generic_chromatogram(
                "TIC", rt, intens, 'blue')
            chrom.ax.set_ylim(0, max(intens) * 1.1)

        if base_peak_chromatogram is not None:
            rt, intens = base_peak_chromatogram.as_arrays()
            chrom.draw_generic_chromatogram(
                "BPC", rt, intens, 'green')
        return chrom

    def aggregated_abundance(self, min_score=0.4):
        agg = AggregatedAbundanceArtist(
            BundledGlycanComposition.aggregate([
                sol for sol in self.solutions if (sol.score > min_score and
                                                  not sol.used_as_adduct and
                                                  sol.glycan_composition is not None)]),
            ax=figax())
        if len(agg) == 0:
            ax = agg.ax
            ax.text(0.5, 0.5, "No Entities Matched", ha='center')
            ax.set_axis_off()
        else:
            agg.draw()
        return agg

    def draw(self, min_score=0.4, min_signal=0.2, colorizer=None, total_ion_chromatogram=None,
             base_peak_chromatogram=None):
        chrom = self.chromatograms(min_score, min_signal, colorizer,
                                   total_ion_chromatogram, base_peak_chromatogram)
        agg = self.aggregated_abundance(min_score)
        return chrom, agg


def breaklines(cases):
    counts = OrderedDict()
    counter = 0
    ms2_score = 0
    cases = sorted(cases, key=lambda x: x.ms2_score, reverse=True)
    i = 0
    for case in cases:
        i += 1
        if case.ms2_score == ms2_score:
            counter += 1
        else:
            counts[ms2_score] = counter
            ms2_score = case.ms2_score
            counter += 1
    if counts[0] == 0:
        counts.pop(0)
    return counts


def plot_tapering(cases, threshold=0.05, ax=None, **kwargs):
    plot_kwargs = {
        "alpha": 0.5,
        "lw": 2
    }
    plot_kwargs.update(kwargs)
    counts = breaklines(cases)
    if ax is None:
        fig, ax = plt.subplots(1)

    score_at_threshold = float('inf')
    for t in cases:
        if t.q_value < 0.05:
            if t.score < score_at_threshold:
                score_at_threshold = t.score

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.plot(*zip(*counts.items()), **plot_kwargs)
    ax.set_xlim(*(ax.get_xlim()[::-1]))

    xlim = ax.get_xlim()
    ax.hlines(counts[score_at_threshold], max(xlim), 0, linestyles='--')

    ax.set_xlabel("PSM Score Threshold", fontsize=18)
    ax.set_ylabel("# of PSMS < Threshold", fontsize=18)
    return ax
