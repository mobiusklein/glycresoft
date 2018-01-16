from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from .glycan_visual_classification import (
    NGlycanCompositionColorizer,
    NGlycanCompositionOrderer,
    GlycanLabelTransformer)
from .chromatogram_artist import ArtistBase


class EntitySummaryBarChartArtist(ArtistBase):
    bar_width = 0.5
    alpha = 0.5
    y_label = "<SET self.y_label>"
    plot_title = "<SET self.plot_title>"

    def __init__(self, chromatograms, ax=None, colorizer=None, orderer=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        if colorizer is None:
            colorizer = NGlycanCompositionColorizer
        if orderer is None:
            orderer = NGlycanCompositionOrderer
        self.ax = ax
        self.colorizer = colorizer
        self.orderer = orderer
        self.chromatograms = [c for c in chromatograms if c.glycan_composition is not None]

    def sort_items(self):
        return self.orderer.sort(self.chromatograms, key=lambda x: x.glycan_composition)

    def get_heights(self, items, **kwargs):
        raise NotImplementedError()

    def configure_axes(self):
        self.ax.axes.spines['right'].set_visible(False)
        self.ax.axes.spines['top'].set_visible(False)
        self.ax.yaxis.tick_left()
        self.ax.xaxis.set_ticks_position('none')
        self.ax.xaxis.set_ticks_position('none')
        self.ax.set_title(self.plot_title, fontsize=24)
        self.ax.set_ylabel(self.y_label, fontsize=18)

    def __len__(self):
        return len(self.sort_items())

    def prepare_x_args(self):
        items = self.sort_items()
        if len(items) == 0:
            raise ValueError("Cannot render. Zero items to plot.")
        keys = [c.glycan_composition for c in items]
        include_classes = set(map(self.colorizer.classify, keys))
        xtick_labeler = GlycanLabelTransformer(keys, self.orderer)
        color = map(self.colorizer, keys)
        self.indices = indices = np.arange(len(items))

        self.xtick_labeler = xtick_labeler
        self.keys = keys
        self.color = color
        self.include_classes = include_classes
        self.items = items

        return items, keys, include_classes, xtick_labeler, color, indices

    def configure_x_axis(self):
        ax = self.ax
        ax.set_xticks(self.indices + (self.bar_width))
        font_size = min(
            max(
                (150. / (len(self.indices) / 2.)),
                3),
            24)

        ax.set_xlabel(self.xtick_labeler.label_key, fontsize=14)
        ax.set_xticklabels(tuple(self.xtick_labeler), rotation=90, ha='center', size=font_size)
        if len(self.indices) == 1:
            lo, hi = ax.get_xlim()
            hi *= 2
            ax.set_xlim(lo, hi)

    def draw(self, logscale=False):
        items, keys, include_classes, xtick_labeler, color, indices = self.prepare_x_args()
        heights = self.get_heights(items, logscale)

        self.bars = self.ax.bar(
            indices + self.bar_width, heights,
            width=self.bar_width, color=color, alpha=self.alpha, lw=0)

        self.configure_x_axis()

        handles = NGlycanCompositionColorizer.make_legend(
            include_classes, alpha=self.alpha)
        if handles:
            self.ax.legend(handles=handles, bbox_to_anchor=(1.20, 1.0))

        self.configure_axes()

        return self


class BundledGlycanComposition(object):
    def __init__(self, glycan_composition, total_signal):
        self.glycan_composition = glycan_composition
        self.total_signal = total_signal

    def __hash__(self):
        return hash(self.glycan_composition)

    def __str__(self):
        return str(self.glycan_composition)

    def __eq__(self, other):
        return self.glycan_composition == other

    def __repr__(self):
        return "BundledGlycanComposition(%s, %e)" % (self.glycan_composition, self.total_signal)

    @classmethod
    def aggregate(cls, observations):
        signals = Counter()
        for obs in observations:
            signals[obs.glycan_composition] += obs.total_signal
        return [cls(k, v) for k, v in signals.items()]


class AggregatedAbundanceArtist(EntitySummaryBarChartArtist):
    y_label = "Relative Intensity"
    plot_title = "Glycan Composition\nTotal Abundances"

    def get_heights(self, items, logscale=False, *args, **kwargs):
        heights = [c.total_signal for c in items]
        if logscale:
            heights = np.log2(heights)
        return heights


class ScoreBarArtist(EntitySummaryBarChartArtist):
    y_label = "Composition Score"
    plot_title = "Glycan Composition Scores"

    def get_heights(self, items, *args, **kwargs):
        heights = [c.score for c in items]
        return heights
