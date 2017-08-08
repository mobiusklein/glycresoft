from itertools import cycle

from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib import pyplot as plt

import glypy

from .glycan_visual_classification import (
    NGlycanCompositionColorizer, NGlycanCompositionOrderer,
    GlycanLabelTransformer)
from .base import ArtistBase
from ..chromatogram_tree import get_chromatogram


def split_charge_states(chromatogram):
    charge_states = chromatogram.charge_states
    versions = {}
    last = chromatogram
    for charge_state in charge_states:
        a, b = last.bisect_charge(charge_state)
        versions[charge_state] = a
        last = b
    return versions


def label_include_charges(chromatogram, *args, **kwargs):
    return "%s-%r" % (
        default_label_extractor(chromatogram, **kwargs),
        tuple(chromatogram.charge_states))


def default_label_extractor(chromatogram, **kwargs):
    if chromatogram.composition:
        return str(chromatogram.composition)
    else:
        return "%0.3f %r" % (chromatogram.neutral_mass, tuple(chromatogram.charge_states))


class ColorCycler(object):
    def __init__(self, colors=None):
        if colors is None:
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey', 'black', "orange"]
        self.color_cycler = cycle(colors)

    def __call__(self, *args, **kwargs):
        return next(self.color_cycler)


class NGlycanChromatogramColorizer(object):
    def __call__(self, chromatogram, default_color='black'):
        if chromatogram.composition is None:
            return default_color
        else:
            try:
                return NGlycanCompositionColorizer(chromatogram.glycan_composition)
            except Exception:
                return default_color


n_glycan_colorizer = NGlycanChromatogramColorizer()


class LabelProducer(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, chromatogram, *args, **kwargs):
        return default_label_extractor(chromatogram)


class NGlycanLabelProducer(LabelProducer):
    def __init__(self, monosaccharides=("HexNAc", "Hex", "Fuc", "NeuAc")):
        self.monosaccharides = monosaccharides
        self.stub = glypy.GlycanComposition()
        for x in monosaccharides:
            self.stub[x] = -99
        self.label_key = GlycanLabelTransformer([self.stub], NGlycanCompositionOrderer).label_key

    def __call__(self, chromatogram, *args, **kwargs):
        if chromatogram.composition is not None:
            return list(GlycanLabelTransformer(
                [chromatogram.glycan_composition, self.stub], NGlycanCompositionOrderer))[0]
        else:
            return "%0.3f (%s)" % (chromatogram.neutral_mass, ", ".join(
                map(str, chromatogram.charge_states)))


n_glycan_labeler = NGlycanLabelProducer()


class AbundantLabeler(LabelProducer):
    def __init__(self, labeler, threshold):
        self.labeler = labeler
        self.threshold = threshold

    def __call__(self, chromatogram, *args, **kwargs):
        if chromatogram.total_signal > self.threshold:
            return self.labeler(chromatogram, *args, **kwargs), True
        else:
            return self.labeler(chromatogram, *args, **kwargs), False


class ChromatogramArtist(ArtistBase):
    default_label_function = staticmethod(default_label_extractor)
    include_points = True

    def __init__(self, chromatograms, ax=None, colorizer=None, label_peaks=True):
        if colorizer is None:
            colorizer = ColorCycler()
        if ax is None:
            fig, ax = plt.subplots(1)

        if len(chromatograms) > 0:
            chromatograms = self._resolve_chromatograms_from_argument(chromatograms)
            chromatograms = [get_chromatogram(c) for c in chromatograms]
        else:
            chromatograms = []
        self.chromatograms = chromatograms
        self.minimum_ident_time = float("inf")
        self.maximum_ident_time = 0
        self.maximum_intensity = 0
        self.ax = ax
        self.default_colorizer = colorizer
        self.legend = None
        self.label_peaks = label_peaks

    def _resolve_chromatograms_from_argument(self, chromatograms):
        try:
            # if not hasattr(chromatograms[0], "get_chromatogram"):
            if not get_chromatogram(chromatograms[0]):
                chromatograms = [chromatograms]
        except TypeError:
            chromatograms = [chromatograms]
        return chromatograms

    def draw_generic_chromatogram(self, label, rt, heights, color, fill=False, label_font_size=10):
        if fill:
            s = self.ax.fill_between(
                rt,
                heights,
                alpha=0.25,
                color=color,
                label=label
            )

        else:
            s = self.ax.plot(rt, heights, color=color, label=label, alpha=0.5)[0]

        s.set_gid(str(label) + "-area")
        if self.include_points:
            s = self.ax.scatter(
                rt,
                heights,
                color=color,
                s=1)
            s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = heights.index(apex)
        rt_apex = rt[apex_ind]

        if label is not None:
            self.ax.text(rt_apex, apex + 1200, label, ha='center', fontsize=label_font_size)

    def draw_group(self, label, rt, heights, color, label_peak=True, chromatogram=None, label_font_size=10):
        if chromatogram is not None:
            try:
                key = str(chromatogram.id)
            except AttributeError:
                key = str(id(chromatogram))
        else:
            key = str(label)

        s = self.ax.fill_between(
            rt,
            heights,
            alpha=0.25,
            color=color,
            label=label
        )
        s.set_gid(key + "-area")
        if self.include_points:
            s = self.ax.scatter(
                rt,
                heights,
                color=color,
                s=1)
            s.set_gid(key + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None and label_peak:
            self.ax.text(rt_apex, min(apex * 1.1, apex + 1200), label, ha='center', fontsize=label_font_size)

    def process_group(self, composition, chromatogram, label_function=None, **kwargs):
        if label_function is None:
            label_function = self.default_label_function

        color = self.default_colorizer(chromatogram)

        rt, heights = chromatogram.as_arrays()

        self.maximum_ident_time = max(max(rt), self.maximum_ident_time)
        self.minimum_ident_time = min(min(rt), self.minimum_ident_time)

        self.maximum_intensity = max(max(heights), self.maximum_intensity)

        label = label_function(
            chromatogram, rt=rt, heights=heights, peaks=None)
        if isinstance(label, basestring):
            label = label
            label_peak = True
        else:
            label, label_peak = label
        label_peak = label_peak & self.label_peaks

        self.draw_group(label, rt, heights, color, label_peak, chromatogram, **kwargs)

    def _interpolate_xticks(self, xlo, xhi):
        self.ax.set_xlim(xlo - 0.02, xhi + 0.02)
        span_time = xhi - xlo
        tick_values = np.linspace(
            xlo + min(0.05, 0.1 * span_time),
            xhi - min(0.05, 0.1 * span_time),
            6)
        self.ax.set_xticks(tick_values)
        self.ax.set_xticklabels(["%0.2f" % v for v in tick_values])

    def layout_axes(self, legend=True, axis_font_size=18, axis_label_font_size=24):
        self._interpolate_xticks(self.minimum_ident_time, self.maximum_ident_time)
        self.ax.set_ylim(0, self.maximum_intensity * 1.25)
        if legend:
            self.legend = self.ax.legend(bbox_to_anchor=(1.2, 1.), ncol=2, fontsize=10)
        self.ax.axes.spines['right'].set_visible(False)
        self.ax.axes.spines['top'].set_visible(False)
        self.ax.yaxis.tick_left()
        self.ax.xaxis.tick_bottom()
        self.ax.set_xlabel("Retention Time", fontsize=axis_label_font_size)
        self.ax.set_ylabel("Relative Abundance", fontsize=axis_label_font_size)
        [t.set(fontsize=axis_font_size) for t in self.ax.get_xticklabels()]
        [t.set(fontsize=axis_font_size) for t in self.ax.get_yticklabels()]

    def draw(self, filter_function=lambda x, y: False, label_function=None,
             legend=True, label_font_size=10):
        if label_function is None:
            label_function = self.default_label_function
        for chroma in self.chromatograms:
            composition = chroma.composition
            if composition is not None:
                if hasattr(chroma, 'entity') and chroma.entity is not None:
                    gc = chroma.glycan_composition
                else:
                    gc = glypy.GlycanComposition.parse(composition)
            else:
                gc = None
            if filter_function(gc, chroma):
                continue

            self.process_group(composition, chroma, label_function, label_font_size=label_font_size)
        self.layout_axes(legend=legend)
        return self


class SmoothingChromatogramArtist(ChromatogramArtist):
    def __init__(self, chromatograms, ax=None, colorizer=None, smoothing_factor=1.0, label_peaks=True):
        super(SmoothingChromatogramArtist, self).__init__(
            chromatograms, ax=ax, colorizer=colorizer, label_peaks=label_peaks)
        self.smoothing_factor = smoothing_factor

    def draw_group(self, label, rt, heights, color, label_peak=True, chromatogram=None, label_font_size=10):
        if chromatogram is not None:
            try:
                key = str(chromatogram.id)
            except AttributeError:
                key = str(id(chromatogram))
        else:
            key = str(label)
        heights = gaussian_filter1d(heights, self.smoothing_factor)
        s = self.ax.fill_between(
            rt,
            heights,
            alpha=0.25,
            color=color,
            label=label
        )
        s.set_gid(key + "-area")
        s = self.ax.scatter(
            rt,
            heights,
            color=color,
            s=1)
        s.set_gid(key + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None and label_peak:
            self.ax.text(rt_apex, min(apex * 1.1, apex + 1200), label, ha='center', fontsize=label_font_size)

    def draw_generic_chromatogram(self, label, rt, heights, color, fill=False, label_font_size=10):
        heights = gaussian_filter1d(heights, self.smoothing_factor)
        if fill:
            s = self.ax.fill_between(
                rt,
                heights,
                alpha=0.25,
                color=color,
                label=label
            )

        else:
            s = self.ax.plot(rt, heights, color=color, label=label, alpha=0.5)[0]

        s.set_gid(str(label) + "-area")
        s = self.ax.scatter(
            rt,
            heights,
            color=color,
            s=1)
        s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None:
            self.ax.text(rt_apex, apex + 1200, label, ha='center', fontsize=label_font_size)


class ChargeSeparatingChromatogramArtist(ChromatogramArtist):
    default_label_function = staticmethod(label_include_charges)

    def process_group(self, composition, chroma, label_function=None, **kwargs):
        if label_function is None:
            label_function = self.default_label_function
        charge_state_map = split_charge_states(chroma)
        for charge_state, component in charge_state_map.items():
            super(ChargeSeparatingChromatogramArtist, self).process_group(
                composition, component, label_function=label_function, **kwargs)


class ChargeSeparatingSmoothingChromatogramArtist(
        ChargeSeparatingChromatogramArtist, SmoothingChromatogramArtist):
    pass
