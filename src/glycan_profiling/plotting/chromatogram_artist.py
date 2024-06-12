from itertools import cycle
from typing import List

from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib import pyplot as plt

import glypy

from .glycan_visual_classification import NGlycanCompositionColorizer, NGlycanCompositionOrderer, GlycanLabelTransformer

from .base import ArtistBase
from ..chromatogram_tree import get_chromatogram, Chromatogram, ChromatogramInterface


def split_charge_states(chromatogram):
    charge_states = chromatogram.charge_states
    versions = {}
    last = chromatogram
    for charge_state in charge_states:
        a, b = last.bisect_charge(charge_state)
        if len(a):
            versions[charge_state] = a
        last = b
    return versions


def label_include_charges(chromatogram, *args, **kwargs):
    return "%s-%r" % (default_label_extractor(chromatogram, **kwargs), tuple(chromatogram.charge_states))


def default_label_extractor(chromatogram, **kwargs):
    if chromatogram.composition:
        return str(chromatogram.composition)
    else:
        return "%0.3f %r" % (chromatogram.neutral_mass, tuple(chromatogram.charge_states))


def binsearch(array: np.ndarray, x: float) -> int:
    lo = 0
    hi = len(array)

    while hi != lo:
        mid = (hi + lo) // 2
        y = array[mid]
        err = y - x
        if abs(err) < 1e-6:
            return mid
        elif (hi - 1) == lo:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


class ColorCycler(object):
    def __init__(self, colors=None):
        if colors is None:
            colors = ["red", "green", "blue", "yellow", "purple", "grey", "black", "orange"]
        self.color_cycler = cycle(colors)

    def __call__(self, *args, **kwargs):
        return next(self.color_cycler)


class NGlycanChromatogramColorizer(object):
    def __call__(self, chromatogram, default_color="black"):
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
    """Create a square-brace enclosed tuplet of digits denoting the count
    of of a specific set of glycan composition components common to N-Glycans.

    Relies on :class:`GlycanLabelTransformer`

    """

    def __init__(self, monosaccharides=("HexNAc", "Hex", "Fuc", "NeuAc")):
        super(NGlycanLabelProducer, self).__init__()
        self.monosaccharides = monosaccharides
        self.stub = glypy.GlycanComposition()
        for x in monosaccharides:
            self.stub[x] = -99
        self.label_key = GlycanLabelTransformer([self.stub], NGlycanCompositionOrderer).label_key

    def __call__(self, chromatogram, *args, **kwargs):
        if chromatogram.composition is not None:
            return list(
                GlycanLabelTransformer([chromatogram.glycan_composition, self.stub], NGlycanCompositionOrderer)
            )[0]
        else:
            return "%0.3f (%s)" % (chromatogram.neutral_mass, ", ".join(map(str, chromatogram.charge_states)))


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
    include_points: bool = True

    def __init__(
        self,
        chromatograms,
        ax=None,
        colorizer=None,
        label_peaks: bool = True,
        clip_labels: bool = True,
        should_draw_tandem_points: bool = True,
    ):
        if colorizer is None:
            colorizer = ColorCycler()
        if ax is None:
            fig, ax = plt.subplots(1)

        if len(chromatograms) > 0:
            chromatograms = self._resolve_chromatograms_from_argument(chromatograms)
            chromatograms = [get_chromatogram(c) for c in chromatograms]
        else:
            chromatograms = []
        self.max_points = float("inf")
        if chromatograms:
            self.max_points = max([len(c) for c in chromatograms])
        self.chromatograms = chromatograms
        self.minimum_ident_time = float("inf")
        self.maximum_ident_time = 0
        self.maximum_intensity = 0
        self.ax = ax
        self.default_colorizer = colorizer
        self.legend = None
        self.label_peaks = label_peaks
        self.clip_labels = clip_labels
        self.should_draw_tandem_points = should_draw_tandem_points

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        return self.chromatograms[i]

    def __len__(self):
        return len(self.chromatograms)

    def _resolve_chromatograms_from_argument(self, chromatograms: List[ChromatogramInterface]) -> List[Chromatogram]:
        try:
            # if not hasattr(chromatograms[0], "get_chromatogram"):
            if not get_chromatogram(chromatograms[0]):
                chromatograms = [chromatograms]
        except TypeError:
            chromatograms = [chromatograms]
        return chromatograms

    def draw_generic_chromatogram(
        self, label, rt: np.ndarray, heights: np.ndarray, color: str, fill: bool = False, label_font_size: int = 10
    ):
        if fill:
            s = self.ax.fill_between(rt, heights, alpha=0.25, color=color, label=label)

        else:
            s = self.ax.plot(rt, heights, color=color, label=label, alpha=0.5)[0]

        s.set_gid(str(label) + "-area")
        if self.include_points:
            s = self.ax.scatter(rt, heights, color=color, s=1)
            s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = heights.index(apex)
        rt_apex = rt[apex_ind]

        if label is not None:
            self.ax.text(rt_apex, apex + 1200, label, ha="center", fontsize=label_font_size, clip_on=self.clip_labels)

    def draw_group(
        self,
        label: str,
        rt: np.ndarray,
        heights: np.ndarray,
        color: str,
        label_peak: bool = True,
        chromatogram: Chromatogram = None,
        label_font_size: int = 10,
    ):
        if chromatogram is not None:
            try:
                key = str(chromatogram.id)
            except AttributeError:
                key = str(id(chromatogram))
        else:
            key = str(label)

        s = self.ax.fill_between(rt, heights, alpha=0.25, color=color, label=label)
        s.set_gid(key + "-area")
        if self.include_points:
            s = self.ax.scatter(rt, heights, color=color, s=1)
            s.set_gid(key + "-points")
        apex = max(heights)
        heights = np.array(heights)
        maximum_height_mask = heights > apex * 0.95
        apex_indices = np.where(maximum_height_mask)[0]
        apex_ind = apex_indices[apex_indices.shape[0] // 2]
        rt_apex = rt[apex_ind]

        if label is not None and label_peak:
            self.ax.text(
                rt_apex,
                min(apex * 1.1, apex + 1200),
                label,
                ha="center",
                fontsize=label_font_size,
                clip_on=self.clip_labels,
            )

    def transform_group(self, rt, heights):
        return rt, heights

    def draw_tandem_points(self, rt: np.ndarray, heights: np.ndarray, tandem_solutions: list, color: str):
        xs = []
        ys = []
        for tandem in tandem_solutions:
            try:
                x = tandem.scan_time
            except AttributeError:
                # we're dealing with a reference or a stub
                continue
            i = binsearch(rt, x)
            x0 = rt[i]
            y0 = heights[i]
            y1 = 0
            if x0 < x:
                if i < len(rt) - 1:
                    x1 = rt[i + 1]
                    y1 = heights[i + 1]
                else:
                    x1 = rt[i]
                    y1 = heights[i]
            else:
                x1 = x0
                y1 = y0
                if i > 0:
                    x0 = rt[i - 1]
                    y0 = heights[i - 1]
                else:
                    x0 = rt[i]
                    y0 = heights[i]
            # interpolate the height at the time coordinate
            den = x1 - x0
            if den == 0:
                y = (y1 + y0) / 2
            else:
                y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
            xs.append(x)
            ys.append(y)
        self.ax.scatter(xs, ys, marker="X", s=35, color=color)

    def process_group(self, composition, chromatogram: Chromatogram, label_function=None, **kwargs):
        if label_function is None:
            label_function = self.default_label_function

        color = self.default_colorizer(chromatogram)

        rt, heights = self.transform_group(*chromatogram.as_arrays())

        self.maximum_ident_time = max(max(rt), self.maximum_ident_time)
        self.minimum_ident_time = min(min(rt), self.minimum_ident_time)

        self.maximum_intensity = max(max(heights), self.maximum_intensity)

        label = label_function(chromatogram, rt=rt, heights=heights, peaks=None)
        if isinstance(label, (str, glypy.GlycanComposition)):
            label = label
            label_peak = True
        else:
            label, label_peak = label
        label_peak = label_peak & self.label_peaks

        self.draw_group(label, rt, heights, color, label_peak, chromatogram, **kwargs)

        try:
            tandem_solutions = chromatogram.tandem_solutions
        except AttributeError:
            tandem_solutions = []

        if self.should_draw_tandem_points:
            self.draw_tandem_points(rt, heights, tandem_solutions, color)

    def _interpolate_xticks(self, xlo, xhi):
        self.ax.set_xlim(xlo - 0.01, xhi + 0.01)
        tick_values = np.linspace(xlo, xhi, min(5, self.max_points))
        self.ax.set_xticks(tick_values)
        self.ax.set_xticklabels(["%0.2f" % v for v in tick_values])

    def layout_axes(
        self, legend: bool = True, axis_font_size: int = 18, axis_label_font_size: int = 16, legend_cols: int = 2
    ):
        self._interpolate_xticks(self.minimum_ident_time, self.maximum_ident_time)
        self.ax.set_ylim(0, self.maximum_intensity * 1.25)
        if legend:
            try:
                self.legend = self.ax.legend(bbox_to_anchor=(1.2, 1.0), ncol=legend_cols, fontsize=10)
            except ValueError:
                # matplotlib 2.1.1 bug compares array-like colors using == and expects a
                # scalar boolean, triggering a ValueError. When this happens, we can't
                # render a legend.
                self.legend = None
        self.ax.axes.spines["right"].set_visible(False)
        self.ax.axes.spines["top"].set_visible(False)
        self.ax.yaxis.tick_left()
        self.ax.xaxis.tick_bottom()
        self.ax.set_xlabel("Retention Time", fontsize=axis_label_font_size)
        self.ax.set_ylabel("Relative Abundance", fontsize=axis_label_font_size)
        self.ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        [t.set(fontsize=axis_font_size) for t in self.ax.get_xticklabels()]
        [t.set(fontsize=axis_font_size) for t in self.ax.get_yticklabels()]

    def draw(
        self,
        label_function=None,
        legend: bool = True,
        label_font_size: int = 10,
        axis_label_font_size: int = 16,
        axis_font_size: int = 18,
        legend_cols: int = 2,
    ):
        if label_function is None:
            label_function = self.default_label_function
        for chroma in self.chromatograms:
            composition = chroma.composition
            self.process_group(composition, chroma, label_function, label_font_size=label_font_size)
        self.layout_axes(
            legend=legend,
            axis_label_font_size=axis_label_font_size,
            axis_font_size=axis_font_size,
            legend_cols=legend_cols,
        )
        return self


class SmoothingChromatogramArtist(ChromatogramArtist):
    def __init__(
        self,
        chromatograms,
        ax=None,
        colorizer=None,
        smoothing_factor=1.0,
        label_peaks=True,
        clip_labels=True,
        should_draw_tandem_points: bool = True,
    ):
        super(SmoothingChromatogramArtist, self).__init__(
            chromatograms,
            ax=ax,
            colorizer=colorizer,
            label_peaks=label_peaks,
            clip_labels=clip_labels,
            should_draw_tandem_points=should_draw_tandem_points,
        )
        self.smoothing_factor = smoothing_factor

    def transform_group(self, rt, heights):
        heights = gaussian_filter1d(heights, self.smoothing_factor)
        return rt, heights

    def draw_generic_chromatogram(self, label, rt, heights, color, fill=False, label_font_size=10):
        heights = gaussian_filter1d(heights, self.smoothing_factor)
        if fill:
            s = self.ax.fill_between(rt, heights, alpha=0.25, color=color, label=label)

        else:
            s = self.ax.plot(rt, heights, color=color, label=label, alpha=0.5)[0]

        s.set_gid(str(label) + "-area")
        s = self.ax.scatter(rt, heights, color=color, s=1)
        s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None:
            self.ax.text(rt_apex, apex + 1200, label, ha="center", fontsize=label_font_size, clip_on=self.clip_labels)


class ChargeSeparatingChromatogramArtist(ChromatogramArtist):
    default_label_function = staticmethod(label_include_charges)

    def process_group(self, composition, chroma, label_function=None, **kwargs):
        if label_function is None:
            label_function = self.default_label_function
        charge_state_map = split_charge_states(chroma)
        for charge_state, component in charge_state_map.items():
            super(ChargeSeparatingChromatogramArtist, self).process_group(
                composition, component, label_function=label_function, **kwargs
            )


class ChargeSeparatingSmoothingChromatogramArtist(ChargeSeparatingChromatogramArtist, SmoothingChromatogramArtist):
    pass


def mass_shift_separating_chromatogram(chroma, ax=None, **kwargs):
    mass_shifts = list(chroma.mass_shifts)
    labels = {}
    for mass_shift in mass_shifts:
        with_mass_shift, _ = chroma.bisect_mass_shift(mass_shift)
        if len(with_mass_shift):
            labels[mass_shift] = with_mass_shift
    mass_shift_plot = SmoothingChromatogramArtist(list(labels.values()), colorizer=lambda *a, **k: "green", ax=ax).draw(
        label_function=lambda *a, **k: tuple(a[0].mass_shifts)[0].name, legend=False, **kwargs
    )
    rt, intens = chroma.as_arrays()
    mass_shift_plot.draw_generic_chromatogram("Total", rt, intens, color="steelblue")
    ymin = mass_shift_plot.ax.get_ylim()[0]
    mass_shift_plot.ax.set_ylim(ymin, intens.max() * 1.01)
    mass_shift_plot.ax.set_title("Mass Shift-Separated\nExtracted Ion Chromatogram", fontsize=20)
    return mass_shift_plot.ax
