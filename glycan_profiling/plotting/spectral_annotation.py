import numpy as np
from matplotlib import pyplot as plt, font_manager
from ms_peak_picker.utils import draw_peaklist


default_ion_series_to_color = {
    "y": 'red',
    "b": 'blue',
    'B': 'blue',
    'Y': 'red',
    'oxonium_ion': 'green',
    'stub_glycopeptide': 'goldenrod'
}


font_options = font_manager.FontProperties(family='sans serif')


class SpectrumMatchAnnotator(object):
    def __init__(self, spectrum_match, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        self.spectrum_match = spectrum_match
        self.ax = ax
        self.upper = max(
            spectrum_match.spectrum, key=lambda x: x.intensity
        ).intensity * 1.35

    def draw_all_peaks(self, color='black', alpha=0.5, **kwargs):
        draw_peaklist(
            self.spectrum_match.spectrum,
            alpha=0.3, color='grey', ax=self.ax, **kwargs)
        try:
            draw_peaklist(
                self.spectrum_match._sanitized_spectrum,
                color='grey', ax=self.ax, alpha=0.5, **kwargs)
        except AttributeError:
            pass

    def label_peak(self, fragment, peak, fontsize=12, rotation=90, **kw):
        label = "%s" % fragment.name
        if peak.charge > 1:
            label += "$^{%d}$" % peak.charge
        y = peak.intensity
        y = min(y + 100, self.upper * 0.95)

        return self.ax.text(
            peak.mz, y, label, rotation=rotation, va='bottom',
            ha='center', fontsize=fontsize, fontproperties=font_options,
            clip_on=True)

    def format_axes(self):
        draw_peaklist([], self.ax, pretty=True)
        self.ax.set_ylim(0, self.upper)

    def draw_matched_peaks(self, color='red', alpha=0.8, fontsize=12, ion_series_to_color=None, **kwargs):
        if ion_series_to_color is None:
            ion_series_to_color = {}

        for peak, fragment in self.spectrum_match.solution_map:
            try:
                peak_color = ion_series_to_color.get(fragment.series, color)
            except AttributeError:
                peak_color = ion_series_to_color.get(fragment.kind, color)
            draw_peaklist([peak], alpha=alpha, ax=self.ax, color=peak_color)
            self.label_peak(fragment, peak, fontsize=fontsize, **kwargs)

    def draw_spectrum_graph(self, color='red', alpha=0.8, fontsize=12, **kwargs):
        try:
            graph = self.spectrum_match.spectrum_graph
        except AttributeError:
            return

        paths = graph.longest_paths()

        for path in paths:
            for edge in path:
                self.draw_peak_pair((edge.start, edge.end), color, alpha, fontsize,
                                    label=edge.annotation, **kwargs)

    def draw_peak_pair(self, pair, color='red', alpha=0.8, fontsize=12, label=None, rotation=45, **kwargs):
        p1, p2 = pair
        self.ax.plot((p1.mz, p2.mz), (p1.intensity, p2.intensity),
                     color=color, alpha=alpha, **kwargs)
        draw_peaklist(pair, ax=self.ax, alpha=0.4, color='orange')
        if label:
            midx = (p1.mz + p2.mz) / 2
            # interpolate the midpoint's height
            midy = (p1.intensity * (p2.mz - midx) + p2.intensity * (midx - p1.mz)) / (p2.mz - p1.mz)
            if isinstance(label, (list, tuple)):
                label = '-'.join(map(str, label))
            else:
                label = str(label)
            self.ax.text(midx, midy, label, fontsize=fontsize,
                         ha='center', va='bottom', rotation=rotation, clip_on=True)

    def draw(self, **kwargs):
        fontsize = kwargs.pop('fontsize', 9)
        rotation = kwargs.pop("rotation", 90)
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        self.draw_all_peaks(**kwargs)
        self.draw_matched_peaks(
            fontsize=fontsize, ion_series_to_color=ion_series_to_color,
            rotation=rotation, **kwargs)
        self.draw_spectrum_graph(fontsize=fontsize, rotation=rotation / 2)
        self.format_axes()
        return self

    def __repr__(self):
        return "{self.__class__.__name__}({self.spectrum_match})".format(
            self=self)
