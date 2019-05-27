import numpy as np
from matplotlib import pyplot as plt, font_manager
from ms_peak_picker.utils import draw_peaklist
from .sequence_fragment_logo import glycopeptide_match_logo


default_ion_series_to_color = {
    "y": 'red',
    'z': 'maroon',
    "b": 'blue',
    'c': 'navy',
    'B': 'blue',
    'Y': 'red',
    'oxonium_ion': 'green',
    'stub_glycopeptide': 'goldenrod',
    'precursor': 'orange',
}


font_options = font_manager.FontProperties(family='sans serif')


class SpectrumMatchAnnotator(object):
    def __init__(self, spectrum_match, ax=None, clip_labels=True):
        if ax is None:
            _, ax = plt.subplots(1)
        self.spectrum_match = spectrum_match
        self.ax = ax
        self.clip_labels = clip_labels
        self.upper = max(
            spectrum_match.spectrum, key=lambda x: x.intensity
        ).intensity * 1.35
        self.peak_labels = []

    def draw_all_peaks(self, color='black', alpha=0.5, **kwargs):
        draw_peaklist(
            self.spectrum_match.deconvoluted_peak_set,
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

        kw.setdefault("clip_on", self.clip_labels)
        clip_on = kw['clip_on']

        text = self.ax.text(
            peak.mz, y, label, rotation=rotation, va='bottom',
            ha='center', fontsize=fontsize, fontproperties=font_options,
            clip_on=clip_on)
        self.peak_labels.append(text)
        return text

    def format_axes(self):
        draw_peaklist([], self.ax, pretty=True)
        self.ax.set_ylim(0, self.upper)

    def draw_matched_peaks(self, color='red', alpha=0.8, fontsize=12, ion_series_to_color=None, **kwargs):
        if ion_series_to_color is None:
            ion_series_to_color = {}
        try:
            solution_map = self.spectrum_match.solution_map
        except AttributeError:
            return
        for peak, fragment in solution_map:
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

        paths = graph.longest_paths(limit=100)

        for path in paths:
            for edge in path:
                self.draw_peak_pair((edge.start, edge.end), color, alpha, fontsize,
                                    label=edge.annotation, **kwargs)

    def draw_peak_pair(self, pair, color='red', alpha=0.8, fontsize=12, label=None, rotation=45, **kwargs):
        p1, p2 = pair
        self.ax.plot((p1.mz, p2.mz), (p1.intensity, p2.intensity),
                     color=color, alpha=alpha, **kwargs)
        kwargs.setdefault("clip_on", self.clip_labels)
        clip_on = kwargs['clip_on']
        draw_peaklist(pair, ax=self.ax, alpha=0.4, color='orange')
        if label:
            midx = (p1.mz + p2.mz) / 2
            # interpolate the midpoint's height
            midy = (p1.intensity * (p2.mz - midx) + p2.intensity * (midx - p1.mz)) / (p2.mz - p1.mz)

            # find the angle of the line connecting the two peaks
            xlo = min(p1.mz, p2.mz)
            xhi = max(p1.mz, p2.mz)
            adj = xhi - xlo
            ylo = min(p1.intensity, p2.intensity)
            yhi = max(p1.intensity, p2.intensity)
            opp = yhi - ylo
            hypot = np.hypot(adj, opp)
            rotation = np.arccos(adj / hypot)

            if isinstance(label, (list, tuple)):
                label = '-'.join(map(str, label))
            else:
                label = str(label)
            self.ax.text(midx, midy, label, fontsize=fontsize,
                         ha='center', va='bottom', rotation=rotation, clip_on=clip_on)

    def draw(self, **kwargs):
        fontsize = kwargs.pop('fontsize', 9)
        rotation = kwargs.pop("rotation", 90)
        clip_labels = kwargs.pop("clip_labels", self.clip_labels)
        self.clip_labels = clip_labels
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        self.draw_all_peaks(**kwargs)
        self.draw_matched_peaks(
            fontsize=fontsize, ion_series_to_color=ion_series_to_color,
            rotation=rotation, **kwargs)
        self.draw_spectrum_graph(fontsize=fontsize, rotation=rotation / 2)
        self.format_axes()
        return self

    def add_logo_plot(self, xrel=0.15, yrel=0.8, width=0.67, height=0.13, **kwargs):
        figure = self.ax.figure
        iax = figure.add_axes([xrel, yrel, width, height])
        logo = glycopeptide_match_logo(self.spectrum_match, ax=iax, **kwargs)
        return logo

    def _draw_mass_accuracy_plot(self, ax, error_tolerance=2e-5, **kwargs):
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        match = self.spectrum_match
        ax.scatter(*zip(*[(pp.peak.mz, pp.mass_accuracy()) for pp in match.solution_map]),
                   alpha=0.5, edgecolor='black', color=[
                       ion_series_to_color[pp.fragment.series] for pp in match.solution_map])
        limits = error_tolerance
        ax.set_ylim(-limits, limits)
        xlim = 0, max(match.deconvoluted_peak_set, key=lambda x: x.mz).mz + 100
        ax.hlines(0, *xlim, linestyle='--')
        ax.hlines(0, *xlim, linestyle='--')
        ax.hlines(limits / 2, *xlim, linestyle='--', lw=0.5)
        ax.hlines(-limits / 2, *xlim, linestyle='--', lw=0.5)
        ax.set_xlim(*xlim)
        labels = ax.get_yticks()
        labels = ['%0.2f ppm' % (label * 1e6) for label in labels]
        ax.set_yticklabels(labels)
        ax.set_xlabel("m/z", fontsize=16)
        ax.set_ylabel("Mass Accuracy", fontsize=16)
        return ax

    def __repr__(self):
        return "{self.__class__.__name__}({self.spectrum_match})".format(
            self=self)


class TidySpectrumMatchAnnotator(SpectrumMatchAnnotator):
    def label_peak(self, fragment, peak, fontsize=12, rotation=90, **kw):
        min_intensity = 0.02 * (self.upper / 1.35)
        if fragment.series == 'oxonium_ion':
            if peak.intensity < min_intensity:
                return
        super(TidySpectrumMatchAnnotator, self).label_peak(fragment, peak, fontsize, rotation, **kw)
