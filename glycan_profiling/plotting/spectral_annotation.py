from matplotlib import pyplot as plt
from ms_peak_picker.utils import draw_peaklist


default_ion_series_to_color = {
    "y": 'red',
    "b": 'blue',
    'B': 'blue',
    'Y': 'red',
    'oxonium_ion': 'green',
    'stub_glycopeptide': 'goldenrod'
}


class SpectrumMatchAnnotator(object):
    def __init__(self, spectrum_match, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        self.spectrum_match = spectrum_match
        self.ax = ax
        self.upper = max(
            spectrum_match.spectrum, key=lambda x: x.intensity
        ).intensity * 1.2

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
            ha='center', fontsize=fontsize)

    def format_axes(self):
        draw_peaklist([], self.ax, pretty=True)

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

    def draw(self, **kwargs):
        fontsize = kwargs.pop('fontsize', 12)
        rotation = kwargs.pop("rotation", 90)
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        self.draw_all_peaks(**kwargs)
        self.draw_matched_peaks(
            fontsize=fontsize, ion_series_to_color=ion_series_to_color,
            rotation=rotation, **kwargs)
        self.format_axes()
        return self

    def __repr__(self):
        return "{self.__class__.__name__}({self.spectrum_match})".format(
            self=self)
