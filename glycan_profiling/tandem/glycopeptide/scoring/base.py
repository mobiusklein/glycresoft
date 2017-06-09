from ...spectrum_matcher_base import SpectrumMatcherBase

try:
    from ms_peak_picker.utils import draw_peaklist
except ImportError:
    def draw_peaklist(*args, **kwargs):
        raise Exception("Could not import draw_peaklist")

try:
    from ...spectrum_annotation import annotate_matched_deconvoluted_peaks
except ImportError:
    def annotate_matched_deconvoluted_peaks(*args, **kwargs):
        raise Exception("Could not import draw_peaklist")


class GlycopeptideSpectrumMatcherBase(SpectrumMatcherBase):
    def annotate(self, ax=None, label_font_size=12, labeler=None, **kwargs):
        ax = draw_peaklist(self.spectrum, alpha=0.3, color='grey', ax=ax, **kwargs)
        try:
            draw_peaklist(self._sanitized_spectrum, color='grey', ax=ax, alpha=0.5, **kwargs)
        except AttributeError:
            pass
        annotate_matched_deconvoluted_peaks(
            self.solution_map.items(), ax, labeler=labeler, fontsize=label_font_size)
        return draw_peaklist(
            sorted(self.solution_map.values(), key=lambda x: x.neutral_mass),
            ax=ax, color='red', **kwargs)
