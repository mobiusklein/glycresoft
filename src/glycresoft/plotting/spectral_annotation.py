from typing import TYPE_CHECKING, List, Optional

import numpy as np

from matplotlib import pyplot as plt, font_manager, axes
from matplotlib import patheffects as path_effects


from six import PY3

from ms_peak_picker.utils import draw_peaklist

from ms_deisotope.data_source.scan.base import ChargeNotProvided
from ms_deisotope.data_source import ProcessedScan

from .sequence_fragment_logo import IonAnnotationGlyphBase, glycopeptide_match_logo, GlycopeptideStubFragmentGlyph, PeptideFragmentGlyph, OxoniumIonGlyph

if TYPE_CHECKING:
    from glycresoft.tandem.spectrum_match.spectrum_match import SpectrumMatcherBase


default_ion_series_to_color = {
    "y": "red",
    "z": "maroon",
    "b": "blue",
    "c": "navy",
    "B": "blue",
    "Y": "red",
    "oxonium_ion": "green",
    "stub_glycopeptide": "goldenrod",
    "precursor": "orange",
}


monosaccharide_to_symbol = {
    "Hex": "\u25CB",
    "HexNAc": "\u25A1",
    "Fuc": "\u25B3",
    "dHex": "\u25B3",
    "HexN": "\u25E9",
    "NeuAc": "\u25C6",
    "NeuGc": "\u25C7",
}


def format_stub_annotation(frag):
    """
    An amusing diversion with unicode, but impractical for
    narrow spectrum plots.
    """
    stack = []
    base = ["Hex", "HexNAc"]
    for k in sorted(frag.glycosylation, key=lambda x: x.mass(), reverse=True):
        if k not in base:
            base.append(k)
    for k in base:
        v = frag.glycosylation[k]
        if not v:
            continue
        stack.append(f" {monosaccharide_to_symbol[k]}{v}")
    stack.append("Pep")
    return "\n".join(stack)


if PY3:
    font_options = font_manager.FontProperties(family="sans serif")
    font_options.set_math_fontfamily("dejavusans")
else:
    font_options = font_manager.FontProperties(family="sans serif")


unicode_superscript = {
    "0": "\u2070",
    "1": "\u00B9",
    "2": "\u00B2",
    "3": "\u00B3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
    "+": "\u207A",
    "-": "\u207B",
}


def encode_superscript(number: int) -> str:
    return "".join([unicode_superscript[c] for c in str(number)])


class SpectrumMatchAnnotator(object):
    usemathtext: bool
    spectrum_match: "SpectrumMatcherBase"
    ax: axes.Axes
    clip_labels: bool
    upper: float
    upshift: float
    peak_labels: List[plt.Text]
    intensity_scale: float
    use_glyphs: bool

    def __init__(
        self,
        spectrum_match: "SpectrumMatcherBase",
        ax: Optional[axes.Axes] = None,
        clip_labels: bool = True,
        usemathtext: bool = False,
        normalize: bool = False,
        use_glyphs=False,
    ):
        if ax is None:
            _, ax = plt.subplots(1)
        self.spectrum_match = spectrum_match
        self.ax = ax
        self.clip_labels = clip_labels
        self.upper = max(spectrum_match.spectrum, key=lambda x: x.intensity).intensity * 1.35
        self.peak_labels = []
        self.upshift = 10
        self.sequence_logo = None
        self.usemathtext = usemathtext
        self.normalize = normalize
        self.intensity_scale = 1.0
        self.use_glyphs = use_glyphs
        if self.use_glyphs:
            self.upper = max(spectrum_match.spectrum, key=lambda x: x.intensity).intensity * 1.0
        if self.normalize:
            self.intensity_scale = max(spectrum_match.spectrum, key=lambda x: x.intensity).intensity
        self.compute_scale()

    def compute_scale(self):
        peaks = self.spectrum_match.deconvoluted_peak_set
        mzs = [p.mz for p in peaks]
        if not mzs:
            self.xscale = 1.0
        else:
            self.xscale = max(mzs) - min(mzs)
        self.yscale = self.upper

    def draw_all_peaks(self, color="grey", **kwargs):
        draw_peaklist(self.spectrum_match.deconvoluted_peak_set, alpha=0.3, color=color, ax=self.ax, **kwargs)
        try:
            draw_peaklist(self.spectrum_match._sanitized_spectrum, color=color, ax=self.ax, alpha=0.5, **kwargs)
        except AttributeError:
            pass

    def add_summary_labels(self, x=0.95, y=0.9):
        prec_purity = self.spectrum_match.scan.annotations.get("precursor purity")
        if prec_purity is not None:
            prec_purity = "%0.2f" % prec_purity
        else:
            prec_purity = "-"

        prec_z = self.spectrum_match.precursor_information.charge
        if prec_z is None or prec_z == ChargeNotProvided:
            prec_z = "-"
        else:
            prec_z = str(prec_z)

        self.ax.text(
            x,
            y,
            "Isolation Purity: %s\nPrec. Z: %s" % (prec_purity, prec_z),
            transform=self.ax.transAxes,
            ha="right",
            color="darkslategrey",
        )

    def label_peak(self, fragment, peak, fontsize=12, rotation=90, **kw):
        label = f"{fragment.name}"
        if peak.charge > 1:
            if self.usemathtext:
                label += f"$^{peak.charge}$"
            else:
                label += encode_superscript(peak.charge)
        y = peak.intensity
        upshift = self.upshift
        if self.use_glyphs:
            y = y + upshift
        else:
            y = min(y + upshift, self.upper * 0.9)

        kw.setdefault("clip_on", self.clip_labels)
        clip_on = kw["clip_on"]

        if self.use_glyphs and fragment.series == "stub_glycopeptide":
            art = GlycopeptideStubFragmentGlyph(
                peak.mz,
                y if not self.normalize else y / self.intensity_scale,
                self.ax,
                fragment,
                self.xscale,
                self.yscale,
                peak.charge,
                size=fontsize * 2,
            )
            art.render()
            self.peak_labels.append(art)
            return art
        elif self.use_glyphs and fragment.series in ("b", "y", "c", "z"):
            art = PeptideFragmentGlyph(
                peak.mz,
                y if not self.normalize else y / self.intensity_scale,
                self.ax,
                fragment,
                self.xscale,
                self.yscale,
                peak.charge,
                size=fontsize * 2,
            )
            art.render()
            self.peak_labels.append(art)
            return art
        elif self.use_glyphs and fragment.series == "oxonium_ion":
            art = OxoniumIonGlyph(
                peak.mz,
                y if not self.normalize else y / self.intensity_scale,
                self.ax,
                fragment,
                self.xscale,
                self.yscale,
                peak.charge,
                size=fontsize * 2
            )
            art.render()
            self.peak_labels.append(art)
            return art
        else:
            text = self.ax.text(
                peak.mz,
                y if not self.normalize else y / self.intensity_scale,
                label,
                rotation=rotation,
                va="bottom",
                ha="center",
                fontsize=fontsize,
                fontproperties=font_options,
                clip_on=clip_on,
                parse_math=self.usemathtext,
            )
            self.peak_labels.append(text)
            return text

    def format_axes(self):
        draw_peaklist([], self.ax, pretty=True)
        bboxes = list(map(lambda x: x.bbox().ymax, filter(lambda x: isinstance(x, IonAnnotationGlyphBase), self.peak_labels)))
        if bboxes:
            upper = max(max(bboxes), self.upper)
        else:
            upper = self.upper

        self.ax.set_ylim(0, upper)

    def draw_matched_peaks(self, color="red", alpha=0.8, fontsize=12, ion_series_to_color=None, **kwargs):
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
            if self.normalize:
                peak = peak.clone()
                peak.intensity / self.intensity_scale
                # peak = (peak.mz, peak.intensity / self.intensity_scale)
            draw_peaklist([peak], alpha=alpha, ax=self.ax, color=peak_color)
            self.label_peak(fragment, peak, fontsize=fontsize, **kwargs)

    def draw_spectrum_graph(self, color="red", alpha=0.8, fontsize=12, **kwargs):
        try:
            graph = self.spectrum_match.spectrum_graph
        except AttributeError:
            return

        paths = graph.longest_paths(limit=100)

        for path in paths:
            for edge in path:
                self.draw_peak_pair((edge.start, edge.end), color, alpha, fontsize, label=edge.annotation, **kwargs)

    def draw_peak_pair(self, pair, color="red", alpha=0.8, fontsize=12, label=None, rotation=45, **kwargs):
        p1, p2 = pair
        self.ax.plot((p1.mz, p2.mz), (p1.intensity, p2.intensity), color=color, alpha=alpha, **kwargs)
        kwargs.setdefault("clip_on", self.clip_labels)
        clip_on = kwargs["clip_on"]
        draw_peaklist(pair, ax=self.ax, alpha=0.4, color="orange")
        if label:
            # pylint: disable=assignment-from-no-return
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
                label = "-".join(map(str, label))
            else:
                label = str(label)
            text = self.ax.text(
                midx, midy, label, fontsize=fontsize, ha="center", va="bottom", rotation=rotation, clip_on=clip_on
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=0.5, foreground="white"),
                    path_effects.Normal(),
                ]
            )

    def draw(self, **kwargs):
        fontsize = kwargs.pop("fontsize", 9)
        rotation = kwargs.pop("rotation", 90)
        clip_labels = kwargs.pop("clip_labels", self.clip_labels)
        self.clip_labels = clip_labels
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        self.draw_all_peaks(**kwargs)
        self.draw_matched_peaks(fontsize=fontsize, ion_series_to_color=ion_series_to_color, rotation=rotation, **kwargs)
        self.draw_spectrum_graph(fontsize=fontsize, rotation=rotation / 2)
        self.format_axes()
        return self

    def add_logo_plot(self, xrel=0.15, yrel=0.8, width=0.67, height=0.13, **kwargs):
        figure = self.ax.figure
        iax = figure.add_axes([xrel, yrel, width, height])
        logo = glycopeptide_match_logo(self.spectrum_match, ax=iax, **kwargs)
        self.sequence_logo = logo
        return logo

    def _draw_mass_accuracy_plot(self, ax, error_tolerance=2e-5, **kwargs):
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        match = self.spectrum_match
        ax.scatter(
            *zip(*[(pp.peak.mz, pp.mass_accuracy()) for pp in match.solution_map]),
            alpha=0.5,
            edgecolor="black",
            color=[ion_series_to_color[pp.fragment.series] for pp in match.solution_map],
        )
        limits = error_tolerance
        ax.set_ylim(-limits, limits)
        xlim = 0, max(match.deconvoluted_peak_set, key=lambda x: x.mz).mz + 100
        ax.hlines(0, *xlim, linestyle="--", color="black", lw=0.75)
        ax.hlines(limits / 2, *xlim, linestyle="--", lw=0.5, color="black")
        ax.hlines(-limits / 2, *xlim, linestyle="--", lw=0.5, color="black")
        ax.set_xlim(*xlim)
        labels = ax.get_yticks()
        labels = ["%0.2g" % (label * 1e6) for label in labels]
        ax.set_yticklabels(labels)
        ax.set_xlabel("m/z", fontsize=10)
        ax.set_ylabel("Mass Accuracy (PPM)", fontsize=10)
        return ax

    def __repr__(self):
        return f"{self.__class__.__name__}({self.spectrum_match})"


class TidySpectrumMatchAnnotator(SpectrumMatchAnnotator):
    def label_peak(self, fragment, peak, fontsize=12, rotation=90, **kw):
        min_intensity = 0.02 * (self.upper / 1.35)
        if fragment.series == "oxonium_ion":
            if peak.intensity < min_intensity:
                return
        super(TidySpectrumMatchAnnotator, self).label_peak(fragment, peak, fontsize, rotation, **kw)


def normalize_scan(scan: ProcessedScan, factor=None):
    scan = scan.copy()
    scan.annotations.pop("peak_label_map", None)
    scan.deconvoluted_peak_set = scan.deconvoluted_peak_set.__class__(p.clone() for p in scan.deconvoluted_peak_set)
    bp = scan.base_peak()
    if factor is None:
        factor = bp.intensity / 1000
    for peak in scan:
        peak.intensity /= factor
    return scan


class MirrorSpectrumAnnotatorFacet(TidySpectrumMatchAnnotator):
    def draw_all_peaks(self, color="black", alpha=0.5, mirror=False, **kwargs):
        draw_peaklist(
            self.spectrum_match.deconvoluted_peak_set
            if not mirror
            else [[p.mz, -p.intensity] for p in self.spectrum_match.deconvoluted_peak_set],
            alpha=0.3,
            color="grey",
            ax=self.ax,
            lw=0.75,
            **kwargs,
        )

    def draw_matched_peaks(self, color="red", alpha=0.8, fontsize=12, ion_series_to_color=None, mirror=False, **kwargs):
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
            draw_peaklist(
                [peak] if not mirror else [(peak.mz, -peak.intensity)], alpha=alpha, ax=self.ax, color=peak_color
            )
            self.label_peak(fragment, peak, fontsize=fontsize, mirror=mirror, **kwargs)

    def base_peak_factor(self):
        return self.spectrum_match.scan.base_peak().intensity / 100

    def label_peak(self, fragment, peak, fontsize=12, rotation=90, mirror=False, **kw):
        label = "%s" % fragment.name
        if fragment.series == "oxonium_ion":
            return ""
        if peak.charge > 1:
            if self.usemathtext:
                label += f"$^{peak.charge}$"
            else:
                label += encode_superscript(peak.charge)
        y = peak.intensity
        upshift = 2
        sign = 1 if y > 0 else -1
        y = min(y + sign * upshift, self.upper * 0.9)

        if mirror:
            y = -y

        kw.setdefault("clip_on", self.clip_labels)
        clip_on = kw["clip_on"]

        text = self.ax.text(
            peak.mz,
            y,
            label,
            rotation=rotation,
            va="bottom" if not mirror else "top",
            ha="center",
            fontsize=fontsize,
            fontproperties=font_options,
            clip_on=clip_on,
        )
        self.peak_labels.append(text)
        return text

    def draw(self, mirror=False, **kwargs):
        fontsize = kwargs.pop("fontsize", 9)
        rotation = kwargs.pop("rotation", 90)
        clip_labels = kwargs.pop("clip_labels", self.clip_labels)
        self.clip_labels = clip_labels
        ion_series_to_color = kwargs.pop("ion_series_to_color", default_ion_series_to_color)
        self.draw_all_peaks(mirror=mirror, **kwargs)
        self.draw_matched_peaks(
            fontsize=fontsize, ion_series_to_color=ion_series_to_color, rotation=rotation, mirror=mirror, **kwargs
        )
        self.draw_spectrum_graph(fontsize=fontsize, rotation=rotation / 2)
        self.format_axes()
        return self


def mirror_spectra(psm_a, psm_b):
    art = MirrorSpectrumAnnotatorFacet(psm_a)
    art.draw()
    reflect = MirrorSpectrumAnnotatorFacet(psm_b, ax=art.ax)
    reflect.draw(mirror=True)
    reflect.ax.set_ylim(-1500, 1200)
    ax = art.ax
    ax.set_yticks(np.arange(-1000, 1000 + 250, 250))
    ax.set_yticklabels(map(lambda x: str(x) + "%", list(range(100, -25, -25)) + list(range(25, 125, 25))))
    return art.ax, art, reflect
