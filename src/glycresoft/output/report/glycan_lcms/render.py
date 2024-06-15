import os

from glycresoft import serialize
from glycresoft.plotting import summaries, figax, SmoothingChromatogramArtist
from glycresoft.plotting.chromatogram_artist import ChargeSeparatingSmoothingChromatogramArtist
from glycresoft.scoring.utils import logit
from glycresoft.chromatogram_tree import ChromatogramFilter

from jinja2 import Template
from markupsafe import Markup


from glycresoft.output.report.base import (
    svguri_plot, ReportCreatorBase)


def chromatogram_figures(chroma):
    figures = []
    plot = SmoothingChromatogramArtist(
        [chroma], colorizer=lambda *a, **k: 'green', ax=figax()).draw(
        label_function=lambda *a, **k: "", legend=False).ax
    plot.set_title("Aggregated\nExtracted Ion Chromatogram", fontsize=24)
    chroma_svg = svguri_plot(
        plot, bbox_inches='tight', height=5, width=9, svg_width="100%")
    figures.append(chroma_svg)
    if len(chroma.mass_shifts) > 1:
        mass_shifts = list(chroma.mass_shifts)
        labels = {}
        rest = chroma
        for mass_shift in mass_shifts:
            with_mass_shift, rest = rest.bisect_mass_shift(mass_shift)
            labels[mass_shift] = with_mass_shift
        mass_shift_plot = SmoothingChromatogramArtist(
            labels.values(),
            colorizer=lambda *a, **k: 'green', ax=figax()).draw(
            label_function=lambda *a, **k: tuple(a[0].mass_shifts)[0].name,
            legend=False).ax
        mass_shift_plot.set_title(
            "mass_shift-Separated\nExtracted Ion Chromatogram", fontsize=24)
        mass_shift_separation = svguri_plot(
            mass_shift_plot, bbox_inches='tight', height=5, width=9, svg_width="100%")
        figures.append(mass_shift_separation)
    if len(chroma.charge_states) > 1:
        charge_separating_plot = ChargeSeparatingSmoothingChromatogramArtist(
            [chroma], ax=figax()).draw(
            label_function=lambda x, *a, **kw: str(
                tuple(x.charge_states)[0]), legend=False).ax
        charge_separating_plot.set_title(
            "Charge-Separated\nExtracted Ion Chromatogram", fontsize=24)
        charge_separation = svguri_plot(
            charge_separating_plot, bbox_inches='tight', height=5, width=9,
            svg_width="100%")
        figures.append(charge_separation)
    return figures


def chromatogram_link(chromatogram):
    id_string = str(chromatogram.id)
    return Markup("<a href=\"#detail-{0}\">{1}</a>").format(id_string, str(chromatogram.key))


class GlycanChromatogramReportCreator(ReportCreatorBase):
    def __init__(self, database_path, analysis_id, stream=None, threshold=5):
        super(GlycanChromatogramReportCreator, self).__init__(
            database_path, analysis_id, stream)
        self.set_template_loader(os.path.dirname(__file__))
        self.threshold = threshold
        self.glycan_chromatograms = ChromatogramFilter([])
        self.unidentified_chromatograms = ChromatogramFilter([])

    def glycan_link(self, key):
        match = self.glycan_chromatograms.find_key(key)
        if match is not None:
            return chromatogram_link(match)
        match = self.unidentified_chromatograms.find_key(key)
        if match is not None:
            return chromatogram_link(match)
        return None

    def prepare_environment(self):
        super(GlycanChromatogramReportCreator, self).prepare_environment()
        self.env.filters["logit"] = logit
        self.env.filters['chromatogram_figures'] = chromatogram_figures
        self.env.filters['glycan_link'] = self.glycan_link

    def make_template_stream(self):
        template_obj = self.env.get_template("overview.templ")

        ads = serialize.AnalysisDeserializer(
            self.database_connection._original_connection,
            analysis_id=self.analysis_id)

        self.glycan_chromatograms = gcs = ads.load_glycan_composition_chromatograms()
        # und = ads.load_unidentified_chromatograms()
        self.unidentified_chromatograms = und = ChromatogramFilter(
            ads.query(serialize.UnidentifiedChromatogram).filter(
                serialize.UnidentifiedChromatogram.analysis_id == self.analysis_id).all())

        if len(gcs) == 0:
            self.log("No glycan compositions were identified. Skipping report building")
            templ = Template('''
                <html>
                <style>
                body {
                    font-family: sans-serif;
                }
                </style>
                <body>
                    <h3>No glycan compositions were identified</h3>
                </body>
                </html>
                ''')
            return templ.stream()

        summary_plot = summaries.GlycanChromatographySummaryGraphBuilder(
            filter(lambda x: x.score > self.threshold, gcs + und))
        lcms_plot, composition_abundance_plot = summary_plot.draw(min_score=5)

        try:
            lcms_plot.ax.legend_.set_visible(False)
        except AttributeError:
            # The legend may not have been created
            pass
        lcms_plot.ax.set_title("Glycan Composition\nLC-MS Aggregated EICs", fontsize=24)

        fig = lcms_plot.ax.figure
        fig.set_figwidth(fig.get_figwidth() * 2.)
        fig.set_figheight(fig.get_figheight() * 2.)

        composition_abundance_plot.ax.set_title("Glycan Composition\nTotal Abundances", fontsize=24)
        composition_abundance_plot.ax.set_xlabel(
            composition_abundance_plot.ax.get_xlabel(), fontsize=14)

        def resolve_key(key):
            match = gcs.find_key(key)
            if match is None:
                match = und.find_key(key)
            return match

        template_stream = (template_obj.stream(
            analysis=ads.analysis, lcms_plot=svguri_plot(
                lcms_plot.ax, bbox_inches='tight', patchless=True,
                svg_width="100%"),
            composition_abundance_plot=svguri_plot(
                composition_abundance_plot.ax, bbox_inches='tight', patchless=True,
                svg_width="100%"),
            glycan_chromatograms=gcs,
            unidentified_chromatograms=und,
            resolve_key=resolve_key
        ))
        return template_stream
