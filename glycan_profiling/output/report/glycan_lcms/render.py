import os

from glycan_profiling import serialize
from glycan_profiling.plotting import summaries, figax, SmoothingChromatogramArtist
from glycan_profiling.plotting.chromatogram_artist import ChargeSeparatingSmoothingChromatogramArtist
from glycan_profiling.scoring.chromatogram_solution import logit
from glycan_profiling.chromatogram_tree import ChromatogramFilter

from jinja2 import Markup, Template

try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote


from glycan_profiling.output.report.base import (
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
    if len(chroma.adducts) > 1:
        adducts = list(chroma.adducts)
        labels = {}
        rest = chroma
        for adduct in adducts:
            with_adduct, rest = rest.bisect_adduct(adduct)
            labels[adduct] = with_adduct
        adduct_plot = SmoothingChromatogramArtist(
            labels.values(),
            colorizer=lambda *a, **k: 'green', ax=figax()).draw(
            label_function=lambda *a, **k: tuple(a[0].adducts)[0].name,
            legend=False).ax
        adduct_plot.set_title(
            "Adduct-Separated\nExtracted Ion Chromatogram", fontsize=24)
        adduct_separation = svguri_plot(
            adduct_plot, bbox_inches='tight', height=5, width=9, svg_width="100%")
        figures.append(adduct_separation)
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

        lcms_plot.ax.legend_.set_visible(False)
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
