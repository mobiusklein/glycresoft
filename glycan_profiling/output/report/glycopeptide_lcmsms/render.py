import os
import textwrap

from collections import OrderedDict

from glycopeptidepy.structure.glycan import GlycosylationType

from glycan_profiling import serialize
from glycan_profiling.serialize import (
    Protein, Glycopeptide, IdentifiedGlycopeptide,
    func, MSScan, DatabaseBoundOperation)
from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.tandem.ref import SpectrumReference
from glycan_profiling.plotting.glycan_visual_classification import (
    GlycanCompositionClassifierColorizer,
    NGlycanCompositionColorizer)
from glycan_profiling.plotting import (figax, SmoothingChromatogramArtist)
from glycan_profiling.plotting.sequence_fragment_logo import glycopeptide_match_logo
from glycan_profiling.plotting.plot_glycoforms import (
    GlycoformLayout)
from glycan_profiling.plotting.spectral_annotation import TidySpectrumMatchAnnotator
from glycan_profiling.tandem.glycopeptide.identified_structure import IdentifiedGlycoprotein
from glycan_profiling.tandem.glycopeptide.scoring import CoverageWeightedBinomialScorer
from glycan_profiling.plotting.entity_bar_chart import (
    AggregatedAbundanceArtist, BundledGlycanComposition)
from glycan_profiling.output.report.base import (
    svguri_plot, png_plot, ReportCreatorBase)

from ms_deisotope.output.mzml import ProcessedMzMLDeserializer



glycan_colorizer_type_map = {
    GlycosylationType.n_linked: NGlycanCompositionColorizer,
    GlycosylationType.glycosaminoglycan: GlycanCompositionClassifierColorizer({}, 'slateblue'),
    GlycosylationType.o_linked: GlycanCompositionClassifierColorizer({}, 'slateblue')
}


def scale_fix_xml_transform(root):
    view_box_str = root.attrib["viewBox"]
    x_start, y_start, x_end, y_end = map(float, view_box_str.split(" "))
    x_start += 0
    updated_view_box_str = " ".join(map(str, [x_start, y_start, x_end, y_end]))
    root.attrib["viewBox"] = updated_view_box_str
    fig_g = root.find(".//{http://www.w3.org/2000/svg}g[@id=\"figure_1\"]")
    fig_g.attrib["transform"] = "scale(1.0, 1.0)"
    return root


class IdentifiedGlycopeptideDescriberBase(object):
    def __init__(self, database_path, analysis_id, mzml_path=None):
        self.database_connection = DatabaseBoundOperation(database_path)
        self.analysis_id = analysis_id
        self.analysis = self.session.query(serialize.Analysis).get(self.analysis_id)
        self.mzml_path = mzml_path
        self.scan_loader = None
        self._make_scan_loader()

    def spectrum_match_info(self, glycopeptide):
        spectrum_match_ref = glycopeptide.best_spectrum_match
        scan_id = spectrum_match_ref.scan.scan_id
        scan = self.scan_loader.get_scan_by_id(scan_id)
        try:
            mass_shift = spectrum_match_ref.mass_shift
        except Exception:
            mass_shift = Unmodified
        if mass_shift.name != Unmodified.name:
            mass_shift = mass_shift.convert()
        else:
            mass_shift = Unmodified

        match = CoverageWeightedBinomialScorer.evaluate(
            scan,
            glycopeptide.structure.convert(),
            error_tolerance=self.analysis.parameters["fragment_error_tolerance"],
            mass_shift=mass_shift)
        specmatch_artist = TidySpectrumMatchAnnotator(match, ax=figax())
        specmatch_artist.draw(fontsize=10, pretty=True)
        annotated_match_ax = specmatch_artist.ax

        scan_title = scan.id
        if len(scan_title) > 60:
            scan_title = '\n'.join(textwrap.wrap(scan_title, 60))

        annotated_match_ax.set_title(scan_title, fontsize=18)
        annotated_match_ax.set_ylabel(
            annotated_match_ax.get_ylabel(), fontsize=16)
        annotated_match_ax.set_xlabel(
            annotated_match_ax.get_xlabel(), fontsize=16)

        sequence_logo_plot = glycopeptide_match_logo(match, ax=figax())
        xlim = list(sequence_logo_plot.get_xlim())
        xlim[0] += 1

        sequence_logo_plot.set_xlim(xlim[0], xlim[1])

        spectrum_plot = png_plot(
            annotated_match_ax, svg_width="100%", bbox_inches='tight', height=3 * 1.5,
            width=8 * 1.5,
            img_width="100%",
            patchless=True)
        logo_plot = png_plot(
            sequence_logo_plot,
            svg_width="100%",
            img_width="100%",
            xml_transform=scale_fix_xml_transform,
            bbox_inches='tight',
            height=2, width=6 * 1.5, patchless=True)
        return dict(
            spectrum_plot=spectrum_plot, logo_plot=logo_plot,
            precursor_mass_accuracy=match.precursor_mass_accuracy(),
            spectrum_match=match)

    def _make_scan_loader(self):
        if self.mzml_path is not None:
            if not os.path.exists(self.mzml_path):
                raise IOError("No such file {}".format(self.mzml_path))
            self.scan_loader = ProcessedMzMLDeserializer(self.mzml_path)
        else:
            self.mzml_path = self.analysis.parameters['sample_path']
            if not os.path.exists(self.mzml_path):
                raise IOError((
                    "No such file {}. If {} was relocated, you may need to explicily pass the"
                    " corrected file path.").format(
                        self.mzml_path,
                        self.database_connection._original_connection))
            self.scan_loader = ProcessedMzMLDeserializer(self.mzml_path)


class IdentifiedGlycopeptideDescriberWorker(IdentifiedGlycopeptideDescriberBase):

    def __call__(self, glycopeptide_id):
        glycopeptide = self._glycopeptide_from_id(glycopeptide_id)
        return self.spectrum_match_info(glycopeptide)

    def _glycopeptide_from_id(self, glycopeptide_id):
        return self.database_connection.query(
            IdentifiedGlycopeptide).get(glycopeptide_id)


class GlycopeptideDatabaseSearchReportCreator(ReportCreatorBase, IdentifiedGlycopeptideDescriberBase):
    def __init__(self, database_path, analysis_id, stream=None, threshold=5,
                 mzml_path=None):
        super(GlycopeptideDatabaseSearchReportCreator, self).__init__(
            database_path, analysis_id, stream)
        self.set_template_loader(os.path.dirname(__file__))
        self.mzml_path = mzml_path
        self.scan_loader = None
        self.threshold = threshold
        self.use_dynamic_display_mode = 0
        self.analysis = self.session.query(serialize.Analysis).get(self.analysis_id)
        self._resolve_hypothesis_id()
        self._build_protein_index()
        self._make_scan_loader()
        self._glycopeptide_counter = 0
        if len(self.protein_index) > 10:
            self.use_dynamic_display_mode = 1

    def _spawn(self):
        return IdentifiedGlycopeptideDescriberWorker(self.database_connection, self.analysis_id, self.mzml_path)

    def _resolve_hypothesis_id(self):
        self.hypothesis_id = self.analysis.hypothesis_id
        hypothesis = self.session.query(serialize.GlycopeptideHypothesis).get(self.hypothesis_id)
        if hypothesis is None:
            self.hypothesis_id = 1
            hypothesis = self.session.query(serialize.GlycopeptideHypothesis).get(
                self.hypothesis_id)
            if hypothesis is None:
                raise ValueError("Could not resolve Glycopeptide Hypothesis!")

    def prepare_environment(self):
        super(GlycopeptideDatabaseSearchReportCreator, self).prepare_environment()

    def _build_protein_index(self):
        hypothesis_id = self.hypothesis_id
        theoretical_counts = self.session.query(Protein.name, Protein.id, func.count(Glycopeptide.id)).join(
            Glycopeptide).group_by(Protein.id).filter(
            Protein.hypothesis_id == hypothesis_id).all()
        matched_counts = self.session.query(
            Protein.name, Protein.id, func.count(IdentifiedGlycopeptide.id)).join(Protein.glycopeptides).join(
            IdentifiedGlycopeptide, IdentifiedGlycopeptide.structure_id == Glycopeptide.id).group_by(Protein.id).filter(
            IdentifiedGlycopeptide.ms2_score > self.threshold,
            IdentifiedGlycopeptide.analysis_id == self.analysis_id).all()
        listing = []
        index = {}
        for protein_name, protein_id, glycopeptide_count in theoretical_counts:
            index[protein_id] = {
                "protein_name": protein_name,
                "protein_id": protein_id,
            }
        for protein_name, protein_id, glycopeptide_count in matched_counts:
            entry = index[protein_id]
            entry['identified_glycopeptide_count'] = glycopeptide_count
            listing.append(entry)
        self.protein_index = sorted(listing, key=lambda x: x["identified_glycopeptide_count"], reverse=True)
        for protein_entry in self.protein_index:
            protein_entry['protein'] = self.session.query(Protein).get(protein_entry["protein_id"])
        return self.protein_index

    def iterglycoproteins(self):
        n = float(len(self.protein_index))
        for i, row in enumerate(self.protein_index, 1):
            protein = row['protein']
            glycopeptides = self.session.query(
                IdentifiedGlycopeptide).join(Glycopeptide).join(
                Protein).filter(
                    IdentifiedGlycopeptide.analysis_id == self.analysis_id,
                    Glycopeptide.hypothesis_id == self.hypothesis_id,
                    IdentifiedGlycopeptide.ms2_score > self.threshold,
                    Protein.id == protein.id).all()
            glycoprotein = IdentifiedGlycoprotein(protein, glycopeptides)
            self.status_update(
                "Processing %s (%d/%d) %0.2f%%" % (
                    protein.name, i, n, (i / n * 100)))
            yield i, glycoprotein

    def site_specific_abundance_plots(self, glycoprotein):
        axes = OrderedDict()
        for glyco_type in glycoprotein.glycosylation_types:
            for site in sorted(glycoprotein.glycosylation_sites_for(glyco_type)):
                spanning_site = glycoprotein.site_map[glyco_type][site]
                if len(spanning_site) == 0:
                    continue
                bundle = BundledGlycanComposition.aggregate(spanning_site)
                if len(bundle) == 0:
                    continue
                ax = figax()
                AggregatedAbundanceArtist(
                    bundle, ax=ax, colorizer=glycan_colorizer_type_map[glyco_type]).draw()
                ax.set_title("%s Glycans\nat Site %d" % (glyco_type.name, site + 1,), fontsize=18)
                axes[site, glyco_type] = svguri_plot(ax, bbox_inches='tight')
        return axes

    def draw_glycoforms(self, glycoprotein):
        ax = figax()
        layout = GlycoformLayout(glycoprotein, glycoprotein.identified_glycopeptides, ax=ax)
        layout.draw()
        svg = layout.to_svg(scale=2.0, height_padding_scale=1.1)
        return svg

    def chromatogram_plot(self, glycopeptide):
        ax = figax()
        try:
            SmoothingChromatogramArtist(
                glycopeptide, ax=ax, label_peaks=False,
                colorizer=lambda x: "#48afd0").draw(legend=False)
            ax.set_xlabel("Time (Minutes)", fontsize=16)
            ax.set_ylabel("Relative Abundance", fontsize=16)
            return png_plot(ax, bbox_inches='tight', img_height='100%')
        except ValueError:
            return "<div style='text-align:center;'>No Chromatogram Found</div>"

    def track_entry(self, glycopeptide):
        self._glycopeptide_counter += 1
        if self._glycopeptide_counter % 15 == 0:
            self.status_update(
                " ... %d glycopeptides handled" % (self._glycopeptide_counter,))
        return self._glycopeptide_counter

    def make_template_stream(self):
        template_obj = self.env.get_template("overview.templ")

        ads = serialize.AnalysisDeserializer(
            self.database_connection._original_connection,
            analysis_id=self.analysis_id)

        hypothesis = ads.analysis.hypothesis
        sample_run = ads.analysis.sample_run
        if self.use_dynamic_display_mode:
            self.status_update("Using dynamic display mode")
        template_stream = template_obj.stream(
            analysis=ads.analysis,
            hypothesis=hypothesis,
            sample_run=sample_run,
            protein_index=self.protein_index,
            glycoprotein_iterator=self.iterglycoproteins(),
            renderer=self,
            use_dynamic_display_mode=self.use_dynamic_display_mode)

        return template_stream
