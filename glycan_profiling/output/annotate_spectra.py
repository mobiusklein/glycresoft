import os
import logging
import string
import platform

from glycan_profiling import serialize
from glycan_profiling.serialize import (
    Protein, Glycopeptide, IdentifiedGlycopeptide,
    func, MSScan, GlycopeptideSpectrumMatch)

from glycan_profiling.task import TaskBase
from glycan_profiling.serialize import DatabaseBoundOperation

from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.tandem.ref import SpectrumReference
from glycan_profiling.tandem.glycopeptide.scoring import CoverageWeightedBinomialModelTree

from glycan_profiling.plotting import figure
from glycan_profiling.plotting.sequence_fragment_logo import glycopeptide_match_logo
from glycan_profiling.plotting.spectral_annotation import TidySpectrumMatchAnnotator

from ms_deisotope.output.mzml import ProcessedMzMLDeserializer

from matplotlib import pyplot as plt, style
from matplotlib import rcParams as mpl_params


status_logger = logging.getLogger("glycresoft.status")


def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')
    return filename


class SpectrumAnnotatorExport(TaskBase, DatabaseBoundOperation):
    def __init__(self, database_connection, analysis_id, output_path, mzml_path=None):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.analysis_id = analysis_id
        self.mzml_path = mzml_path
        self.output_path = output_path
        self.analysis = self.session.query(serialize.Analysis).get(self.analysis_id)
        self.scan_loader = None
        self._mpl_style = {
            'figure.facecolor': 'white',
            'figure.edgecolor': 'white',
            'font.size': 10,
            'savefig.dpi': 72,
            'figure.subplot.bottom': .125
        }

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
        return self.scan_loader

    def _load_spectrum_matches(self):
        query = self.query(GlycopeptideSpectrumMatch).join(
            GlycopeptideSpectrumMatch.scan).filter(
            GlycopeptideSpectrumMatch.analysis_id == self.analysis_id).order_by(
            MSScan.index)
        return query.all()

    def run(self):
        scan_loader = self._make_scan_loader()
        gpsms = self._load_spectrum_matches()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        n = len(gpsms)
        self.log("%d Spectrum Matches" % (n,))
        for i, gpsm in enumerate(gpsms):
            scan = scan_loader.get_scan_by_id(gpsm.scan.scan_id)
            gpep = gpsm.structure.convert()
            if i % 10 == 0:
                self.log("... %0.2f%%: %s @ %s" % (((i + 1) / float(n) * 100.0), gpep, scan.id))
            with style.context(self._mpl_style):
                fig = figure()
                grid = plt.GridSpec(nrows=5, ncols=1)
                ax1 = fig.add_subplot(grid[1, 0])
                ax2 = fig.add_subplot(grid[2:, 0])
                ax3 = fig.add_subplot(grid[0, 0])
                match = CoverageWeightedBinomialModelTree.evaluate(scan, gpep)
                ax3.text(0, 0.5, (
                    str(match.target) + '\n' + scan.id +
                    '\nscore=%0.3f    q value=%0.3g' % (gpsm.score, gpsm.q_value)), va='center')
                ax3.axis('off')
                match.plot(ax=ax2)
                glycopeptide_match_logo(match, ax=ax1)
                fname = format_filename("%s_%s.pdf" % (scan.id, gpep))
                path = os.path.join(self.output_path, fname)
                abspath = os.path.abspath(path)
                if len(abspath) > 259 and platform.system().lower() == 'windows':
                    abspath = '\\\\?\\' + abspath
                fig.savefig(abspath, bbox_inches='tight')
                plt.close(fig)
