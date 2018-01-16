from glypy.io import iupac, glycoct
from glypy import Glycan

from ms_deisotope.output import mgf, mzml
from ms_deisotope.data_source import ScanBunch

from  sqlalchemy.orm import aliased
from glycan_profiling import serialize
from glycan_profiling.task import TaskBase


def build_partial_subgraph_n_glycan(composition):
    composition = composition.clone()
    if composition["HexNAc"] < 2:
        raise ValueError("Not enough HexNAc present to extract N-glycan core")
    root = iupac.loads("?-D-Glcp2NAc").clone(prop_id=False)
    composition['HexNAc'] -= 1
    b2 = iupac.loads("b-D-Glcp2NAc").clone(prop_id=False)
    root.add_monosaccharide(b2, position=4, child_position=1)
    composition['HexNAc'] -= 1
    if composition['Hex'] < 3:
        raise ValueError("Not enough Hex present to extract N-glycan core")
    composition['Hex'] -= 3
    b3 = iupac.loads("b-D-Manp").clone()
    b2.add_monosaccharide(b3, position=4, child_position=1)
    b4 = iupac.loads("a-D-Manp").clone()
    b3.add_monosaccharide(b4, position=3, child_position=1)
    b5 = iupac.loads("a-D-Manp").clone()
    b3.add_monosaccharide(b5, position=6, child_position=1)
    subgraph = Glycan(root, index_method=None)
    return composition, subgraph


def to_glycoct_partial_n_glycan(composition):
    a, b = build_partial_subgraph_n_glycan(composition)
    writer = glycoct.OrderRespectingGlycoCTWriter(b)
    writer.handle_glycan(writer.structure)
    writer.add_glycan_composition(a)
    return writer.buffer.getvalue()


class GlycoCTCompositionListExporter(TaskBase):
    def __init__(self, outstream, glycan_composition_iterable):
        self.outstream = outstream
        self.glycan_composition_iterable = glycan_composition_iterable

    def run(self):
        for glycan_composition in self.glycan_composition_iterable:
            text = to_glycoct_partial_n_glycan(glycan_composition)
            self.outstream.write(text)
            self.outstream.write("\n")


class AnnotatedMGFSerializer(mgf.MGFSerializer):

    def write_header(self, header_dict):
        super(AnnotatedMGFSerializer, self).write_header(header_dict)
        self.add_parameter("precursor_defaulted", header_dict['defaulted'])
        self.add_parameter("activation_method", header_dict['precursor_activation_method'])
        self.add_parameter("activation_energy", header_dict['precursor_activation_energy'])
        self.add_parameter("analyzers", header_dict.get('analyzers'))
        self.add_parameter("scan_id", header_dict['id'])
        self.add_parameter("precursor_scan_id", header_dict['precursor_scan_id'])
        for key, value in header_dict['annotations'].items():
            self.add_parameter(key, value)


class TrainingMGFExporterBase(TaskBase):
    def __init__(self, outstream, spectrum_match_iterable):
        self.outstream = outstream
        self.spectrum_match_iterable = spectrum_match_iterable

    def prepare_scan(self, spectrum_match):
        scan = spectrum_match.scan
        scan = scan.clone()
        scan.annotations['structure'] = spectrum_match.target
        scan.annotations['ms2_score'] = spectrum_match.score
        try:
            scan.annotations['q_value'] = spectrum_match.q_value
        except AttributeError:
            pass
        return scan

    def run(self):
        writer = AnnotatedMGFSerializer(self.outstream)
        for spectrum_match in self.spectrum_match_iterable:
            scan = self.prepare_scan(spectrum_match)
            writer.save(ScanBunch(None, [scan]))
        writer.complete()


class TrainingMGFExporter(TrainingMGFExporterBase):
    def __init__(self, outstream, spectrum_match_iterable, mzml_path):
        super(TrainingMGFExporter, self).__init__(outstream, spectrum_match_iterable)
        self.scan_loader = mzml.ProcessedMzMLDeserializer(mzml_path)

    def prepare_scan(self, spectrum_match):
        scan = self.scan_loader.get_scan_by_id(spectrum_match.scan.scan_id)
        spectrum_match.scan = scan.clone()
        scan = super(TrainingMGFExporter, self).prepare_scan(spectrum_match)
        return scan

    @classmethod
    def _from_analysis_id_query(cls, database_connection, analysis_id, threshold=None):
        A = aliased(serialize.GlycopeptideSpectrumMatch)
        B = aliased(serialize.GlycopeptideSpectrumMatch)

        qinner = database_connection.query(
            B.id.label('inner_id'), serialize.func.max(B.score).label("inner_score"),
            B.scan_id.label("inner_scan_id")).group_by(B.scan_id).selectable

        q = database_connection.query(A).join(
            qinner, qinner.c.inner_id == A.id and A.score == qinner.inner_score).filter(
            A.analysis_id == analysis_id)
        if threshold is not None:
            q = q.filter(A.score > threshold)
        for gsm in q:
            yield gsm.convert()

    @classmethod
    def from_analysis(cls, database_connection, analysis_id, outstream, scan_loader_path=None, threshold=None):
        if scan_loader_path is None:
            analysis = database_connection.query(serialize.Analysis).get(analysis_id)
            scan_loader_path = analysis.parameters['sample_path']
        iterable = cls._from_analysis_id_query(database_connection, analysis_id, threshold=threshold)
        return cls(outstream, iterable, scan_loader_path)
