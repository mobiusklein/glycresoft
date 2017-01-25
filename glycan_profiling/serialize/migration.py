from collections import defaultdict

from glycan_profiling.task import TaskBase
from glycan_profiling.serialize.hypothesis import GlycanComposition
from glycan_profiling.serialize import DatabaseBoundOperation
from glycan_profiling.database.builder.glycan.migrate import (
    GlycanHypothesisMigrator)
from glycan_profiling.serialize.serializer import AnalysisSerializer

from ms_deisotope.output.db import (
    SampleRun, MSScan, DeconvolutedPeak, PrecursorInformation)


def fetch_scans_used_in_chromatogram(chromatogram_set, extractor):
    scan_ids = set()
    for chroma in chromatogram_set:
        scan_ids.update(chroma.scan_ids)
    scans = []
    for scan_id in scan_ids:
        scans.append(extractor.peak_loader.get_scan_header_by_id(scan_id))
    return sorted(scans, key=lambda x: x.index)


def fetch_peaks_used_in_chromatograms(chromatogram_set, extractor):
    scan_peak_map = defaultdict(set)
    for chroma in chromatogram_set:
        for node in chroma:
            for peak in node.peaks:
                scan_peak_map[node.scan_id].add(peak)
    return scan_peak_map


def fetch_glycan_compositions_from_chromatograms(chromatogram_set, glycan_db):
    ids = set()
    for chroma in chromatogram_set:
        if chroma.composition:
            ids.add(chroma.composition.id)
    glycan_compositions = []
    for glycan_id in ids:
        gc = glycan_db.query(GlycanComposition).get(glycan_id)
        glycan_compositions.append(gc)
    return glycan_compositions


def update_glycan_chromatogram_composition_ids(hypothesis_migration, glycan_chromatograms):
    mapping = dict()
    for chromatogram in glycan_chromatograms:
        if chromatogram.composition is None:
            continue
        mapping[chromatogram.composition.id] = chromatogram.composition
    for key, value in mapping.items():
        value.id = hypothesis_migration.glycan_composition_id_map[value.id]


class SampleMigrator(DatabaseBoundOperation, TaskBase):
    def __init__(self, connection):
        DatabaseBoundOperation.__init__(self, connection)
        self.sample_run_id = None
        self.ms_scan_id_map = dict()
        self.peak_id_map = dict()

    def _migrate(self, obj):
        self.session.add(obj)
        self.session.flush()
        new_id = obj.id
        return new_id

    def commit(self):
        self.session.commit()

    def migrate_sample_run(self, sample_run):
        new_sample_run = SampleRun(
            name=sample_run.name,
            uuid=sample_run.uuid,
            sample_type=sample_run.sample_type,
            completed=sample_run.completed
        )
        new_id = self._migrate(new_sample_run)
        self.sample_run_id = new_id

    def scan_id(self, ms_scan):
        try:
            scan_id = ms_scan.scan_id
        except:
            scan_id = ms_scan.id
        return scan_id

    def peak_id(self, peak, scan_id):
        try:
            if peak.id:
                peak_id = (scan_id, peak.convert())
        except AttributeError:
            peak_id = (scan_id, peak)
        return peak_id

    def migrate_precursor_information(self, prec_info):
        prec_id = self.scan_id(prec_info.precursor)
        prod_id = self.scan_id(prec_info.product)
        new_info = PrecursorInformation(
            sample_run_id=self.sample_run_id,
            precursor_id=self.ms_scan_id_map[prec_id],
            product_id=self.ms_scan_id_map[prod_id],
            neutral_mass=prec_info.neutral_mass,
            charge=prec_info.charge,
            intensity=prec_info.intensity)
        return self._migrate(new_info)

    def migrate_ms_scan(self, ms_scan):
        scan_id = self.scan_id(ms_scan)
        new_scan = MSScan._serialize_scan(ms_scan, self.sample_run_id)
        try:
            new_scan.info.update(dict(ms_scan.info))
        except AttributeError:
            pass
        new_id = self._migrate(new_scan)
        self.ms_scan_id_map[scan_id] = new_id
        if ms_scan.precursor_information:
            self.migrate_precursor_information(
                ms_scan.precursor_information)

    def migrate_peak(self, peak, scan_id):
        new_scan_id = self.ms_scan_id_map[scan_id]
        new_peak = DeconvolutedPeak.serialize(peak)
        new_peak.scan_id = new_scan_id
        new_id = self._migrate(new_peak)
        peak_id = self.peak_id(peak, scan_id)
        self.peak_id_map[peak_id] = new_id


class GlycanCompositionChromatogramAnalysisSerializer(DatabaseBoundOperation, TaskBase):
    def __init__(self, connection, analysis_name, sample_run,
                 chromatogram_set, glycan_db,
                 chromatogram_extractor):
        DatabaseBoundOperation.__init__(self, connection)
        self._seed_analysis_name = analysis_name
        self.sample_run = sample_run
        self._analysis_serializer = None
        self._sample_migrator = None
        self._glycan_hypothesis_migrator = None

        self.chromatogram_extractor = chromatogram_extractor
        self.glycan_db = glycan_db
        self.chromatogram_set = chromatogram_set

    @property
    def analysis(self):
        return self._analysis_serializer.analysis

    def migrate_sample(self):
        scans = fetch_scans_used_in_chromatogram(
            self.chromatogram_set, self.chromatogram_extractor)

        self._sample_migrator = SampleMigrator(self._original_connection)
        self._sample_migrator.migrate_sample_run(self.sample_run)

        for scan in sorted(scans, key=lambda x: x.ms_level):
            self._sample_migrator.migrate_ms_scan(scan)

        scan_peak_map = fetch_peaks_used_in_chromatograms(
            self.chromatogram_set, self.chromatogram_extractor)

        for scan_id, peaks in scan_peak_map.items():
            for peak in peaks:
                self._sample_migrator.migrate_peak(peak, scan_id)

        self._sample_migrator.commit()

    def migrate_hypothesis(self):
        self._glycan_hypothesis_migrator = GlycanHypothesisMigrator(
            self._original_connection)
        self._glycan_hypothesis_migrator.migrate_hypothesis(
            self.glycan_db.hypothesis)
        for glycan_composition in fetch_glycan_compositions_from_chromatograms(
                self.chromatogram_set, self.glycan_db):
            self._glycan_hypothesis_migrator.migrate_glycan_composition(
                glycan_composition)
        self._glycan_hypothesis_migrator.commit()

    def set_analysis_type(self, type_string):
        self._analysis_serializer.analysis.analysis_type = type_string
        self._analysis_serializer.session.add(self.analysis)
        self._analysis_serializer.session.commit()

    def set_parameters(self, parameters):
        self._analysis_serializer.analysis.parameters = parameters
        self._analysis_serializer.session.add(self.analysis)
        self._analysis_serializer.commit()

    def create_analysis(self):
        self._analysis_serializer = AnalysisSerializer(
            self._original_connection,
            analysis_name=self._seed_analysis_name,
            sample_run_id=self._sample_migrator.sample_run_id)
        update_glycan_chromatogram_composition_ids(
            self._glycan_hypothesis_migrator,
            self.chromatogram_set)

        self._analysis_serializer.set_peak_lookup_table(self._sample_migrator.peak_id_map)

        n = len(self.chromatogram_set)
        i = 0
        for chroma in self.chromatogram_set:
            i += 1
            if i % 100 == 0:
                self.log("%0.2f%% of Chromatograms Saved (%d/%d)" % (
                    i * 100. / n, i, n))
            if chroma.composition is not None:
                self._analysis_serializer.save_glycan_composition_chromatogram_solution(
                    chroma)
            else:
                self._analysis_serializer.save_unidentified_chromatogram_solution(
                    chroma)
        self._analysis_serializer.commit()

    def run(self):
        self.migrate_hypothesis()
        self.migrate_sample()
        self.create_analysis()
