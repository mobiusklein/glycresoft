import time

from datetime import timedelta
from collections import defaultdict
from typing import DefaultDict, List, Optional, Sequence, Set, Tuple

from glycresoft.structure.scan import ScanInformation
from glycresoft.trace.extract import ChromatogramExtractor

from glypy.composition import formula

from glycresoft.task import TaskBase
from glycresoft.chromatogram_tree import (
    ChromatogramFilter,
    DisjointChromatogramSet,
    Chromatogram
)

from glycresoft.database.builder.glycan.migrate import (
    GlycanHypothesisMigrator)

from glycresoft.database.builder.glycopeptide.migrate import (
    GlycopeptideHypothesisMigrator,
    Glycopeptide,
    Peptide,
    Protein,
    GlycanCombination)

from glycresoft.serialize import (
    AnalysisTypeEnum,
    ChromatogramSolutionMassShiftedToChromatogramSolution)

from glycresoft.tandem.glycopeptide.identified_structure import (
    IdentifiedGlycopeptide as MemoryIdentifiedGlycopeptide)
from glycresoft.scoring.chromatogram_solution import ChromatogramSolution

from glycresoft.serialize.utils import toggle_indices
from glycresoft.serialize.hypothesis import GlycanComposition
from glycresoft.serialize import DatabaseBoundOperation
from glycresoft.serialize.serializer import AnalysisSerializer

from glycresoft.serialize.spectrum import (
    SampleRun,
    MSScan,
    DeconvolutedPeak,
    PrecursorInformation)

from ms_deisotope.data_source import ChargeNotProvided
from ms_deisotope.output.common import SampleRun as MemorySampleRun

def slurp(session, model, ids):
    ids = sorted(ids)
    total = len(ids)
    last = 0
    step = 100
    results = []
    while last < total:
        results.extend(session.query(model).filter(
            model.id.in_(ids[last:last + step])))
        last += step
    return results


def fetch_scans_used_in_chromatogram(chromatogram_set: List[Chromatogram],
                                     extractor: ChromatogramExtractor) -> List[ScanInformation]:
    scan_ids = set()
    for chroma in chromatogram_set:
        scan_ids.update(chroma.scan_ids)

        tandem_solutions = getattr(chroma, "tandem_solutions", [])
        for tsm in tandem_solutions:
            scan_ids.add(tsm.scan.id)
            scan_ids.add(tsm.scan.precursor_information.precursor_scan_id)

    scans = []
    with extractor.toggle_peak_loading():
        for scan_id in scan_ids:
            scans.append(
                ScanInformation.from_scan(extractor.get_scan_header_by_id(scan_id))
            )
    return sorted(scans, key=lambda x: (x.ms_level, x.index))


def fetch_glycan_compositions_from_chromatograms(chromatogram_set: Sequence[ChromatogramSolution],
                                                 glycan_db: DatabaseBoundOperation) -> List[GlycanComposition]:
    ids = set()
    for chroma in chromatogram_set:
        if chroma.composition:
            ids.add(chroma.composition.id)
    glycan_compositions = []
    for glycan_id in ids:
        gc = glycan_db.query(GlycanComposition).get(glycan_id)
        glycan_compositions.append(gc)
    return glycan_compositions


def update_glycan_chromatogram_composition_ids(hypothesis_migration,
                                               glycan_chromatograms: Sequence[ChromatogramSolution]):
    mapping = dict()
    tandem_mapping = defaultdict(list)
    for chromatogram in glycan_chromatograms:
        if chromatogram.composition is None:
            continue
        mapping[chromatogram.composition.id] = chromatogram.composition
        for match in getattr(chromatogram, 'tandem_solutions', []):
            tandem_mapping[match.target.id].append(match.target)

    for key, value in mapping.items():
        value.id = hypothesis_migration.glycan_composition_id_map[value.id]
    for key, group in tandem_mapping.items():
        for value in group:
            value.id = hypothesis_migration.glycan_composition_id_map[key]


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

    def clear(self):
        self.peak_id_map.clear()
        self.ms_scan_id_map.clear()

    def migrate_sample_run(self, sample_run):
        new_sample_run = SampleRun(
            name=sample_run.name,
            uuid=sample_run.uuid,
            sample_type=sample_run.sample_type,
            completed=sample_run.completed
        )
        new_id = self._migrate(new_sample_run)
        self.sample_run_id = new_id

    def scan_id(self, ms_scan) -> str:
        if ms_scan is None:
            return None
        try:
            scan_id = ms_scan.scan_id
        except AttributeError:
            scan_id = ms_scan.id
        return scan_id

    def peak_id(self, peak, scan_id) -> Tuple[str, DeconvolutedPeak]:
        try:
            if peak.id:
                peak_id = (scan_id, peak.convert())
        except AttributeError:
            peak_id = (scan_id, peak)
        return peak_id

    def migrate_precursor_information(self, prec_info):
        try:
            # prec_id = self.scan_id(prec_info.precursor)
            prec_id = prec_info.precursor_scan_id
        except KeyError as err:
            prec_id = None
            self.log("Unable to locate precursor scan with ID %r" % (err , ))
        # prod_id = self.scan_id(prec_info.product)
        prod_id = prec_info.product_scan_id
        charge = prec_info.charge
        if charge == ChargeNotProvided:
            charge = 0
        new_info = PrecursorInformation(
            sample_run_id=self.sample_run_id,
            # precursor may be missing
            precursor_id=self.ms_scan_id_map.get(prec_id),
            product_id=self.ms_scan_id_map[prod_id],
            neutral_mass=prec_info.neutral_mass,
            charge=charge,
            intensity=prec_info.intensity)
        return self._migrate(new_info)

    def migrate_ms_scan(self, ms_scan):
        if ms_scan is None:
            return
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


class AnalysisMigrationBase(DatabaseBoundOperation, TaskBase):
    sample_run: MemorySampleRun
    chromatogram_extractor: ChromatogramExtractor

    _seed_analysis_name: str
    _analysis_serializer: Optional[AnalysisSerializer]
    _sample_migrator: Optional[SampleMigrator]

    def __init__(self, connection: str, analysis_name: str,
                 sample_run: MemorySampleRun,
                 chromatogram_extractor: ChromatogramExtractor):
        DatabaseBoundOperation.__init__(self, connection)

        self.sample_run = sample_run
        self.chromatogram_extractor = chromatogram_extractor

        self._seed_analysis_name = analysis_name
        self._analysis_serializer = None
        self._sample_migrator = None

    @property
    def analysis(self):
        return self._analysis_serializer.analysis

    def set_analysis_type(self, type_string: str):
        self._analysis_serializer.analysis.analysis_type = type_string
        self._analysis_serializer.session.add(self.analysis)
        self._analysis_serializer.session.commit()

    def set_parameters(self, parameters: dict):
        self._analysis_serializer.analysis.parameters = parameters
        self._analysis_serializer.session.add(self.analysis)
        self._analysis_serializer.commit()

    def fetch_peaks(self, chromatogram_set: Sequence[Chromatogram]) -> DefaultDict[str, Set[DeconvolutedPeak]]:
        scan_peak_map = defaultdict(set)
        for chroma in chromatogram_set:
            if chroma is None:
                continue
            for node in chroma:
                for peak in node.peaks:
                    scan_peak_map[node.scan_id].add(peak)
        return scan_peak_map

    def fetch_scans(self) -> List[ScanInformation]:
        raise NotImplementedError()

    def migrate_sample(self):
        self.log("... Loading Scans")
        scans = self.fetch_scans()
        self._sample_migrator = SampleMigrator(self._original_connection)
        self._sample_migrator.migrate_sample_run(self.sample_run)

        self.log("... Migrating Scans")
        for scan in sorted(scans, key=lambda x: x.ms_level):
            self._sample_migrator.migrate_ms_scan(scan)

        scan_peak_map = self.fetch_peaks(self.chromatogram_set)
        self.log("... Migrating Peaks")
        for scan_id, peaks in scan_peak_map.items():
            for peak in peaks:
                self._sample_migrator.migrate_peak(peak, scan_id)

        self._sample_migrator.commit()

    def migrate_hypothesis(self):
        raise NotImplementedError()

    def create_analysis(self):
        raise NotImplementedError()

    def run(self):
        self.log("Migrating Hypothesis")
        self.migrate_hypothesis()
        self.log("Migrating Sample Run")
        self.migrate_sample()
        self.log("Creating Analysis Record")
        self.create_analysis()


class GlycanCompositionChromatogramAnalysisSerializer(AnalysisMigrationBase):
    _glycan_hypothesis_migrator: Optional[GlycanHypothesisMigrator]
    glycan_db: DatabaseBoundOperation
    chromatogram_set: ChromatogramFilter

    def __init__(self, connection, analysis_name, sample_run,
                 chromatogram_set, glycan_db,
                 chromatogram_extractor):
        AnalysisMigrationBase.__init__(
            self, connection, analysis_name, sample_run,
            chromatogram_extractor)
        self._glycan_hypothesis_migrator = None

        self.glycan_db = glycan_db
        self.chromatogram_set = ChromatogramFilter(chromatogram_set)
        self._index_chromatogram_set()

    def _index_chromatogram_set(self):
        for i, chroma in enumerate(self.chromatogram_set):
            chroma.id = i

    @property
    def analysis(self):
        return self._analysis_serializer.analysis

    def fetch_scans(self):
        scans = fetch_scans_used_in_chromatogram(
            self.chromatogram_set, self.chromatogram_extractor)
        return scans

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

    def locate_mass_shift_reference_solution(self, chromatogram):
        if chromatogram.used_as_mass_shift:
            for key, mass_shift in chromatogram.used_as_mass_shift:
                disjoint_set_for_key = self.chromatogram_set.key_map[key]
                if not isinstance(disjoint_set_for_key, DisjointChromatogramSet):
                    disjoint_set_for_key = DisjointChromatogramSet(disjoint_set_for_key)
                hit = disjoint_set_for_key.find_overlap(chromatogram)
                if hit is not None:
                    yield hit, mass_shift

    def _get_solution_db_id_by_reference(self, ref_id):
        return self._analysis_serializer._chromatogram_solution_id_map[ref_id]

    def _get_mass_shift_id(self, mass_shift):
        return self._analysis_serializer._mass_shift_cache.serialize(mass_shift).id

    def express_mass_shift_relation(self, chromatogram: Chromatogram):
        try:
            source_id = self._get_solution_db_id_by_reference(chromatogram.id)
        except KeyError as e:
            self.error(
                    f"Chromatogram not registered: {e} -> {chromatogram}")
        seen = set()
        for related_solution, mass_shift in self.locate_mass_shift_reference_solution(
                chromatogram):
            if related_solution.id in seen:
                continue
            else:
                seen.add(related_solution.id)
            try:
                referenced_id = self._get_solution_db_id_by_reference(related_solution.id)
                mass_shift_id = self._get_mass_shift_id(mass_shift)
                self._analysis_serializer.session.execute(
                    ChromatogramSolutionMassShiftedToChromatogramSolution.__table__.insert(),
                    {
                        "mass_shifted_solution_id": source_id,
                        "owning_solution_id": referenced_id,
                        "mass_shift_id": mass_shift_id
                    })
            except KeyError as e:
                self.error(
                    f"Failed to express mass shift relation: {e} -> {mass_shift} @ {chromatogram}")
                continue

    def create_analysis(self):
        self._analysis_serializer = AnalysisSerializer(
            self._original_connection,
            analysis_name=self._seed_analysis_name,
            sample_run_id=self._sample_migrator.sample_run_id)

        self._analysis_serializer.set_analysis_type(AnalysisTypeEnum.glycan_lc_ms.name)

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

        for chroma in self.chromatogram_set:
            if chroma.chromatogram.used_as_mass_shift:
                self.express_mass_shift_relation(chroma)

        self._analysis_serializer.commit()


class GlycopeptideMSMSAnalysisSerializer(AnalysisMigrationBase):


    _glycopeptide_hypothesis_migrator: GlycopeptideHypothesisMigrator
    _glycopeptide_ids: Set[int]

    _identified_glycopeptide_set: List[MemoryIdentifiedGlycopeptide]
    _unassigned_chromatogram_set: List[Chromatogram]


    def __init__(self, connection, analysis_name, sample_run,
                 identified_glycopeptide_set, unassigned_chromatogram_set,
                 glycopeptide_db, chromatogram_extractor):
        AnalysisMigrationBase.__init__(
            self, connection, analysis_name, sample_run, chromatogram_extractor)

        self._glycopeptide_hypothesis_migrator = None

        self.glycopeptide_db = glycopeptide_db
        self._identified_glycopeptide_set = identified_glycopeptide_set

        self._unassigned_chromatogram_set = unassigned_chromatogram_set
        self._glycopeptide_ids = None

        self.aggregate_glycopeptide_ids()

    def aggregate_glycopeptide_ids(self):
        """Accumulate a set of all unique structure IDs over all spectrum matches.

        Returns
        -------
        set
        """
        aggregate = set()
        for gp in self._identified_glycopeptide_set:
            for solution_set in gp.spectrum_matches:
                for match in solution_set:
                    aggregate.add(match.target.id)
        self._glycopeptide_ids = aggregate

    def update_glycopeptide_ids(self):
        '''Build a mapping from searched hypothesis structure ID to saved subset structure
        ID, ensuring that each object has its new ID assigned exactly once.

        The use of :func:`id` here is to ensure that exactness.
        '''
        hypothesis_migration = self._glycopeptide_hypothesis_migrator
        identified_glycopeptide_set = self._identified_glycopeptide_set
        mapping = defaultdict(dict)
        for glycopeptide in identified_glycopeptide_set:
            mapping[glycopeptide.structure.id][id(
                glycopeptide.structure)] = glycopeptide.structure
            for solution_set in glycopeptide.spectrum_matches:
                for match in solution_set:
                    mapping[match.target.id][id(match.target)] = match.target
        for key, instances in mapping.items():
            for value in instances.values():
                try:
                    value.id = hypothesis_migration.glycopeptide_id_map[value.id]
                except KeyError:
                    self.log(f"Could not find a mapping from ID {value.id} for {value} "
                              "in the new keyspace, reconstructing to recover...")
                    value.id = self._migrate_single_glycopeptide(value)

    @property
    def chromatogram_set(self):
        return [
            gp.chromatogram for gp in self._identified_glycopeptide_set
        ] + list(self._unassigned_chromatogram_set)

    def migrate_hypothesis(self):
        start = time.time()
        self._glycopeptide_hypothesis_migrator = GlycopeptideHypothesisMigrator(
            self._original_connection)
        self._glycopeptide_hypothesis_migrator.migrate_hypothesis(
            self.glycopeptide_db.hypothesis)

        self.log("... Migrating Glycans")
        glycan_compositions, glycan_combinations = self.fetch_glycan_compositions(
            self._glycopeptide_ids)

        for i, gc in enumerate(glycan_compositions):
            if i % 1000 == 0 and i:
                self.log("...... Migrating Glycan %d" % (i, ))
            self._glycopeptide_hypothesis_migrator.migrate_glycan_composition(gc)
        for i, gc in enumerate(glycan_combinations):
            if i % 1000 == 0 and i:
                self.log("...... Migrating Glycan Combination %d" % (i, ))
            self._glycopeptide_hypothesis_migrator.migrate_glycan_combination(gc)

        self.log("... Loading Proteins and Peptides from Glycopeptides")
        peptides = self.fetch_peptides(self._glycopeptide_ids)
        proteins = self.fetch_proteins(peptides)
        self.log("... Migrating Proteins (%d)" % (len(proteins), ))
        n = len(proteins)
        index_toggler = toggle_indices(self._glycopeptide_hypothesis_migrator.session, Protein)
        index_toggler.drop()
        for i, protein in enumerate(proteins):
            if i % 5000 == 0 and i:
                self.log("...... Migrating Protein %d/%d (%0.2f%%) %s" % (i, n, i * 100.0 / n, protein.name))
            self._glycopeptide_hypothesis_migrator.migrate_protein(protein)
        proteins = []
        index_toggler.create()
        index_toggler = toggle_indices(
            self._glycopeptide_hypothesis_migrator.session, Peptide)
        index_toggler.drop()
        n = len(peptides)
        self.log("... Migrating Peptides (%d)" % (n, ))
        self._glycopeptide_hypothesis_migrator.migrate_peptides_bulk(peptides)
        index_toggler.create()
        peptides = []

        glycopeptides = self.fetch_glycopeptides(self._glycopeptide_ids)

        index_toggler = toggle_indices(self._glycopeptide_hypothesis_migrator.session, Glycopeptide)
        index_toggler.drop()
        n = len(glycopeptides)
        self.log("... Migrating Glycopeptides (%d)" % (n, ))
        self._glycopeptide_hypothesis_migrator.migrate_glycopeptide_bulk(glycopeptides)
        index_toggler.create()
        self._glycopeptide_hypothesis_migrator.commit()
        elapsed = timedelta(seconds=time.time()) - timedelta(seconds=start)
        self.log(f"... Hypothesis migration time {elapsed}")

    def fetch_scans(self):
        scan_ids = set()
        precursor_ids = dict()
        with self.chromatogram_extractor.toggle_peak_loading():
            for glycopeptide in self._identified_glycopeptide_set:
                if glycopeptide.chromatogram is not None:
                    scan_ids.update(glycopeptide.chromatogram.scan_ids)
                for msms in glycopeptide.spectrum_matches:
                    scan_ids.add(msms.scan.id)
                    try:
                        scan_ids.add(precursor_ids[msms.scan.id])
                    except KeyError:
                        precursor_ids[msms.scan.id] = self.chromatogram_extractor.get_scan_header_by_id(
                            msms.scan.id).precursor_information.precursor_scan_id
                        scan_ids.add(precursor_ids[msms.scan.id])
            scans = []
            for scan_id in scan_ids:
                if scan_id is not None:
                    try:
                        scans.append(ScanInformation.from_scan(self.chromatogram_extractor.get_scan_header_by_id(scan_id)))
                    except KeyError:
                        self.log("Could not locate scan for ID %r" % (scan_id, ))
        return sorted(scans, key=lambda x: (x.ms_level, x.index))

    def _get_glycan_combination_for_glycopeptide(self, glycopeptide_id: int) -> int:
        gc_comb_id = self.glycopeptide_db.query(
            Glycopeptide.glycan_combination_id).filter(
            Glycopeptide.id == glycopeptide_id)
        return gc_comb_id[0]

    def fetch_glycan_compositions(self, glycopeptide_ids: Set[int]) -> Tuple[List[GlycanComposition],
                                                                             List[GlycanCombination]]:
        glycan_compositions = dict()
        glycan_combinations = dict()

        glycan_combination_ids = set()
        for glycopeptide_id in glycopeptide_ids:
            gc_comb_id = self._get_glycan_combination_for_glycopeptide(glycopeptide_id)
            glycan_combination_ids.add(gc_comb_id)

        for i in glycan_combination_ids:
            gc_comb = self.glycopeptide_db.query(GlycanCombination).get(i)
            glycan_combinations[gc_comb.id] = gc_comb

            for composition in gc_comb.components:
                glycan_compositions[composition.id] = composition

        return list(glycan_compositions.values()), list(glycan_combinations.values())

    def _get_peptide_id_for_glycopeptide(self, glycopeptide_id: int) -> int:
        db_glycopeptide = self.glycopeptide_db.query(Glycopeptide).get(glycopeptide_id)
        return db_glycopeptide.peptide_id

    def fetch_peptides(self, glycopeptide_ids: Set[int]) -> List[Peptide]:
        peptides = set()
        for glycopeptide_id in glycopeptide_ids:
            peptide_id = self._get_peptide_id_for_glycopeptide(glycopeptide_id)
            peptides.add(peptide_id)

        return slurp(self.glycopeptide_db, Peptide, peptides)

    def fetch_proteins(self, peptide_set: List[Peptide]) -> List[Protein]:
        proteins = set()
        for peptide in peptide_set:
            proteins.add(peptide.protein_id)
        return slurp(self.glycopeptide_db, Protein, proteins)

    def fetch_glycopeptides(self, glycopeptide_ids: Set[int]) -> List[Glycopeptide]:
        return slurp(self.glycopeptide_db, Glycopeptide, glycopeptide_ids)

    def _migrate_single_glycopeptide(self, glycopeptide: Glycopeptide) -> int:
        """
        Stupendously slow if used in bulk, intended for recovering from cases where
        a single glycopeptide somehow slipped through the cracks.
        """
        gp = self.glycopeptide_db.query(Glycopeptide).get(id)
        self._glycopeptide_hypothesis_migrator.migrate_glycopeptide(gp)
        return self._glycopeptide_hypothesis_migrator.glycopeptide_id_map[glycopeptide.id]

    def create_analysis(self):
        self._analysis_serializer = AnalysisSerializer(
            self._original_connection,
            analysis_name=self._seed_analysis_name,
            sample_run_id=self._sample_migrator.sample_run_id)

        self._analysis_serializer.set_analysis_type(
            AnalysisTypeEnum.glycopeptide_lc_msms.name)

        self._analysis_serializer.set_peak_lookup_table(
            self._sample_migrator.peak_id_map)

        # Patch the in-memory objects to be mapped through to the new database keyspace
        self.update_glycopeptide_ids()

        # Free up memory from the glycopeptide identity map
        self._glycopeptide_hypothesis_migrator.clear()
        self._analysis_serializer.save_glycopeptide_identification_set(
            self._identified_glycopeptide_set)

        for chroma in self._unassigned_chromatogram_set:
            self._analysis_serializer.save_unidentified_chromatogram_solution(
                chroma)

        self.log("Committing Analysis")
        self._analysis_serializer.commit()


class DynamicGlycopeptideMSMSAnalysisSerializer(GlycopeptideMSMSAnalysisSerializer):
    def _get_glycan_combination_for_glycopeptide(self, glycopeptide_id) -> int:
        return glycopeptide_id.glycan_combination_id

    def _get_peptide_id_for_glycopeptide(self, glycopeptide_id) -> int:
        return glycopeptide_id.peptide_id

    def fetch_glycopeptides(self, glycopeptide_ids) -> List[Glycopeptide]:
        aggregate = dict()
        for gp in self._identified_glycopeptide_set:
            for solution_set in gp.spectrum_matches:
                for match in solution_set:
                    aggregate[match.target.id] = match.target
        out = []
        for i, obj in enumerate(aggregate.values(), 1):
            inst = Glycopeptide(
                id=obj.id,
                peptide_id=obj.id.peptide_id,
                glycan_combination_id=obj.id.glycan_combination_id,
                protein_id=obj.id.protein_id,
                hypothesis_id=obj.id.hypothesis_id,
                glycopeptide_sequence=obj.get_sequence(),
                calculated_mass=obj.total_mass,
                formula=formula(obj.total_composition()))
            out.append(inst)
        return out

    def _migrate_single_glycopeptide(self, glycopeptide: Glycopeptide) -> int:
        inst = Glycopeptide(
            id=glycopeptide.id,
            peptide_id=glycopeptide.id.peptide_id,
            glycan_combination_id=glycopeptide.id.glycan_combination_id,
            protein_id=glycopeptide.id.protein_id,
            hypothesis_id=glycopeptide.id.hypothesis_id,
            glycopeptide_sequence=glycopeptide.get_sequence(),
            calculated_mass=glycopeptide.total_mass,
            formula=formula(glycopeptide.total_composition()))
        self._glycopeptide_hypothesis_migrator.migrate_glycopeptide(inst)
        self._glycopeptide_hypothesis_migrator.commit()
        return self._glycopeptide_hypothesis_migrator.glycopeptide_id_map[glycopeptide.id]
