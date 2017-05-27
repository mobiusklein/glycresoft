from collections import defaultdict
from uuid import uuid4

from ms_deisotope.output.db import (
    MSScan,
    DatabaseScanDeserializer,
    DatabaseBoundOperation)

from glycan_profiling.task import TaskBase

from .analysis import Analysis
from .chromatogram import (
    Chromatogram,
    MassShiftSerializer,
    CompositionGroupSerializer,
    ChromatogramSolution,
    GlycanCompositionChromatogram,
    UnidentifiedChromatogram,
    ChromatogramSolutionAdductedToChromatogramSolution)

from .hypothesis import (
    GlycanComposition,
    GlycanCombinationGlycanComposition,
    GlycanCombination,
    Glycopeptide)

from .tandem import (
    GlycopeptideSpectrumCluster)

from .identification import (
    AmbiguousGlycopeptideGroup,
    IdentifiedGlycopeptide)


class AnalysisSerializer(DatabaseBoundOperation, TaskBase):
    def __init__(self, connection, sample_run_id, analysis_name, analysis_id=None):
        DatabaseBoundOperation.__init__(self, connection)
        session = self.session
        self.sample_run_id = sample_run_id

        self._analysis = None
        self._analysis_id = analysis_id
        self._analysis_name = None
        self._seed_analysis_name = analysis_name
        self._peak_lookup_table = None
        self._mass_shift_cache = MassShiftSerializer(session)
        self._composition_cache = CompositionGroupSerializer(session)
        self._node_peak_map = dict()
        self._scan_id_map = self._build_scan_id_map()
        self._chromatogram_solution_id_map = dict()

    def __repr__(self):
        return "AnalysisSerializer(%s, %d)" % (self.analysis.name, self.analysis_id)

    def set_peak_lookup_table(self, mapping):
        self._peak_lookup_table = mapping

    def build_peak_lookup_table(self, minimum_mass=500):
        peak_loader = DatabaseScanDeserializer(self._original_connection, sample_run_id=self.sample_run_id)
        accumulated = peak_loader.ms1_peaks_above(minimum_mass)
        peak_mapping = {x[:2]: x[2] for x in accumulated}
        self.set_peak_lookup_table(peak_mapping)

    def _build_scan_id_map(self):
        return dict(self.session.query(
            MSScan.scan_id, MSScan.id).filter(
            MSScan.sample_run_id == self.sample_run_id))

    @property
    def analysis(self):
        if self._analysis is None:
            self._construct_analysis()
        return self._analysis

    @property
    def analysis_id(self):
        if self._analysis_id is None:
            self._construct_analysis()
        return self._analysis_id

    @property
    def analysis_name(self):
        if self._analysis_name is None:
            self._construct_analysis()
        return self._analysis_name

    def set_analysis_type(self, type_string):
        self.analysis.analysis_type = type_string
        self.session.add(self.analysis)
        self.session.commit()

    def set_parameters(self, parameters):
        self.analysis.parameters = parameters
        self.session.add(self.analysis)
        self.commit()

    def _retrieve_analysis(self):
        self._analysis = self.session.query(Analysis).get(self._analysis_id)
        self._analysis_name = self._analysis.name

    def _create_analysis(self):
        self._analysis = Analysis(
            name=self._seed_analysis_name, sample_run_id=self.sample_run_id,
            uuid=str(uuid4()))
        self.session.add(self._analysis)
        self.session.flush()
        self._analysis_id = self._analysis.id
        self._analysis_name = self._analysis.name

    def _construct_analysis(self):
        if self._analysis_id is not None:
            self._retrieve_analysis()
        else:
            self._create_analysis()

    def save_chromatogram_solution(self, solution, commit=False):
        result = ChromatogramSolution.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map,
            node_peak_map=self._node_peak_map)
        try:
            self._chromatogram_solution_id_map[solution.id] = result.id
        except AttributeError:
            pass
        if commit:
            self.commit()
        return result

    def save_glycan_composition_chromatogram_solution(self, solution, commit=False):
        result = GlycanCompositionChromatogram.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map,
            node_peak_map=self._node_peak_map)
        try:
            self._chromatogram_solution_id_map[solution.id] = result.solution.id
        except AttributeError:
            pass

        if commit:
            self.commit()
        return result

    def save_unidentified_chromatogram_solution(self, solution, commit=False):
        result = UnidentifiedChromatogram.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map,
            node_peak_map=self._node_peak_map)
        try:
            self._chromatogram_solution_id_map[solution.id] = result.solution.id
        except AttributeError:
            pass

        if commit:
            self.commit()
        return result

    def save_glycopeptide_identification(self, identification, commit=False):
        if identification.chromatogram is not None:
            chromatogram_solution = self.save_chromatogram_solution(
                identification.chromatogram, commit=False)
            chromatogram_solution_id = chromatogram_solution.id
        else:
            chromatogram_solution_id = None
        cluster = GlycopeptideSpectrumCluster.serialize(
            identification, self.session, self._scan_id_map,
            analysis_id=self.analysis_id)
        cluster_id = cluster.id
        inst = IdentifiedGlycopeptide.serialize(
            identification, self.session, chromatogram_solution_id, cluster_id, analysis_id=self.analysis_id)
        if commit:
            self.commit()
        return inst

    def save_glycopeptide_identification_set(self, identification_set, commit=False):
        cache = defaultdict(list)
        no_chromatograms = []
        out = []
        n = len(identification_set)
        i = 0
        for case in identification_set:
            i += 1
            if i % 100 == 0:
                self.log("%0.2f%% glycopeptides saved. (%d/%d), %r" % (i * 100. / n, i, n, case))
            saved = self.save_glycopeptide_identification(case)
            if case.chromatogram is not None:
                cache[case.chromatogram].append(saved)
            else:
                no_chromatograms.append(saved)
            out.append(saved)
        for chromatogram, members in cache.items():
            AmbiguousGlycopeptideGroup.serialize(
                members, self.session, analysis_id=self.analysis_id)
        for case in no_chromatograms:
            AmbiguousGlycopeptideGroup.serialize(
                [case], self.session, analysis_id=self.analysis_id)
        return out

    def commit(self):
        self.session.commit()


class AnalysisDeserializer(DatabaseBoundOperation):
    def __init__(self, connection, analysis_name=None, analysis_id=None):
        DatabaseBoundOperation.__init__(self, connection)

        if analysis_name is analysis_id is None:
            analysis_id = 1

        self._analysis = None
        self._analysis_id = analysis_id
        self._analysis_name = analysis_name

    def _retrieve_analysis(self):
        if self._analysis_id is not None:
            self._analysis = self.session.query(Analysis).get(self._analysis_id)
            self._analysis_name = self._analysis.name
        elif self._analysis_name is not None:
            self._analysis = self.session.query(Analysis).filter(Analysis.name == self._analysis_name).one()
            self._analysis_id = self._analysis.id
        else:
            raise ValueError("No Analysis identification information provided")

    @property
    def analysis(self):
        if self._analysis is None:
            self._retrieve_analysis()
        return self._analysis

    @property
    def analysis_id(self):
        if self._analysis_id is None:
            self._retrieve_analysis()
        return self._analysis_id

    @property
    def analysis_name(self):
        if self._analysis_name is None:
            self._retrieve_analysis()
        return self._analysis_name

    def load_unidentified_chromatograms(self):
        from glycan_profiling.chromatogram_tree import ChromatogramFilter
        q = self.query(UnidentifiedChromatogram).filter(
            UnidentifiedChromatogram.analysis_id == self.analysis_id).yield_per(100)
        chroma = ChromatogramFilter([c.convert() for c in q])
        return chroma

    def load_glycan_composition_chromatograms(self):
        from glycan_profiling.chromatogram_tree import ChromatogramFilter
        q = self.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id).yield_per(100)
        chroma = ChromatogramFilter([c.convert() for c in q])
        return chroma

    def load_identified_glycopeptides_for_protein(self, protein_id):
        q = self.query(IdentifiedGlycopeptide).join(Glycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id,
            Glycopeptide.protein_id == protein_id).yield_per(100)
        gps = [c.convert() for c in q]
        return gps

    def load_identified_glycopeptides(self):
        q = self.query(IdentifiedGlycopeptide).join(Glycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id).yield_per(100)
        gps = [c.convert() for c in q]
        return gps

    def load_glycans_from_identified_glycopeptides(self):
        q = self.query(GlycanComposition).join(
            GlycanCombinationGlycanComposition).join(GlycanCombination).join(
            Glycopeptide,
            Glycopeptide.glycan_combination_id == GlycanCombination.id).join(
            IdentifiedGlycopeptide,
            IdentifiedGlycopeptide.structure_id == Glycopeptide.id).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id)
        gcs = [c for c in q]
        return gcs


class AnalysisDestroyer(DatabaseBoundOperation):
    def __init__(self, database_connection, analysis_id):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.analysis_id = analysis_id

    def delete_chromatograms(self):
        self.session.query(Chromatogram).filter(
            Chromatogram.analysis_id == self.analysis_id).delete(synchronize_session=False)
        self.session.flush()

    def delete_chromatogram_solutions(self):
        self.session.query(ChromatogramSolution).filter(
            ChromatogramSolution.analysis_id == self.analysis_id).delete(synchronize_session=False)
        self.session.flush()

    def delete_glycan_composition_chromatograms(self):
        self.session.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id).delete(synchronize_session=False)
        self.session.flush()

    def delete_unidentified_chromatograms(self):
        self.session.query(UnidentifiedChromatogram).filter(
            UnidentifiedChromatogram.analysis_id == self.analysis_id).delete(synchronize_session=False)
        self.session.flush()

    def delete_ambiguous_glycopeptide_groups(self):
        self.session.query(AmbiguousGlycopeptideGroup).filter(
            AmbiguousGlycopeptideGroup.analysis_id == self.analysis_id).delete(synchronize_session=False)
        self.session.flush()

    def delete_identified_glycopeptides(self):
        self.session.query(IdentifiedGlycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id).delete(synchronize_session=False)
        self.session.flush()

    def delete_analysis(self):
        self.session.query(Analysis).filter(Analysis.id == self.analysis_id).delete(synchronize_session=False)

    def run(self):
        self.delete_ambiguous_glycopeptide_groups()
        self.delete_identified_glycopeptides()
        self.delete_glycan_composition_chromatograms()
        self.delete_unidentified_chromatograms()
        self.delete_chromatograms()
        self.delete_analysis()
        self.session.commit()
