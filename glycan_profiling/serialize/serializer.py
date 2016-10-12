from collections import defaultdict


from ms_deisotope.output.db import (
    Base, DeconvolutedPeak, MSScan, Mass, HasUniqueName,
    SampleRun, DatabaseScanDeserializer, DatabaseBoundOperation)

from .analysis import Analysis
from .chromatogram import (
    MassShiftSerializer, CompositionGroupSerializer, ChromatogramSolution,
    GlycanCompositionChromatogram, UnidentifiedChromatogram)

from .tandem import (
    GlycopeptideSpectrumCluster)

from .identification import (
    AmbiguousGlycopeptideGroup, IdentifiedGlycopeptide)


class AnalysisSerializer(DatabaseBoundOperation):
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
        self._scan_id_map = self._build_scan_id_map()

    def __repr__(self):
        return "AnalysisSerializer(%s, %d)" % (self.analysis.name, self.analysis_id)

    def set_peak_lookup_table(self, mapping):
        self._peak_lookup_table = mapping

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

    def _retrieve_analysis(self):
        self._analysis = self.session.query(Analysis).get(self._analysis_id)
        self._analysis_name = self._analysis.name

    def _create_analysis(self):
        self._analysis = Analysis(name=self._seed_analysis_name, sample_run_id=self.sample_run_id)
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
            scan_lookup_table=self._scan_id_map)
        if commit:
            self.commit()
        return result

    def save_glycan_composition_chromatogram_solution(self, solution, commit=False):
        result = GlycanCompositionChromatogram.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map)
        if commit:
            self.commit()
        return result

    def save_unidentified_chromatogram_solution(self, solution, commit=False):
        result = UnidentifiedChromatogram.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map)
        if commit:
            self.commit()
        return result

    def save_glycopeptide_identification(self, identification, commit=False):
        chromatogram_solution = self.save_chromatogram_solution(
            identification.chromatogram, commit=False)
        chromatogram_solution_id = chromatogram_solution.id
        cluster = GlycopeptideSpectrumCluster.serialize(
            identification, self.session, self._scan_id_map,
            analysis_id=self.analysis_id)
        cluster_id = cluster.id
        inst = IdentifiedGlycopeptide.serialize(
            identification, self.session, chromatogram_solution_id, cluster_id)
        if commit:
            self.commit()
        return inst

    def save_glycopeptide_identification_set(self, identification_set, commit=False):
        cache = defaultdict(list)
        out = []
        for case in identification_set:
            saved = self.save_glycopeptide_identification(case)
            cache[case.chromatogram].append(saved)
            out.append(saved)
        for chromatogram, members in cache.items():
            AmbiguousGlycopeptideGroup.serialize(members, self.session)
        return out

    def commit(self):
        self.session.commit()


class AnalysisDeserializer(DatabaseBoundOperation):
    def __init__(self, connection, analysis_name=None, analysis_id=None):
        DatabaseBoundOperation.__init__(self, connection)

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
        q = self.query(UnidentifiedChromatogram).filter(
            UnidentifiedChromatogram.analysis_id == self.analysis_id).yield_per(100)
        chroma = [c.convert() for c in q]
        return chroma

    def load_glycan_composition_chromatograms(self):
        q = self.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id).yield_per(100)
        chroma = [c.convert() for c in q]
        return chroma
