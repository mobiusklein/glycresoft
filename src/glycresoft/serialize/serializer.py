import time

from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, List, Tuple, TYPE_CHECKING
from uuid import uuid4

from sqlalchemy.orm import Query, aliased

from ms_deisotope.output.common import (ScanDeserializerBase, ScanBunch)

from glycresoft.task import TaskBase

from .connection import DatabaseBoundOperation
from .spectrum import (
    MSScan, PrecursorInformation, SampleRun, DeconvolutedPeak)

from .analysis import Analysis, _AnalysisParametersProps
from .chromatogram import (
    Chromatogram,
    MassShiftSerializer,
    CompositionGroupSerializer,
    ChromatogramSolution,
    GlycanCompositionChromatogram,
    UnidentifiedChromatogram)

from .hypothesis import (
    GlycanComposition,
    GlycanCombinationGlycanComposition,
    GlycanCombination,
    Glycopeptide)

from .tandem import (
    GlycopeptideSpectrumMatch,
    GlycopeptideSpectrumSolutionSet,
    SpectrumClusterBase,
    GlycopeptideSpectrumCluster,
    GlycanCompositionSpectrumCluster,
    UnidentifiedSpectrumCluster,
    GlycanCompositionChromatogramToGlycanCompositionSpectrumCluster,
    UnidentifiedChromatogramToUnidentifiedSpectrumCluster)

from .identification import (
    AmbiguousGlycopeptideGroup,
    IdentifiedGlycopeptide,
    IdentifiedGlycopeptideSummary)


if TYPE_CHECKING:
    from glycresoft.tandem.glycopeptide.identified_structure import (
        IdentifiedGlycopeptide as MemoryIdentifiedGlycopeptide
    )


class AnalysisSerializer(DatabaseBoundOperation, TaskBase):
    _analysis: Analysis
    _analysis_id: int
    _analysis_name: str
    _seed_analysis_name: str
    _peak_lookup_table: Dict[Tuple[str, DeconvolutedPeak], int]
    _mass_shift_cache: MassShiftSerializer
    _composition_cache: CompositionGroupSerializer
    _node_peak_map: Dict
    _scan_id_map: Dict[str, int]
    _chromatogram_solution_id_map: Dict
    _tandem_cluster_cache: Dict[FrozenSet, SpectrumClusterBase]

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
        self._tandem_cluster_cache = dict()

    def __repr__(self):
        return "AnalysisSerializer(%s, %d)" % (self.analysis.name, self.analysis_id)

    def set_peak_lookup_table(self, mapping: Dict[Tuple, DeconvolutedPeak]):
        self._peak_lookup_table = mapping

    def build_peak_lookup_table(self, minimum_mass=500):
        peak_loader = DatabaseScanDeserializer(self._original_connection, sample_run_id=self.sample_run_id)
        accumulated = peak_loader.ms1_peaks_above(minimum_mass)
        peak_mapping = {x[:2]: x[2] for x in accumulated}
        self.set_peak_lookup_table(peak_mapping)

    def _build_scan_id_map(self) -> Dict[str, int]:
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

    def set_parameters(self, parameters: Dict):
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
        try:
            cluster = self.save_glycan_composition_spectrum_cluster(solution)
            cluster_id = cluster.id
        except AttributeError:
            cluster_id = None
        result = GlycanCompositionChromatogram.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map,
            node_peak_map=self._node_peak_map)
        if cluster_id is not None:
            self.session.execute(
                GlycanCompositionChromatogramToGlycanCompositionSpectrumCluster.insert(), {
                    "chromatogram_id": result.id,
                    "cluster_id": cluster_id
                })
            self.session.flush()
        try:
            self._chromatogram_solution_id_map[solution.id] = result.solution.id
        except AttributeError:
            pass

        if commit:
            self.commit()
        return result

    def save_unidentified_chromatogram_solution(self, solution, commit=False):
        try:
            cluster = self.save_unidentified_spectrum_cluster(solution)
            cluster_id = cluster.id
        except AttributeError:
            cluster_id = None
        result = UnidentifiedChromatogram.serialize(
            solution, self.session, analysis_id=self.analysis_id,
            peak_lookup_table=self._peak_lookup_table,
            mass_shift_cache=self._mass_shift_cache,
            composition_cache=self._composition_cache,
            scan_lookup_table=self._scan_id_map,
            node_peak_map=self._node_peak_map)
        if cluster_id is not None:
            self.session.execute(
                UnidentifiedChromatogramToUnidentifiedSpectrumCluster.insert(), {
                    "chromatogram_id": result.id,
                    "cluster_id": cluster_id
                })
            self.session.flush()
        try:
            self._chromatogram_solution_id_map[solution.id] = result.solution.id
        except AttributeError:
            pass

        if commit:
            self.commit()
        return result

    def save_glycan_composition_spectrum_cluster(self, glycan_spectrum_cluster, commit=False):
        inst = GlycanCompositionSpectrumCluster.serialize(
            glycan_spectrum_cluster,
            self.session,
            self._scan_id_map,
            self._mass_shift_cache,
            self.analysis_id)
        if commit:
            self.commit()
        return inst

    def save_unidentified_spectrum_cluster(self, unidentified_spectrum_cluster, commit=False):
        inst = UnidentifiedSpectrumCluster.serialize(
            unidentified_spectrum_cluster,
            self.session,
            self._scan_id_map,
            self._mass_shift_cache,
            self.analysis_id)
        if commit:
            self.commit()
        return inst

    def save_glycopeptide_identification(self, identification: 'MemoryIdentifiedGlycopeptide',
                                         commit: bool=False):
        _start_time = time.time()
        if identification.chromatogram is not None:
            chromatogram_solution = self.save_chromatogram_solution(
                identification.chromatogram, commit=False)
            chromatogram_solution_id = chromatogram_solution.id
        else:
            chromatogram_solution_id = None
        _chromatogram_saving_elapsed = time.time() - _start_time

        # Compute the cluster key to avoid duplicating the storage of duplicate matches
        cluster_key = GlycopeptideSpectrumCluster.compute_key(identification.tandem_solutions)
        _cluster_key_elapsed = time.time() - (_chromatogram_saving_elapsed + _start_time)
        if cluster_key in self._tandem_cluster_cache:
            cluster = self._tandem_cluster_cache[cluster_key]
            _cluster_serialized_elapsed = 0
        else:
            cluster = GlycopeptideSpectrumCluster.serialize(
                identification, self.session, self._scan_id_map, self._mass_shift_cache,
                analysis_id=self.analysis_id)
            self._tandem_cluster_cache[cluster_key] = cluster
            _cluster_serialized_elapsed = time.time() - (_cluster_key_elapsed +
                                                         _chromatogram_saving_elapsed + _start_time)

        _total_elapsed = (_cluster_serialized_elapsed + _cluster_key_elapsed + _chromatogram_saving_elapsed)

        if _total_elapsed > 5:
            _cluster_size = sum(map(len, identification.tandem_solutions))
            _chromatogram_size = len(identification.chromatogram) if identification.chromatogram else 0
            self.log(f"..... Saving {identification} took {_total_elapsed:0.2f} sec "
                     f"(chromatogram {_chromatogram_saving_elapsed:0.2f}s of {_chromatogram_size} nodes,"
                     f" key {_cluster_key_elapsed:0.2f}s,"
                     f" cluster {_cluster_serialized_elapsed:0.2f}s of {_cluster_size} entries)")

        cluster_id = cluster.id

        best_match = identification.best_spectrum_match

        inst = IdentifiedGlycopeptide.serialize(
            identification, self.session, chromatogram_solution_id, cluster_id, analysis_id=self.analysis_id)

        if identification.chromatogram is not None:
            apex_time = identification.apex_time
            total_signal = identification.total_signal
            start_time = identification.start_time
            end_time = identification.end_time
            weighted_neutral_mass = identification.weighted_neutral_mass

            best_spectrum_match_id = None
            sset: GlycopeptideSpectrumSolutionSet = self.session.query(
                GlycopeptideSpectrumSolutionSet).join(MSScan).filter(
                MSScan.scan_id == best_match.scan_id,
                GlycopeptideSpectrumSolutionSet.analysis_id == self.analysis_id
            ).first()

            if sset is not None:
                db_best_match = sset.spectrum_matches.filter(
                    GlycopeptideSpectrumMatch.structure_id == inst.structure_id
                ).first()
                if best_match is not None:
                    if db_best_match is not None:
                        best_spectrum_match_id = db_best_match.id

            if best_spectrum_match_id is None:
                self.log(f"Failed to resolve best spectrum match for {identification} in the database")

            summary = IdentifiedGlycopeptideSummary(
                id=inst.id,
                weighted_neutral_mass=weighted_neutral_mass,
                apex_time=apex_time,
                total_signal=total_signal,
                start_time=start_time,
                end_time=end_time,
                best_spectrum_match_id=best_spectrum_match_id,
            )
            self.session.add(summary)
            self.session.flush()
        if commit:
            self.commit()
        return inst

    def save_glycopeptide_identification_set(self, identification_set: Iterable['MemoryIdentifiedGlycopeptide'],
                                             commit: bool=False):
        cache = defaultdict(list)
        no_chromatograms = []
        out = []
        n = len(identification_set)
        i = 0

        chromatogram_node_counter = 0
        gpsms_solution_counter = 0
        spectra_counter = 0
        for case in identification_set:
            i += 1
            if i % 100 == 0:
                self.log(
                    f"{i * 100. / n:0.2f}% glycopeptides saved. ({i}/{n}), {case.structure}:"
                    f"MS2={case.ms2_score:0.2f}:{case.q_value:0.3f}, MS1={case.ms1_score:0.2f}:{case.total_signal:0.2e}"
                    f"({spectra_counter} spectra, {gpsms_solution_counter} solutions, "
                    f"{chromatogram_node_counter} XIC points)"
                )
            gpsms_solution_counter += sum(map(len, case.tandem_solutions))
            spectra_counter += len(case.tandem_solutions)
            chromatogram_node_counter += len(case.chromatogram) if case.chromatogram is not None else 0
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
        if commit:
            self.commit()
        return out

    def commit(self):
        self.session.commit()


class AnalysisDeserializer(DatabaseBoundOperation, _AnalysisParametersProps):
    def __init__(self, connection: str, analysis_name=None, analysis_id=None):
        DatabaseBoundOperation.__init__(self, connection)

        if analysis_name is analysis_id is None:
            analysis_id = 1

        self._analysis = None
        self._analysis_id = analysis_id
        self._analysis_name = analysis_name

    @property
    def name(self):
        return self.analysis_name

    @property
    def chromatogram_scoring_model(self):
        try:
            return self.analysis.parameters["scoring_model"]
        except KeyError:
            from glycresoft.models import GeneralScorer
            return GeneralScorer

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
    def analysis(self) -> Analysis:
        if self._analysis is None:
            self._retrieve_analysis()
        return self._analysis

    @property
    def analysis_id(self) -> int:
        if self._analysis_id is None:
            self._retrieve_analysis()
        return self._analysis_id

    @property
    def analysis_name(self) -> str:
        if self._analysis_name is None:
            self._retrieve_analysis()
        return self._analysis_name

    @property
    def parameters(self):
        return self.analysis.parameters

    def load_unidentified_chromatograms(self):
        from glycresoft.chromatogram_tree import ChromatogramFilter
        node_type_cache = dict()
        scan_id_cache = dict()
        q = self.query(UnidentifiedChromatogram).filter(
            UnidentifiedChromatogram.analysis_id == self.analysis_id).all()
        chroma = ChromatogramFilter([c.convert(
            chromatogram_scoring_model=self.chromatogram_scoring_model,
            node_type_cache=node_type_cache,
            scan_id_cache=scan_id_cache) for c in q])
        return chroma

    def load_glycan_composition_chromatograms(self):
        from glycresoft.chromatogram_tree import ChromatogramFilter
        node_type_cache = dict()
        scan_id_cache = dict()
        q = self.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id).all()
        chroma = ChromatogramFilter([c.convert(
            chromatogram_scoring_model=self.chromatogram_scoring_model,
            node_type_cache=node_type_cache,
            scan_id_cache=scan_id_cache) for c in q])
        return chroma

    def load_identified_glycopeptides_for_protein(self, protein_id):
        q = self.query(IdentifiedGlycopeptide).join(Glycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id,
            Glycopeptide.protein_id == protein_id)
        gps = [c.convert() for c in q]
        return gps

    def load_identified_glycopeptides(self, min_q_value: float=0.2):
        q = self.query(IdentifiedGlycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id).all()
        gps = IdentifiedGlycopeptide.bulk_convert(q, min_q_value=min_q_value)
        return gps

    def load_glycans_from_identified_glycopeptides(self) -> List[GlycanComposition]:
        q = self.query(GlycanComposition).join(
                GlycanCombinationGlycanComposition).join(GlycanCombination).join(
                Glycopeptide,
                Glycopeptide.glycan_combination_id == GlycanCombination.id).join(
                IdentifiedGlycopeptide,
                IdentifiedGlycopeptide.structure_id == Glycopeptide.id).filter(
                IdentifiedGlycopeptide.analysis_id == self.analysis_id)
        gcs = [c for c in q]
        return gcs

    def get_glycopeptide_by_sequence(self, sequence: str) -> Query:
        return self.query(IdentifiedGlycopeptide).join(
            IdentifiedGlycopeptide.structure).filter(
                Glycopeptide.glycopeptide_sequence == sequence)

    def get_glycopeptide_spectrum_matches(self, q_value_threshold: float=0.05) -> Query:
        query = self.query(GlycopeptideSpectrumMatch).join(
            GlycopeptideSpectrumSolutionSet).join(
            GlycopeptideSpectrumCluster).join(
            IdentifiedGlycopeptide).filter(
            GlycopeptideSpectrumMatch.is_best_match,
            # GlycopeptideSpectrumMatch.structure_id == IdentifiedGlycopeptide.structure_id,
            GlycopeptideSpectrumMatch.q_value < q_value_threshold,
            GlycopeptideSpectrumMatch.analysis_id == self.analysis_id)
        return query

    def _get_glycopeptide_spectrum_matches_not_matching_owning_structure(self, q_value_threshold: float=0.05) -> Query:
        GPSM_Glycopeptide = aliased(Glycopeptide)
        mixed = self.query(GlycopeptideSpectrumMatch, IdentifiedGlycopeptide).join(
            GlycopeptideSpectrumMatch.solution_set).join(
                GlycopeptideSpectrumSolutionSet.cluster).join(
                    GlycopeptideSpectrumCluster.owners).join(
                        GPSM_Glycopeptide,
                        GlycopeptideSpectrumMatch.structure_id == GPSM_Glycopeptide.id
                    ).join(
                        IdentifiedGlycopeptide.structure
                    ).filter(
            GlycopeptideSpectrumMatch.is_best_match,
            GlycopeptideSpectrumMatch.q_value < q_value_threshold,
            GPSM_Glycopeptide.glycopeptide_sequence != Glycopeptide.glycopeptide_sequence,
            GPSM_Glycopeptide.peptide_id != Glycopeptide.peptide_id,
            GlycopeptideSpectrumMatch.analysis_id == self.analysis_id
        )
        return mixed

    def __repr__(self):
        template = "{self.__class__.__name__}({self.engine!r}, {self.analysis_name!r})"
        return template.format(self=self)


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


def flatten(iterable):
    return [y for x in iterable for y in x]


class DatabaseScanDeserializer(ScanDeserializerBase, DatabaseBoundOperation):

    def __init__(self, connection, sample_name=None, sample_run_id=None):

        DatabaseBoundOperation.__init__(self, connection)

        self._sample_run = None
        self._sample_name = sample_name
        self._sample_run_id = sample_run_id
        self._iterator = None
        self._scan_id_to_retention_time_cache = None

    def _intialize_scan_id_to_retention_time_cache(self):
        self._scan_id_to_retention_time_cache = dict(
            self.session.query(MSScan.scan_id, MSScan.scan_time).filter(
                MSScan.sample_run_id == self.sample_run_id))

    def __reduce__(self):
        return self.__class__, (
            self._original_connection, self.sample_name, self.sample_run_id)

    # Sample Run Bound Handle API

    @property
    def sample_run_id(self):
        if self._sample_run_id is None:
            self._retrieve_sample_run()
        return self._sample_run_id

    @property
    def sample_run(self):
        if self._sample_run is None:
            self._retrieve_sample_run()
        return self._sample_run

    @property
    def sample_name(self):
        if self._sample_name is None:
            self._retrieve_sample_run()
        return self._sample_name

    def _retrieve_sample_run(self):
        session = self.session
        if self._sample_name is not None:
            sr = session.query(SampleRun).filter(
                SampleRun.name == self._sample_name).one()
        elif self._sample_run_id is not None:
            sr = session.query(SampleRun).filter(
                SampleRun.id == self._sample_run_id).one()
        else:
            sr = session.query(SampleRun).first()
        self._sample_run = sr
        self._sample_run_id = sr.id
        self._sample_name = sr.name

    # Scan Generator Public API

    def get_scan_by_id(self, scan_id):
        q = self._get_by_scan_id(scan_id)
        if q is None:
            raise KeyError(scan_id)
        mem = q.convert()
        if mem.precursor_information:
            mem.precursor_information.source = self
        return mem

    def convert_scan_id_to_retention_time(self, scan_id):
        if self._scan_id_to_retention_time_cache is None:
            self._intialize_scan_id_to_retention_time_cache()
        try:
            return self._scan_id_to_retention_time_cache[scan_id]
        except KeyError:
            q = self.session.query(MSScan.scan_time).filter(
                MSScan.scan_id == scan_id, MSScan.sample_run_id == self.sample_run_id).scalar()
            self._scan_id_to_retention_time_cache[scan_id] = q
            return q

    def _select_index(self, require_ms1=True):
        indices_q = self.session.query(MSScan.index).filter(
            MSScan.sample_run_id == self.sample_run_id).order_by(MSScan.index.asc())
        if require_ms1:
            indices_q = indices_q.filter(MSScan.ms_level == 1)
        indices = flatten(indices_q.all())
        return indices

    def _iterate_over_index(self, start=0, require_ms1=True):
        indices = self._select_index(require_ms1)
        try:
            i = indices.index(start)
        except ValueError:
            lo = 0
            hi = len(indices)

            while lo != hi:
                mid = (lo + hi) // 2
                x = indices[mid]
                if x == start:
                    i = mid
                    break
                elif lo == (hi - 1):
                    i = mid
                    break
                elif x > start:
                    hi = mid
                else:
                    lo = mid
        items = indices[i:]
        i = 0
        n = len(items)
        while i < n:
            index = items[i]
            scan = self.session.query(MSScan).filter(
                MSScan.index == index,
                MSScan.sample_run_id == self.sample_run_id).one()
            products = [pi.product for pi in scan.product_information]
            yield ScanBunch(scan.convert(), [p.convert() for p in products])
            i += 1

    def __iter__(self):
        return self

    def next(self):
        if self._iterator is None:
            self._iterator = self._iterate_over_index()
        return next(self._iterator)

    def _get_by_scan_id(self, scan_id):
        q = self.session.query(MSScan).filter(
            MSScan.scan_id == scan_id, MSScan.sample_run_id == self.sample_run_id).first()
        return q

    def _get_scan_by_time(self, rt, require_ms1=False):
        times_q = self.session.query(MSScan.scan_time).filter(
            MSScan.sample_run_id == self.sample_run_id).order_by(MSScan.scan_time.asc())
        if require_ms1:
            times_q = times_q.filter(MSScan.ms_level == 1)
        times = flatten(times_q.all())
        try:
            i = times.index(rt)
        except ValueError:
            lo = 0
            hi = len(times)

            while lo != hi:
                mid = (lo + hi) // 2
                x = times[mid]
                if x == rt:
                    i = mid
                    break
                elif lo == (hi - 1):
                    i = mid
                    break
                elif x > rt:
                    hi = mid
                else:
                    lo = mid
        scan = self.session.query(MSScan).filter(
            MSScan.scan_time == times[i],
            MSScan.sample_run_id == self.sample_run_id).one()
        return scan

    def reset(self):
        self._iterator = None

    def get_scan_by_time(self, rt, require_ms1=False):
        q = self._get_scan_by_time(rt, require_ms1)
        mem = q.convert()
        if mem.precursor_information:
            mem.precursor_information.source = self
        return mem

    def _get_scan_by_index(self, index):
        q = self.session.query(MSScan).filter(
            MSScan.index == index, MSScan.sample_run_id == self.sample_run_id).one()
        return q

    def get_scan_by_index(self, index):
        mem = self._get_scan_by_index(index).convert()
        if mem.precursor_information:
            mem.precursor_information.source = self
        return mem

    def _locate_ms1_scan(self, scan):
        while scan.ms_level != 1:
            scan = self._get_scan_by_index(scan.index - 1)
        return scan

    def start_from_scan(self, scan_id=None, rt=None, index=None, require_ms1=True):
        if scan_id is None:
            if rt is not None:
                scan = self._get_scan_by_time(rt)
            elif index is not None:
                scan = self._get_scan_by_index(index)
        else:
            scan = self._get_by_scan_id(scan_id)

        # We must start at an MS1 scan, so backtrack until we reach one
        if require_ms1:
            scan = self._locate_ms1_scan(scan)
        self._iterator = self._iterate_over_index(scan.index)
        return self

    # LC-MS/MS Database API

    def msms_for(self, neutral_mass, mass_error_tolerance=1e-5, start_time=None, end_time=None):
        m = neutral_mass
        w = neutral_mass * mass_error_tolerance
        q = self.session.query(PrecursorInformation).join(
            PrecursorInformation.precursor).filter(
            PrecursorInformation.neutral_mass.between(m - w, m + w),
            PrecursorInformation.sample_run_id == self.sample_run_id).order_by(
            MSScan.scan_time)
        if start_time is not None:
            q = q.filter(MSScan.scan_time >= start_time)
        if end_time is not None:
            q = q.filter(MSScan.scan_time <= end_time)
        return q

    def ms1_peaks_above(self, threshold=1000):
        accumulate = [
            (x[0], x[1].convert(), x[1].id) for x in self.session.query(MSScan.scan_id, DeconvolutedPeak).join(
                DeconvolutedPeak).filter(
                MSScan.ms_level == 1, MSScan.sample_run_id == self.sample_run_id,
                DeconvolutedPeak.neutral_mass > threshold
            ).order_by(MSScan.index).yield_per(1000)]
        return accumulate

    def precursor_information(self):
        prec_info = self.session.query(PrecursorInformation).filter(
            PrecursorInformation.sample_run_id == self.sample_run_id).all()
        return prec_info
