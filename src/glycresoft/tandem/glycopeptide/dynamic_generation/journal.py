'''Implementation for journal file reading and writing to serialize
glycopeptide spectrum match information to disk during processing.
'''
import os
import io
import csv
import json
import gzip

from multiprocessing import Event, JoinableQueue

from collections import defaultdict
from operator import attrgetter
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Type, Union, TYPE_CHECKING, BinaryIO


import numpy as np

from glycresoft.chromatogram_tree.mass_shift import MassShiftBase

try:
    from pyzstd import ZstdFile
except ImportError:
    ZstdFile = None

from glycopeptidepy.utils import collectiontools

from ms_deisotope.output import ProcessedMSFileLoader

from glycresoft.task import TaskBase, TaskExecutionSequence, Empty

from glycresoft.structure.structure_loader import (
    PeptideProteinRelation, LazyGlycopeptide)
from glycresoft.structure.scan import ScanInformation, ScanInformationLoader
from glycresoft.structure.lru import LRUMapping
from glycresoft.chromatogram_tree import Unmodified

from glycresoft.tandem.ref import SpectrumReference
from glycresoft.tandem.spectrum_match import (
    MultiScoreSpectrumMatch, MultiScoreSpectrumSolutionSet, ScoreSet, FDRSet)

from .search_space import glycopeptide_key_t, StructureClassification

if TYPE_CHECKING:
    from glycresoft.serialize import Analysis, AnalysisDeserializer
    from ms_deisotope.data_source import RandomAccessScanSource


def _zstdopen(path, mode='w'):
    handle = ZstdFile(path, mode=mode.replace('t', 'b'))
    return io.TextIOWrapper(handle, encoding='utf8')


def _zstwrap(fh, mode='w'):
    handle = ZstdFile(fh, mode=mode)
    return io.TextIOWrapper(handle, encoding='utf8')


def _gzopen(path, mode='w'):
    handle = gzip.open(path, mode=mode.replace('t', 'b'))
    return io.TextIOWrapper(handle, encoding='utf8')


def _gzwrap(fh, mode='w'):
    handle = gzip.GzipFile(fileobj=fh, mode=mode)
    return io.TextIOWrapper(handle, encoding='utf8')


if ZstdFile is None:
    _file_opener = _gzopen
    _file_wrapper = _gzwrap
else:
    _file_opener = _zstdopen
    _file_wrapper = _zstwrap


class JournalFileWriter(TaskBase):
    """A task for writing glycopeptide spectrum matches to a TSV-formatted
    journal file. This format is an intermediary result, and will contain many
    random or non-useful matches.

    """

    path: os.PathLike
    handle: io.TextIOWrapper

    include_auxiliary: bool
    include_fdr: bool
    writer: csv.writer
    spectrum_counter: int
    solution_counter: int

    score_columns: List[str]

    def __init__(self, path, include_fdr=False, include_auxiliary=False, score_columns: List[str]=None):
        if score_columns is None:
            score_columns = ScoreSet.field_names()
        self.path = path
        if not hasattr(path, 'write'):
            self.handle = _file_opener(path, 'wb')
        else:
            self.handle = _file_wrapper(self.path, 'w')
        self.include_fdr = include_fdr
        self.include_auxiliary = include_auxiliary
        self.spectrum_counter = 0
        self.solution_counter = 0

        self.score_columns = score_columns

        self.writer = csv.writer(self.handle, delimiter='\t')
        self.write_header()

    def _get_headers(self):
        names = [
            'scan_id',
            'precursor_mass_accuracy',
            'peptide_start',
            'peptide_end',
            'peptide_id',
            'protein_id',
            'hypothesis_id',
            'glycan_combination_id',
            'match_type',
            'site_combination_index',
            'glycosylation_type',
            'glycopeptide_sequence',
            'mass_shift',
        ]
        names.extend(self.score_columns)
        if self.include_fdr:
            names.extend([
                "peptide_q_value",
                "glycan_q_value",
                "glycopeptide_q_value",
                "total_q_value"
            ])
        if self.include_auxiliary:
            names.append("auxiliary")
        return names

    def write_header(self):
        self.writer.writerow(self._get_headers())

    def _prepare_fields(self, psm: MultiScoreSpectrumMatch) -> List[str]:
        error = (psm.target.total_mass - psm.precursor_information.neutral_mass
                ) / psm.precursor_information.neutral_mass
        fields = ([psm.scan_id, error, ] + list(psm.target.id) + [
            psm.target,
            psm.mass_shift.name,
        ])
        fields.extend(psm.score_set.values())
        if self.include_fdr:
            q_value_set = psm.q_value_set
            if q_value_set is None:
                fdr_fields = [
                    1, 1, 1, 1
                ]
            else:
                fdr_fields = [
                    q_value_set.peptide_q_value,
                    q_value_set.glycan_q_value,
                    q_value_set.glycopeptide_q_value,
                    q_value_set.total_q_value
                ]
            fields.extend(fdr_fields)
        if self.include_auxiliary:
            fields.append(json.dumps(psm.get_auxiliary_data(), sort_keys=True))
        fields = [str(f) for f in fields]
        return fields

    def write(self, psm: MultiScoreSpectrumMatch):
        self.solution_counter += 1
        self.writer.writerow(self._prepare_fields(psm))

    def writeall(self, solution_sets: List[MultiScoreSpectrumSolutionSet]):
        for solution_set in solution_sets:
            self.spectrum_counter += 1
            for solution in solution_set:
                self.write(solution)
        self.flush()

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


class JournalingConsumer(TaskExecutionSequence):
    journal_file: JournalFileWriter
    in_queue: JoinableQueue
    in_done_event: Event
    done_event: Event

    def __init__(self, journal_file, in_queue, in_done_event):
        self.journal_file = journal_file
        self.in_queue = in_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

    def run(self):
        has_work = True
        while has_work and not self.error_occurred():
            try:
                solutions = self.in_queue.get(True, 5)
                self.journal_file.writeall(solutions)
                # Only log if something changed.
                if solutions:
                    self.debug("... Handled %d spectra with %d solutions so far" % (
                        self.journal_file.spectrum_counter, self.journal_file.solution_counter))
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()

    def close_stream(self):
        self.journal_file.close()


def parse_float(value: str) -> float:
    value = float(value)
    if np.isnan(value):
        return 0
    return value


try:
    _parse_float = parse_float
    from glycresoft._c.tandem.tandem_scoring_helpers import parse_float
except ImportError:
    pass


class JournalFileIndex(TaskBase):
    def __init__(self, path):
        self.path = path
        self.scan_ids = set()
        self.glycopeptides = set()
        self.build_index()

    def build_index(self):
        reader = self.open()
        for row in reader:
            self.scan_ids.add(row['scan_id'])
            self.glycopeptides.add(row['glycopeptide_sequence'])

    def has_scan(self, scan_id: str) -> bool:
        return scan_id in self.scan_ids

    def has_glycopeptide(self, glycopeptide: str) -> bool:
        return glycopeptide in self.glycopeptides

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"

    def __contains__(self, key: str):
        return self.has_glycopeptide(key) or self.has_scan(key)

    def open(self):
        if not hasattr(self.path, 'read') and not hasattr(self.path, 'open'):
            handle = _file_opener(self.path, 'rt')
        elif hasattr(self.path, 'open'):
            handle = _file_wrapper(self.path.open(raw=True), 'r')
        else:
            handle = _file_wrapper(self.path, 'r')
        reader = csv.DictReader(handle, delimiter='\t')
        return reader

    def collect_for_scan_id(self, scan_id: str):
        reader = self.open()
        for row in reader:
            if scan_id == row['scan_id']:
                yield row


class JournalFileReader(TaskBase):
    path: os.PathLike
    handle: io.TextIOBase
    reader: csv.DictReader

    glycopeptide_cache: LRUMapping
    mass_shift_map: Dict[str, MassShiftBase]
    scan_loader: Optional[ScanInformationLoader]
    include_fdr: bool

    score_set_type: Type
    score_columns: List[str]

    def __init__(self, path, cache_size=2 ** 16, mass_shift_map=None, scan_loader=None,
                 include_fdr=False, score_set_type=None):
        if mass_shift_map is None:
            mass_shift_map = {Unmodified.name: Unmodified}
        else:
            mass_shift_map.setdefault(Unmodified.name, Unmodified)
        if score_set_type is None:
            score_set_type = ScoreSet
        self.path = path
        if not hasattr(path, 'read'):
            self.handle = _file_opener(path, 'rt')
        else:
            self.handle = _file_wrapper(self.path, 'r')
        self.reader = csv.DictReader(self.handle, delimiter='\t')
        self.glycopeptide_cache = LRUMapping(cache_size or 2 ** 12)
        self.mass_shift_map = mass_shift_map
        self.scan_loader = scan_loader
        self.include_fdr = include_fdr

        self.score_set_type = score_set_type
        self.score_columns = score_set_type.field_names()
        for name in ScoreSet.field_names():
            self.score_columns.remove(name)

    def _build_key(self, row) -> glycopeptide_key_t:
        glycopeptide_id_key = glycopeptide_key_t(
            int(row['peptide_start']), int(row['peptide_end']), int(
                row['peptide_id']), int(row['protein_id']),
            int(row['hypothesis_id']), int(row['glycan_combination_id']),
            StructureClassification[row['match_type']],
            int(row['site_combination_index']), row.get('glycosylation_type'))
        return glycopeptide_id_key

    def _build_protein_relation(self, glycopeptide_id_key: glycopeptide_key_t) -> PeptideProteinRelation:
        return PeptideProteinRelation(
            glycopeptide_id_key.start_position, glycopeptide_id_key.end_position,
            glycopeptide_id_key.protein_id, glycopeptide_id_key.hypothesis_id)

    def glycopeptide_from_row(self, row: Dict) -> LazyGlycopeptide:
        glycopeptide_id_key = self._build_key(row)
        if glycopeptide_id_key in self.glycopeptide_cache:
            return self.glycopeptide_cache[glycopeptide_id_key]
        if glycopeptide_id_key.structure_type & StructureClassification.target_peptide_decoy_glycan.value:
            glycopeptide = LazyGlycopeptide(row['glycopeptide_sequence'], glycopeptide_id_key)
        else:
            glycopeptide = LazyGlycopeptide(row['glycopeptide_sequence'], glycopeptide_id_key)
        glycopeptide.protein_relation = self._build_protein_relation(glycopeptide_id_key)
        self.glycopeptide_cache[glycopeptide_id_key] = glycopeptide
        return glycopeptide

    def _build_score_set(self, row) -> ScoreSet:
        score_set_entries = [
            parse_float(row['total_score']),
            parse_float(row['peptide_score']),
            parse_float(row['glycan_score']),
            float(row['glycan_coverage']),
            float(row['stub_glycopeptide_intensity_utilization']),
            float(row['oxonium_ion_intensity_utilization']),
            int(row['n_stub_glycopeptide_matches']),
            float(row.get('peptide_coverage', 0.0)),
            float(row.get('total_signal_utilization', 0.0))
        ]
        for name in self.score_columns:
            score_set_entries.append(parse_float(row.get(name, '0.0')))

        score_set = self.score_set_type(*score_set_entries)
        return score_set

    def _build_fdr_set(self, row) -> FDRSet:
        fdr_set = FDRSet(
            float(row['total_q_value']),
            float(row['peptide_q_value']),
            float(row['glycan_q_value']),
            float(row['glycopeptide_q_value']))
        return fdr_set

    def _make_mass_shift(self, row) -> MassShiftBase:
        mass_shift = self.mass_shift_map[row['mass_shift']]
        return mass_shift

    def _make_scan(self, row) -> Union[SpectrumReference, ScanInformation]:
        if self.scan_loader is None:
            scan = SpectrumReference(row['scan_id'])
        else:
            scan = self.scan_loader.get_scan_by_id(row['scan_id'])
        return scan

    def spectrum_match_from_row(self, row) -> MultiScoreSpectrumMatch:
        glycopeptide = self.glycopeptide_from_row(row)
        scan = self._make_scan(row)
        fdr_set = None
        if self.include_fdr:
            fdr_set = self._build_fdr_set(row)
        score_set = self._build_score_set(row)
        mass_shift = self._make_mass_shift(row)
        match = MultiScoreSpectrumMatch(
            scan, glycopeptide, score_set, mass_shift=mass_shift,
            q_value_set=fdr_set,
            match_type=str(glycopeptide.id.structure_type))
        return match

    def __iter__(self):
        return self

    def __next__(self) -> MultiScoreSpectrumMatch:
        return self.spectrum_match_from_row(next(self.reader))

    def next(self):
        return self.__next__()

    def close(self):
        self.handle.close()


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= atol + rtol * abs(b)


class SolutionSetGrouper(TaskBase):
    """
    Partition multi-score glycopeptide identificatins into groups
    according to either glycopeptide target/decoy classification (:attr:`match_type_groups`)
    or by best match to a given scan and target/decoy classification (:attr:`exclusive_match_groups`)

    """
    # All the matches of all match types for every spectrum in no particular order
    spectrum_matches: List[MultiScoreSpectrumMatch]
    # The set of all scan IDs
    spectrum_ids: Set[str]
    # The set of all scan IDs where a target is the best match
    target_owned_spectrum_ids: Set[str]
    # The spectrum solution sets for each target classification group
    match_type_groups: Dict[StructureClassification, List[MultiScoreSpectrumSolutionSet]]
    # The set of all spectrum matches that are the best scoring match for their scan in a specific
    # match type. This is used for FDR estimation.
    exclusive_match_groups: DefaultDict[StructureClassification,
                                        List[MultiScoreSpectrumMatch]]

    def __init__(self, spectrum_matches):
        self.spectrum_matches = list(spectrum_matches)
        self.spectrum_ids = set()
        self.target_owned_spectrum_ids = set()
        self.match_type_groups = self._collect(self.spectrum_matches)
        self.exclusive_match_groups = self._exclusive()

    def __getitem__(self, key):
        return self.exclusive_match_groups[key]

    def __iter__(self):
        return iter(self.exclusive_match_groups.items())

    def _by_scan_id(self):
        acc = []
        for by_scan in collectiontools.groupby(self.spectrum_matches, lambda x: x.scan.id).values():
            scan = by_scan[0].scan
            self.spectrum_ids.add(scan.scan_id)
            ss = MultiScoreSpectrumSolutionSet(scan, by_scan)
            ss.sort()
            acc.append(ss)
        acc.sort(key=lambda x: x.scan.id)
        return acc

    def _collect(
        self, spectrum_matches: List[MultiScoreSpectrumMatch]
    ) -> Dict[StructureClassification, List[MultiScoreSpectrumSolutionSet]]:

        match_type_getter = attrgetter('match_type')
        groups = collectiontools.groupby(spectrum_matches, match_type_getter)
        by_scan_groups = {}
        for group, members in groups.items():
            acc = []
            for by_scan in collectiontools.groupby(members, lambda x: x.scan.id).values():
                scan = by_scan[0].scan
                self.spectrum_ids.add(scan.scan_id)
                ss = MultiScoreSpectrumSolutionSet(scan, by_scan)
                ss.sort()
                acc.append(ss)
            acc.sort(key=lambda x: x.scan.id)
            by_scan_groups[group] = acc
        return by_scan_groups

    def _exclusive(
        self, score_getter: Callable[[MultiScoreSpectrumMatch], float] = None, min_value: float = 0
    ) -> DefaultDict[StructureClassification, List[MultiScoreSpectrumMatch]]:
        if score_getter is None:
            score_getter = attrgetter('score')
        groups = collectiontools.groupby(
            self.spectrum_matches, lambda x: x.scan.id)
        by_match_type = defaultdict(list)
        for scan_id, members in groups.items():
            top_match = max(members, key=score_getter)
            top_score = score_getter(top_match)
            seen = set()
            for match in members:
                if (isclose(top_score, score_getter(match)) and score_getter(match) > min_value
                    and match.match_type not in seen):
                    seen.add(match.match_type)
                    by_match_type[match.match_type].append(match)
                    if match.match_type == StructureClassification.target_peptide_target_glycan:
                        self.target_owned_spectrum_ids.add(scan_id)
        for _group_label, matches in by_match_type.items():
            matches.sort(key=lambda x: (x.scan.id, score_getter(x)))
        return by_match_type

    @property
    def target_matches(self) -> List[MultiScoreSpectrumMatch]:
        try:
            return list(
                filter(
                    lambda x: x.scan.id in self.target_owned_spectrum_ids,
                    self.match_type_groups[StructureClassification.target_peptide_target_glycan]
                )
            )
            # return self.match_type_groups[StructureClassification.target_peptide_target_glycan]
        except KeyError:
            return []

    @property
    def decoy_matches(self) -> List[MultiScoreSpectrumMatch]:
        try:
            return self.match_type_groups[StructureClassification.decoy_peptide_target_glycan]
        except KeyError:
            return []

    def target_count(self):
        return len(self.target_matches)

    def decoy_count(self):
        return len(self.decoy_matches)


class JournalSetLoader(TaskBase):
    """A helper class to load a list of journal file fragments
    into a single cohesive result, as from a previously compiled
    analysis's bundled journal file shards.
    """

    journal_files: List[Union[BinaryIO, os.PathLike, str]]
    journal_reader_args: Dict[str, Any]
    scan_loader: Union['RandomAccessScanSource', ScanInformationLoader]
    mass_shift_map: Dict[str, MassShiftBase]
    solutions: List[MultiScoreSpectrumSolutionSet]

    @classmethod
    def from_analysis(cls, analysis: Union["Analysis", "AnalysisDeserializer"],
                      scan_loader: Optional['RandomAccessScanSource']=None,
                      stub_wrapping: bool=True,
                      score_set_type: Optional[Type[ScoreSet]]=None,
                      **journal_reader_args):
        from glycresoft.serialize import AnalysisDeserializer
        if isinstance(analysis, AnalysisDeserializer):
            analysis = analysis.analysis
        mass_shift_map = {
            m.name: m for m in analysis.parameters['mass_shifts']}
        if scan_loader is None:
            scan_loader = ProcessedMSFileLoader(
                analysis.parameters['sample_path'])
        if stub_wrapping:
            stub_loader = ScanInformationLoader(scan_loader)
        else:
            stub_loader = scan_loader
        if score_set_type is None:
            score_set_type = analysis.parameters[
                'tandem_scoring_model'].get_score_set_type()
        return cls([f.open(raw=True) for f in analysis.files], stub_loader, mass_shift_map,
                   score_set_type=score_set_type, **journal_reader_args)

    def __init__(self, journal_files, scan_loader, mass_shift_map=None, **journal_reader_args):
        if mass_shift_map is None:
            mass_shift_map = {Unmodified.name: Unmodified}
        self.journal_files = journal_files
        self.journal_reader_args = journal_reader_args
        self.scan_loader = scan_loader
        self.mass_shift_map = mass_shift_map
        self.solutions = []

    def load(self):
        n = len(self.journal_files)
        for i, journal_path in enumerate(self.journal_files, 1):
            self.log("... Reading Journal Shard %s, %d/%d" %
                     (journal_path, i, n))
            self._load_identifications_from_journal(journal_path, self.solutions)
        self.log("Partitioning Spectrum Matches...")
        return SolutionSetGrouper(self.solutions)

    def __iter__(self):
        if not self.solutions:
            self.load()
        return iter(self.solutions)

    def _load_identifications_from_journal(self, journal_path: Union[BinaryIO, os.PathLike, str],
                                           accumulator: Optional[List[MultiScoreSpectrumSolutionSet]]=None):
        if accumulator is None:
            accumulator = []
        reader = enumerate(
            JournalFileReader(
                journal_path,
                scan_loader=self.scan_loader,
                mass_shift_map=self.mass_shift_map,
                **self.journal_reader_args),
            len(accumulator))
        i = float(len(accumulator))
        should_log = False
        for i, sol in reader:
            if i % 100000 == 0:
                should_log = True
            if should_log:
                self.log("... %d Solutions Loaded" % (i, ))
                should_log = False
            accumulator.append(sol)
        return accumulator
