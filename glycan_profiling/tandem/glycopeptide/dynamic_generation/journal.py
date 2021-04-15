'''Implementation for journal file reading and writing to serialize
glycopeptide spectrum match information to disk during processing.
'''
import csv
import json

from collections import defaultdict
from operator import attrgetter

import numpy as np

from glycopeptidepy.utils import collectiontools

from ms_deisotope.output import ProcessedMzMLDeserializer

from glycan_profiling.task import TaskBase, TaskExecutionSequence, Empty

from glycan_profiling.structure.structure_loader import (
    FragmentCachingGlycopeptide, DecoyFragmentCachingGlycopeptide,
    PeptideProteinRelation, LazyGlycopeptide)
from glycan_profiling.structure.scan import ScanInformationLoader
from glycan_profiling.structure.lru import LRUMapping
from glycan_profiling.chromatogram_tree import MassShift, Unmodified

from glycan_profiling.tandem.ref import SpectrumReference
from glycan_profiling.tandem.spectrum_match import (
    MultiScoreSpectrumMatch, MultiScoreSpectrumSolutionSet, ScoreSet, FDRSet)

from .search_space import glycopeptide_key_t, StructureClassification


class JournalFileWriter(TaskBase):
    """A task for writing glycopeptide spectrum matches to a TSV-formatted
    journal file. This format is an intermediary result, and will contain many
    random or non-useful matches.

    """
    def __init__(self, path, include_fdr=False, include_auxiliary=False):
        self.path = path
        if not hasattr(path, 'write'):
            self.handle = open(path, 'wt')
        else:
            self.handle = self.path
        self.include_fdr = include_fdr
        self.include_auxiliary = include_auxiliary
        self.writer = csv.writer(self.handle, delimiter='\t')
        self.write_header()
        self.spectrum_counter = 0
        self.solution_counter = 0

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
            'glycopeptide_sequence',
            'mass_shift',
            'total_score',
            'peptide_score',
            'glycan_score',
            'glycan_coverage',
        ]
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

    def _prepare_fields(self, psm):
        error = (psm.target.total_mass - psm.precursor_information.neutral_mass
                ) / psm.precursor_information.neutral_mass
        fields = ([psm.scan_id, error, ] + list(psm.target.id) + [
            psm.target,
            psm.mass_shift.name,
            psm.score,
            psm.score_set.peptide_score,
            psm.score_set.glycan_score,
            psm.score_set.glycan_coverage,
        ])
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

    def write(self, psm):
        self.solution_counter += 1
        self.writer.writerow(self._prepare_fields(psm))

    def writeall(self, solution_sets):
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
                self.log("... Handled %d spectra with %d solutions so far" % (
                    self.journal_file.spectrum_counter, self.journal_file.solution_counter))
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()

    def close_stream(self):
        self.journal_file.close()


def parse_float(value):
    value = float(value)
    if np.isnan(value):
        return 0
    return value


try:
    from glycan_profiling._c.tandem.tandem_scoring_helpers import parse_float
except ImportError:
    pass


class JournalFileReader(TaskBase):
    def __init__(self, path, cache_size=2 ** 16, mass_shift_map=None, scan_loader=None, include_fdr=False):
        if mass_shift_map is None:
            mass_shift_map = {Unmodified.name: Unmodified}
        else:
            mass_shift_map.setdefault(Unmodified.name, Unmodified)
        self.path = path
        if not hasattr(path, 'read'):
            self.handle = open(path, 'rt')
        else:
            self.handle = self.path
        self.reader = csv.DictReader(self.handle, delimiter='\t')
        self.glycopeptide_cache = LRUMapping(cache_size or 2 ** 12)
        self.mass_shift_map = mass_shift_map
        self.scan_loader = scan_loader
        self.include_fdr = include_fdr

    def _build_key(self, row):
        glycopeptide_id_key = glycopeptide_key_t(
            int(row['peptide_start']), int(row['peptide_end']), int(
                row['peptide_id']), int(row['protein_id']),
            int(row['hypothesis_id']), int(row['glycan_combination_id']),
            StructureClassification[row['match_type']],
            int(row['site_combination_index']))
        return glycopeptide_id_key

    def _build_protein_relation(self, glycopeptide_id_key):
        return PeptideProteinRelation(
            glycopeptide_id_key.start_position, glycopeptide_id_key.end_position,
            glycopeptide_id_key.protein_id, glycopeptide_id_key.hypothesis_id)

    def glycopeptide_from_row(self, row):
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

    def _build_score_set(self, row):
        score_set = ScoreSet(
            parse_float(row['total_score']),
            parse_float(row['peptide_score']),
            parse_float(row['glycan_score']),
            float(row['glycan_coverage']))
        return score_set

    def _build_fdr_set(self, row):
        fdr_set = FDRSet(
            float(row['total_q_value']),
            float(row['peptide_q_value']),
            float(row['glycan_q_value']),
            float(row['glycopeptide_q_value']))
        return fdr_set

    def _make_mass_shift(self, row):
        # mass_shift = MassShift(row['mass_shift'], MassShift.get(row['mass_shift']))
        mass_shift = self.mass_shift_map[row['mass_shift']]
        return mass_shift

    def _make_scan(self, row):
        if self.scan_loader is None:
            scan = SpectrumReference(row['scan_id'])
        else:
            scan = self.scan_loader.get_scan_by_id(row['scan_id'])
        return scan

    def spectrum_match_from_row(self, row):
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

    def __next__(self):
        return self.spectrum_match_from_row(next(self.reader))

    def next(self):
        return self.__next__()

    def close(self):
        self.handle.close()


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= atol + rtol * abs(b)


class SolutionSetGrouper(TaskBase):
    """Partition multi-score glycopeptide identificatins into groups
    according to either glycopeptide target/decoy classification (:attr:`match_type_groups`)
    or by best match to a given scan and target/decoy classification (:attr:`exclusive_match_groups`)

    """
    def __init__(self, spectrum_matches):
        self.spectrum_matches = list(spectrum_matches)
        self.spectrum_ids = set()
        self.match_type_groups = self._collect()
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

    def _collect(self):
        match_type_getter = attrgetter('match_type')
        groups = collectiontools.groupby(
            self.spectrum_matches, match_type_getter)
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

    def _exclusive(self, score_getter=None, min_value=0):
        if score_getter is None:
            score_getter = attrgetter('score')
        groups = collectiontools.groupby(
            self.spectrum_matches, lambda x: x.scan.id)
        by_match_type = defaultdict(list)
        for _scan_id, members in groups.items():
            top_match = max(members, key=score_getter)
            top_score = score_getter(top_match)
            seen = set()
            for match in members:
                if isclose(top_score, score_getter(match)) and score_getter(match) > 0 and match.match_type not in seen:
                    seen.add(match.match_type)
                    by_match_type[match.match_type].append(match)
        for _group_label, matches in by_match_type.items():
            matches.sort(key=lambda x: (x.scan.id, score_getter(x)))
        return by_match_type

    @property
    def target_matches(self):
        try:
            return self.match_type_groups[StructureClassification.target_peptide_target_glycan]
        except KeyError:
            return []

    @property
    def decoy_matches(self):
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

    @classmethod
    def from_analysis(cls, analysis, scan_loader=None, stub_wrapping=True):
        mass_shift_map = {
            m.name: m for m in analysis.parameters['mass_shifts']}
        if scan_loader is None:
            scan_loader = ProcessedMzMLDeserializer(analysis.parameters['sample_path'])
        if stub_wrapping:
            stub_loader = ScanInformationLoader(scan_loader)
        else:
            stub_loader = scan_loader
        return cls([f.open() for f in analysis.files], stub_loader, mass_shift_map)

    def __init__(self, journal_files, scan_loader, mass_shift_map=None):
        if mass_shift_map is None:
            mass_shift_map = {Unmodified.name: Unmodified}
        self.journal_files = journal_files
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

    def _load_identifications_from_journal(self, journal_path, accumulator=None):
        if accumulator is None:
            accumulator = []
        reader = enumerate(JournalFileReader(
            journal_path,
            scan_loader=ScanInformationLoader(self.scan_loader),
            mass_shift_map=self.mass_shift_map), len(accumulator))
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
