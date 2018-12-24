import csv

from collections import defaultdict
from operator import attrgetter

import numpy as np

from glycopeptidepy.utils import collectiontools

from glycan_profiling.task import TaskBase, TaskExecutionSequence, Empty

from glycan_profiling.structure.structure_loader import FragmentCachingGlycopeptide, PeptideProteinRelation
from glycan_profiling.structure.lru import LRUMapping
from glycan_profiling.chromatogram_tree import MassShift, Unmodified

from glycan_profiling.tandem.ref import SpectrumReference
from glycan_profiling.tandem.spectrum_match import (
    MultiScoreSpectrumMatch, MultiScoreSpectrumSolutionSet, ScoreSet)

from .search_space import glycopeptide_key_t, StructureClassification


class JournalFileWriter(TaskBase):
    def __init__(self, path):
        self.path = path
        self.path = path
        if not hasattr(path, 'write'):
            self.handle = open(path, 'wb')
        else:
            self.handle = self.path
        self.writer = csv.writer(self.handle, delimiter='\t')
        self.write_header()
        self.spectrum_counter = 0
        self.solution_counter = 0

    def write_header(self):
        self.writer.writerow([
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
        ])

    def _prepare_fields(self, psm):
        error = (psm.target.total_mass - psm.precursor_information.neutral_mass
                 ) / psm.precursor_information.neutral_mass
        fields = map(str, [psm.scan_id, error, ] + list(psm.target.id) + [
            psm.target,
            psm.mass_shift.name,
            psm.score,
            psm.score_set.peptide_score,
            psm.score_set.glycan_score
        ])
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
        while has_work:
            try:
                solutions = self.in_queue.get(True, 5)
                self.journal_file.writeall(solutions)
                self.log("... Handled %d spectrum solutions so far\n" % self.journal_file.spectrum_counter)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


class JournalFileReader(TaskBase):
    def __init__(self, path, cache_size=2 ** 12, mass_shift_map=None, precursor_information_map=None):
        if mass_shift_map is None:
            mass_shift_map = {Unmodified.name: Unmodified}
        if precursor_information_map is None:
            precursor_information_map = {}
        self.path = path
        if not hasattr(path, 'read'):
            self.handle = open(path, 'rb')
        else:
            self.handle = self.path
        self.reader = csv.DictReader(self.handle, delimiter='\t')
        self.glycopeptide_cache = LRUMapping(cache_size or 2 ** 12)
        self.mass_shift_map = mass_shift_map
        self.precursor_information_map = precursor_information_map

    def glycopeptide_from_row(self, row):
        glycopeptide_id_key = glycopeptide_key_t(
            int(row['peptide_start']), int(row['peptide_end']), int(row['peptide_id']), int(row['protein_id']),
            int(row['hypothesis_id']), int(row['glycan_combination_id']),
            StructureClassification[row['match_type']],
            int(row['site_combination_index']))
        if glycopeptide_id_key in self.glycopeptide_cache:
            return self.glycopeptide_cache[glycopeptide_id_key]
        glycopeptide = FragmentCachingGlycopeptide(row['glycopeptide_sequence'])
        glycopeptide.id = glycopeptide_id_key
        glycopeptide.protein_relation = PeptideProteinRelation(
            glycopeptide_id_key.start_position, glycopeptide_id_key.end_position,
            glycopeptide_id_key.protein_id, glycopeptide_id_key.hypothesis_id)
        self.glycopeptide_cache[glycopeptide_id_key] = glycopeptide
        return glycopeptide

    def spectrum_match_from_row(self, row):
        glycopeptide = self.glycopeptide_from_row(row)
        scan = SpectrumReference(
            row['scan_id'],
            self.precursor_information_map.get(row['scan_id']))
        score_set = ScoreSet(
            float(row['total_score']), float(row['peptide_score']),
            float(row['glycan_score']))
        mass_shift = MassShift(row['mass_shift'], MassShift.get(row['mass_shift']))
        match = MultiScoreSpectrumMatch(
            scan, glycopeptide, score_set, mass_shift=mass_shift,
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
    def __init__(self, spectrum_matches):
        self.spectrum_matches = list(spectrum_matches)
        self.spectrum_ids = set()
        self.match_type_groups = self._collect()
        self.exclusive_match_groups = self._exclusive()

    def __getitem__(self, key):
        return self.exclusive_match_groups[key]

    def __iter__(self):
        return iter(self.exclusive_match_groups.items())

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
            by_scan_groups[group] = acc
        return by_scan_groups

    def _exclusive(self):
        groups = collectiontools.groupby(
            self.spectrum_matches, lambda x: x.scan.id)
        score_getter = attrgetter('score')
        by_match_type = defaultdict(list)
        for scan_id, members in groups.items():
            top_match = max(members, key=score_getter)
            top_score = top_match.score
            seen = set()
            for match in members:
                if isclose(top_score, match.score) and match.score > 0 and match.match_type not in seen:
                    seen.add(match.match_type)
                    by_match_type[match.match_type].append(match)
        return by_match_type

    @property
    def target_matches(self):
        return self.match_type_groups[StructureClassification.target_peptide_target_glycan]

    @property
    def decoy_matches(self):
        return self.match_type_groups[StructureClassification.decoy_peptide_target_glycan]

    def target_count(self):
        return len(self.target_matches)

    def decoy_count(self):
        return len(self.decoy_matches)
