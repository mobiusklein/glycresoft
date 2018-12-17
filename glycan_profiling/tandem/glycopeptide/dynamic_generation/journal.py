import csv

from glycan_profiling.task import TaskBase, TaskExecutionSequence, Empty


class JournalFileWriter(TaskBase):
    def __init__(self, path):
        self.path = path
        self.handle = open(self.path, 'wb')
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
            'glycopeptide_sequence',
            'mass_shift',
            'total_score',
            'peptide_score',
            'glycan_score',
        ])

    def write(self, psm):
        error = (psm.target.total_mass - psm.precursor_information.neutral_mass
                 ) / psm.precursor_information.neutral_mass
        self.solution_counter += 1
        self.writer.writerow(
            map(str, [psm.scan_id, error, ] + list(psm.target.id) + [
                psm.target,
                psm.mass_shift.name,
                psm.score,
                psm.score_set.peptide_score,
                psm.score_set.glycan_score
            ]))

    def writeall(self, solution_sets):
        for solution_set in solution_sets:
            self.spectrum_counter += 1
            for solution in solution_set:
                self.write(solution)

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

    def process(self):
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
