from glycan_profiling.chromatogram_tree import ChromatogramWrapper, build_rt_interval_tree
from glycan_profiling.trace import ChromatogramFilter


class TandemAnnotatedChromatogram(ChromatogramWrapper):
    def __init__(self, chromatogram):
        super(TandemAnnotatedChromatogram, self).__init__(chromatogram)
        self.tandem_solutions = []
        self.time_displaced_assignments = []

    def add_solution(self, item):
        self.tandem_solutions.append(item)

    def add_displaced_solution(self, item):
        self.time_displaced_assignments.append(item)

    def merge(self, other):
        new = self.__class__(self.chromatogram.merge(other.chromatogram))
        new.tandem_solutions = self.tandem_solutions + other.tandem_solutions
        new.time_displaced_assignments = self.time_displaced_assignments + other.time_displaced_assignments
        return new

    def merge_in_place(self, other):
        new = self.chromatogram.merge(other.chromatogram)
        self.chromatogram = new
        self.tandem_solutions = self.tandem_solutions + other.tandem_solutions
        self.time_displaced_assignments = self.time_displaced_assignments + other.time_displaced_assignments


class ScanTimeBundle(object):
    def __init__(self, solution, scan_time):
        self.solution = solution
        self.scan_time = scan_time

    def __hash__(self):
        return hash((self.solution, self.scan_time))

    def __eq__(self, other):
        return self.solution == other.solution and self.scan_time == other.scan_time

    def __repr__(self):
        return "ScanTimeBundle(%s, %0.4f)" % (self.solution, self.scan_time)


class ChromatogramMSMSMapper(object):
    def __init__(self, chromatograms, error_tolerance=1e-5, scan_id_to_rt=lambda x: x):
        self.chromatograms = ChromatogramFilter(map(
            TandemAnnotatedChromatogram, chromatograms))
        self.rt_tree = build_rt_interval_tree(self.chromatograms)
        self.scan_id_to_rt = scan_id_to_rt
        self.orphans = []
        self.error_tolerance = error_tolerance

    def _find_chromatogram_spanning(self, time):
        return ChromatogramFilter([interv[0] for interv in self.rt_tree.contains_point(time)])

    def find_chromatogram_for(self, solution):
        precursor_scan_time = self.scan_id_to_rt(
            solution.scan.precursor_information.precursor_scan_id)
        overlapping_chroma = self._find_chromatogram_spanning(precursor_scan_time)
        chroma = overlapping_chroma.find_mass(
            solution.scan.precursor_information.neutral_mass, self.error_tolerance)
        if chroma is None:
            self.orphans.append(ScanTimeBundle(solution, precursor_scan_time))
        else:
            chroma.tandem_solutions.append(solution)

    def assign_solutions_to_chromatograms(self, solutions):
        for solution in solutions:
            self.find_chromatogram_for(solution)

    def distribute_orphans(self):
        for orphan in self.orphans:
            mass = orphan.solution.precursor_ion_mass()
            window = self.error_tolerance * mass
            candidates = self.chromatograms.mass_between(mass - window, mass + window)
            time = orphan.scan_time
            if len(candidates) > 0:
                best_index = 0
                best_distance = float('inf')
                for i, candidate in enumerate(candidates):
                    dist = min(abs(candidate.start_time - time), abs(candidate.end_time - time))
                    if dist < best_distance:
                        best_index = i
                        best_distance = dist
                    new_owner = candidates[best_index]
                    new_owner.add_displaced_solution(orphan)

    def __len__(self):
        return len(self.chromatograms)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.chromatograms[i]
        else:
            return [self.chromatograms[j] for j in i]
