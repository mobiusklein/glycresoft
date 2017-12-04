from collections import defaultdict

from multiprocessing import Manager as IPCManager

from glycopeptidepy.structure.glycan import GlycanCompositionProxy

from glycan_profiling.task import TaskBase
from .scoring.signature_ion_scoring import SignatureIonScorer
from ..chromatogram_mapping import ChromatogramMSMSMapper

from ..process_dispatcher import (
    IdentificationProcessDispatcher,
    SpectrumIdentificationWorkerBase)


class ChromatogramAssignmentRecord(GlycanCompositionProxy):
    def __init__(self, glycan_composition, id_key, index):
        self.glycan_composition = glycan_composition
        self.id_key = id_key
        self.index = index
        GlycanCompositionProxy.__init__(self, glycan_composition)

    @property
    def id(self):
        return self.id_key

    def __repr__(self):
        return "{self.__class__.__name__}({self.glycan_composition}, {self.id_key}, {self.index})".format(
            self=self)

    @classmethod
    def build(cls, chromatogram, default_glycan_composition=None, index=None):
        key = (chromatogram.start_time, chromatogram.end_time, chromatogram.key)
        if chromatogram.glycan_composition:
            glycan_composition = chromatogram.glycan_composition
        else:
            glycan_composition = default_glycan_composition
        return cls(glycan_composition, key, index)


class GlycanCompositionIdentificationWorker(SpectrumIdentificationWorkerBase):
    def __init__(self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
                 spectrum_map, mass_shift_map, log_handler):
        SpectrumIdentificationWorkerBase.__init__(
            self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
            spectrum_map, mass_shift_map, log_handler=log_handler)

    def evaluate(self, scan, structure, *args, **kwargs):
        target = structure
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher


class SignatureIonMapper(TaskBase):

    # simple default value from experimentation
    minimum_score = 0.05

    def __init__(self, tandem_scans, chromatograms, scan_id_to_rt=lambda x: x,
                 adducts=None, minimum_mass=500, chunk_size=1000,
                 default_glycan_composition=None, scorer_type=None,
                 n_processes=4):
        if scorer_type is None:
            scorer_type = SignatureIonScorer
        if adducts is None:
            adducts = []
        self.chromatograms = chromatograms
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass,
            reverse=True)
        self.scan_id_to_rt = scan_id_to_rt
        self.adducts = adducts
        self.minimum_mass = minimum_mass
        self.default_glycan_composition = default_glycan_composition
        self.default_glycan_composition.id = -1
        self.scorer_type = scorer_type
        self.n_processes = n_processes
        self.ipc_manager = IPCManager()

    def prepare_scan_set(self, scan_set):
        if hasattr(scan_set[0], 'convert'):
            out = []
            # Account for cases where the scan may be mentioned in the index, but
            # not actually present in the MS data
            for o in scan_set:
                try:
                    scan = (self.scorer_type.load_peaks(o))
                    out.append(scan)
                except KeyError:
                    self.log("Missing Scan: %s" % (o.id,))
            scan_set = out
        out = []
        for scan in scan_set:
            try:
                out.append(scan)
            except AttributeError as e:
                self.log("Missing Scan: %s %r" % (scan.id, e))
                continue
        return out

    def map_to_chromatograms(self, precursor_error_tolerance=1e-5):
        mapper = ChromatogramMSMSMapper(
            self.chromatograms, error_tolerance=precursor_error_tolerance,
            scan_id_to_rt=self.scan_id_to_rt)
        for scan in self.tandem_scans:
            scan_time = mapper.scan_id_to_rt(scan.precursor_information.precursor_scan_id)
            hits = mapper.find_chromatogram_spanning(scan_time)
            if hits is None:
                continue
            match = hits.find_all_by_mass(
                scan.precursor_information.neutral_mass,
                precursor_error_tolerance)
            if match:
                for m in match:
                    m.add_solution(scan)
            for adduct in self.adducts:
                match = hits.find_all_by_mass(
                    scan.precursor_information.neutral_mass - adduct.mass,
                    precursor_error_tolerance)
                if match:
                    for m in match:
                        m.add_solution(scan)
        return mapper

    def _build_scan_to_entity_map(self, annotated_chromatograms):

        scan_map = dict()
        hit_map = dict()
        hit_to_scan = dict()
        default = self.default_glycan_composition
        for i, chroma in enumerate(annotated_chromatograms, 1):
            if not chroma.tandem_solutions:
                continue
            default = default.clone()
            default.id = -i
            record = ChromatogramAssignmentRecord.build(chroma, default, index=i - 1)
            hit_map[record.id] = record
            scans = []
            for scan in self.prepare_scan_set(chroma.tandem_solutions):
                scan_map[scan.id] = scan
                scans.append(scan.id)
            hit_to_scan[record.id] = scans
        return scan_map, hit_map, hit_to_scan

    def _chunk_chromatograms(self, chromatograms, chunk_size=3500):
        chunk = []
        k = 0
        for chroma in chromatograms:
            chunk.append(chroma)
            k += len(chroma.tandem_solutions)
            if k >= chunk_size:
                yield chunk
                chunk = []
                k = 0
        yield chunk

    def _score_mapped_tandem_parallel(self, annotated_chromatograms, *args, **kwargs):
        for chunk in self._chunk_chromatograms(annotated_chromatograms):
            scan_map, hit_map, hit_to_scan = self._build_scan_to_entity_map(chunk)
            ipd = IdentificationProcessDispatcher(
                worker_type=GlycanCompositionIdentificationWorker,
                scorer_type=self.scorer_type,
                init_args={}, evaluation_args=kwargs,
                n_processes=self.n_processes,
                ipc_manager=self.ipc_manager)
            scan_solution_map = ipd.process(scan_map, hit_map, hit_to_scan, defaultdict(lambda: "Unmodified"))

            for scan_id, matches in scan_solution_map.items():
                for match in matches:
                    chroma = chunk[match.target.index]
                    updated_scans = []
                    for scan in chroma.tandem_solutions:
                        if scan.scan_id == scan_id:
                            # match.scan = SpectrumReference(scan_id, scan.precursor_information)
                            match.target = match.target.glycan_composition
                            updated_scans.append(match)
                        else:
                            updated_scans.append(scan)
                    chroma.tandem_solutions = updated_scans
        return annotated_chromatograms

    def _score_mapped_tandem_sequential(self, annotated_chromatograms, *args, **kwargs):
        i = 0

        ni = len(annotated_chromatograms)
        for chroma in annotated_chromatograms:
            i += 1
            if i % 500 == 0:
                self.log("... Handling chromatogram %d/%d (%0.3f%%)" % (i, ni, (i * 100. / ni)))
            tandem_scans = self.prepare_scan_set(chroma.tandem_solutions)
            if chroma.glycan_composition is None:
                if self.default_glycan_composition is None:
                    continue
                else:
                    glycan_composition = self.default_glycan_composition
            else:
                glycan_composition = chroma.glycan_composition
            solutions = []
            j = 0
            nj = len(tandem_scans)
            for scan in tandem_scans:
                j += 1
                if j % 500 == 0:
                    self.log("...... Handling spectrum match %d/%d (%0.3f%%)" % (j, nj, (j * 100. / nj)))

                solution = self.scorer_type.evaluate(
                    scan, glycan_composition,
                    *args, **kwargs)
                solutions.append(solution)
            chroma.tandem_solutions = solutions
        return annotated_chromatograms

    def score_mapped_tandem(self, annotated_chromatograms, *args, **kwargs):
        if self.n_processes > 1:
            annotated_chromatograms = self._score_mapped_tandem_parallel(
                annotated_chromatograms, **kwargs)
        else:
            annotated_chromatograms = self._score_mapped_tandem_sequential(
                annotated_chromatograms, **kwargs)
        return annotated_chromatograms
