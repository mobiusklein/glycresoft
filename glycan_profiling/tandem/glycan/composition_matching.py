
from glycan_profiling.task import TaskBase
from .scoring.signature_ion_scoring import SignatureIonScorer
from ..chromatogram_mapping import ChromatogramMSMSMapper


class SignatureIonMapper(TaskBase):

    # simple default value from experimentation
    minimum_score = 0.034367

    def __init__(self, tandem_scans, chromatograms, scan_id_to_rt=lambda x: x,
                 adducts=None, minimum_mass=500, chunk_size=1000,
                 default_glycan_composition=None, scorer_type=None):
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
        self.scorer_type = scorer_type

    def prepare_scan_set(self, scan_set):
        if hasattr(scan_set[0], 'convert'):
            out = []
            # Account for cases where the scan may be mentioned in the index, but
            # not actually present in the MS data
            for o in scan_set:
                if o.precursor_information.neutral_mass < self.minimum_mass:
                    continue
                try:
                    out.append(self.scorer_type.load_peaks(o))
                except KeyError as e:
                    self.log("Missing Scan: %s (%r)" % (o.id, e))
            scan_set = out
        else:
            out = []
            for o in scan_set:
                if o.precursor_information.neutral_mass < self.minimum_mass:
                    continue
                out.append(o)
        return out

    def map_to_chromatograms(self, precursor_error_tolerance=1e-5):
        mapper = ChromatogramMSMSMapper(
            self.chromatograms, error_tolerance=precursor_error_tolerance,
            scan_id_to_rt=self.scan_id_to_rt)
        for scan in self.prepare_scan_set(self.tandem_scans):
            hits = mapper.find_chromatogram_spanning(scan.scan_time)
            if hits is None:
                continue
            match = hits.find_all_by_mass(
                scan.precursor_information.neutral_mass,
                precursor_error_tolerance)
            if not match:
                continue
            for m in match:
                m.add_solution(scan)
            for adduct in self.adducts:
                match = hits.find_all_by_mass(
                    scan.precursor_information.neutral_mass - adduct.mass,
                    precursor_error_tolerance)
                if not match:
                    continue
                for m in match:
                    m.add_solution(scan)
        return mapper

    def score_mapped_tandem(self, annotated_chromatograms, *args, **kwargs):
        i = 0
        ni = len(annotated_chromatograms)
        for chroma in annotated_chromatograms:
            i += 1
            if i % 500 == 0:
                self.log("... Handling chromatogram %d/%d (%0.3f%%)" % (i, ni, (i * 100. / ni)))
            tandem_scans = chroma.tandem_solutions
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
                    scan, glycan_composition, include_compound=True,
                    *args, **kwargs)
                solutions.append(solution)
            chroma.tandem_solutions = solutions
        return annotated_chromatograms
