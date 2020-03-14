from collections import defaultdict

from glycan_profiling import serialize

from glycan_profiling.chromatogram_tree import (
    Chromatogram, GlycopeptideChromatogram, ChromatogramTreeList,
    ChromatogramFilter)
from glycan_profiling.scoring import ChromatogramSolution
from glycan_profiling.tandem.chromatogram_mapping import TandemAnnotatedChromatogram
from glycan_profiling.tandem.spectrum_match import (
    ScoreSet, FDRSet, MultiScoreSpectrumMatch, MultiScoreSpectrumSolutionSet)
from glycan_profiling.tandem.glycopeptide.identified_structure import IdentifiedGlycopeptide


class FakeSpectrumMatch(MultiScoreSpectrumMatch):
    def __init__(self, target, mass_shift=None):
        super(FakeSpectrumMatch, self).__init__(
            scan=None, target=target, score_set=ScoreSet(1e-6, 1e-6, 1e-6, 0),
            best_match=True, data_bundle=None, q_value_set=FDRSet(0.99, 0.99, 0.99, 0.99),
            mass_shift=mass_shift, match_type=0)


class MatchBetweenDataset(object):
    def __init__(self, analysis_loader, scan_loader, identified_structures, chromatograms, label=None):
        if label is None:
            label = analysis_loader.analysis.name
        self.analysis_loader = analysis_loader
        self.scan_loader = scan_loader
        self.identified_structures = list(identified_structures)
        self.chromatograms = ChromatogramFilter(chromatograms)
        self._prepare_structure_map()
        self.label = label

    @property
    def ms1_scoring_model(self):
        return self.analysis_loader.analysis.parameters.get('scoring_model')

    def find(self, ids, mass_error_tolerance=1e-5, time_error_tolerance=2.0):
        key = str(ids.structure)
        out = []
        id_out = []
        ids_mass = ids.weighted_neutral_mass
        ids_apex_time = ids.apex_time
        if ids_apex_time is None:
            ids_apex_time = ids.tandem_solutions[0].scan_time
        if key in self._find_by_structure:
            results = self._find_by_structure[key]
            for result in results:
                if abs(result.weighted_neutral_mass - ids_mass) / ids_mass < mass_error_tolerance:
                    apex_time = result.apex_time
                    if apex_time is None:
                        apex_time = result.tandem_solutions[0].scan_time
                    if abs(apex_time - ids_apex_time) < time_error_tolerance:
                        id_out.append((result, None))
        for mshift in ids.mass_shifts:
            qmass = mshift.mass + ids_mass
            chroma = self.chromatograms.find_all_by_mass(
                qmass, mass_error_tolerance)
            for chrom in chroma:
                if abs(chrom.apex_time - ids_apex_time) < time_error_tolerance:
                    if not isinstance(chrom, Chromatogram) and (chrom, None) in id_out:
                        continue
                    out.append((chrom, mshift))
        return id_out + out

    def get_identified_structure_for(self, structure):
        key = str(structure)
        candidates = self._find_by_structure[key]
        for candidate in candidates:
            if candidate.structure == structure:
                return candidate
        raise KeyError("Could not locate %r (%r)" % (structure, structure.protein_relation))

    def create(self, structure, chromatogram, shift):
        chrom = GlycopeptideChromatogram(
            structure, ChromatogramTreeList())
        chrom = chrom.merge(chromatogram, shift)
        ms1_model = self.ms1_scoring_model
        chrom = ChromatogramSolution(
            chrom, ms1_model.logitscore(chrom), scorer=ms1_model)
        chrom = TandemAnnotatedChromatogram(chrom)
        sset = MultiScoreSpectrumSolutionSet(
            None, [FakeSpectrumMatch(structure)])
        sset.q_value = sset.best_solution().q_value
        idgp = IdentifiedGlycopeptide(structure, [sset], chrom)
        return idgp

    def merge(self, structure, chromatogram, shift):
        candidate = self.get_identified_structure_for(structure)
        chrom = GlycopeptideChromatogram(
            structure, ChromatogramTreeList())
        chrom = chrom.merge(chromatogram, shift)
        chrom = TandemAnnotatedChromatogram(chrom)
        candidate.chromatogram = candidate.chromatogram.merge(chrom)

    def add(self, ids):
        key = str(ids.structure)
        self._find_by_structure[key].append(ids)
        self.identified_structures.append(ids)
        self.chromatograms.extend([ids])

    def _protein_name_label_map(self):
        proteins = self.analysis_loader.query(serialize.Protein).all()
        name_map = dict()
        for prot in proteins:
            name_map[prot.id] = prot.name
        return name_map

    def _prepare_structure_map(self):
        self._find_by_structure = defaultdict(list)
        name_map = self._protein_name_label_map()
        for ids in self.identified_structures:
            try:
                pr = ids.protein_relation
                pr.protein_id = name_map.get(pr.protein_id, pr.protein_id)
            except AttributeError:
                pass
            self._find_by_structure[(ids.structure)].append(ids)
