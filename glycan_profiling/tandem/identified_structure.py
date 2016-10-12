class IdentifiedStructure(object):
    def __init__(self, structure, spectrum_matches, chromatogram, shared_with=None):
        if shared_with is None:
            shared_with = []
        self.structure = structure
        self.spectrum_matches = spectrum_matches
        self.chromatogram = chromatogram
        self.ms2_score = max(s.score for s in spectrum_matches)
        self.ms1_score = chromatogram.score
        self.total_signal = chromatogram.total_signal
        self.charge_states = chromatogram.charge_states
        self.shared_with = shared_with

    def __repr__(self):
        return "IdentifiedStructure(%s, %0.3f, %0.3f, %0.3e)" % (
            self.structure, self.ms2_score, self.ms1_score, self.total_signal)

    def get_chromatogram(self):
        return self.chromatogram

    @property
    def tandem_solutions(self):
        return self.spectrum_matches

    @property
    def glycan_composition(self):
        return self.structure.glycan_composition

    def __eq__(self, other):
        try:
            structure_eq = self.structure == other.structure
            if structure_eq:
                chromatogram_eq = self.chromatogram == other.chromatogram
                if chromatogram_eq:
                    spectrum_matches_eq = self.spectrum_matches == other.spectrum_matches
                    return spectrum_matches_eq
            return False
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.structure, self.chromatogram))


def extract_identified_structures(tandem_annotated_chromatograms, result_type=IdentifiedStructure):
    identified_structures = []

    for chroma in tandem_annotated_chromatograms:
        if chroma.composition is not None:
            if hasattr(chroma, 'entity'):
                representers = chroma.most_representative_solutions()
                bunch = []
                for representer in representers:
                    ident = result_type(representer.solution, chroma.tandem_solutions, chroma, [])
                    bunch.append(ident)
                for i, ident in enumerate(bunch):
                    ident.shared_with = bunch[:i] + bunch[i + 1:]
                identified_structures.extend(bunch)
    return identified_structures
