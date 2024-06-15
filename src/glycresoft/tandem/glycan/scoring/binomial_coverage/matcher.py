from ms_deisotope import CompositionListPeakDependenceGraphDeconvoluter, DistinctPatternFitter

from glycresoft.database.composition_aggregation import Formula, CommonComposition, Composition
from glycresoft.tandem.spectrum_match import SpectrumMatcherBase


from .score_components import IntensityRankScorer, FragmentScorer


def get_fragments(assigned_peaks):
    results = set()
    for peak in assigned_peaks:
        results.update(peak.solution.base_data)
    return results


class BinomialCoverageSpectrumMatcher(SpectrumMatcherBase):
    """Encapsulates the process of matching experimental peaks to theoretical peaks.

    Given a :class:`ms_peak_picker.PeakIndex` from a :class:`MSMSScan` and a :class:`StructureRecord`
    instance, this type will determine the set of all compositions needed to search, handling ambiguous
    compositions and managing the searching process.

    Attributes
    ----------
    assigned_peaks : list
        The list of deconvolved peaks that were assigned compositions
    composition_list : list
        The list of compositions to attempt to assign
    matched_fragments : list
        The list of theoretical fragments for which compositions were assigned
    peak_set : ms_peak_picker.PeakIndex
        The list of experimental peaks
    structure_record : StructureRecord
        The structure to search for fragments from
    """
    def __init__(self, scan, target):
        super(BinomialCoverageSpectrumMatcher, self).__init__(scan, target)
        self.peak_set = self.scan.peak_set
        self.composition_list = None
        self.assigned_peaks = None
        self.matched_fragments = None

        self.make_composition_list()

    @property
    def structure(self):
        return self.target

    def make_composition_list(self):
        """Aggregate all theoretical compositions by formula so that each composition
        is present exactly once, containing all possible sources of that composition in
        a single :class:`CommonComposition` instance.

        Sets the :attr:`composition_list` attribute. Called during initialization.
        """
        self.composition_list = CommonComposition.aggregate(
            Formula(f.composition, f) for f in self.structure.fragment_map.values())

        # Is there anything added by including the precursor? Maybe in the graph solving step
        # f = Formula(self.structure_record.structure.total_composition(), self.structure_record)
        # self.composition_list.append(CommonComposition(f, [f.data]))

    def match(self, mass_tolerance=5e-6, charge_carrier=Composition("Na").mass, fitter_parameters=None):
        """Perform the actual matching of peaks, fitting isotopic patterns and retrieving the matched
        structural features.

        Parameters
        ----------
        mass_tolerance : float, optional
            The parts-per-million mass error tolerance allowed. Defaults to 5ppm, entered as 5e-6
        charge_range : tuple, optional
            The a tuple defining the minimum and maximum of the range of charge states to consider
            for each composition. Defaults to positive (1, 3)
        charge_carrier : float, optional
            The mass of the charge carrying element. By default for the experimental set up described
            here, this is the mass of a sodium (Na)

        Returns
        -------
        SpectrumSolution
        """

        if fitter_parameters is None:
            fitter_parameters = {}

        charge_range = (1, self.scan.precursor_information.charge)
        dec = CompositionListPeakDependenceGraphDeconvoluter(
            self.peak_set, self.composition_list,
            scorer=DistinctPatternFitter(**fitter_parameters))
        self.assigned_peaks = dec.deconvolute(
            charge_range=charge_range, charge_carrier=charge_carrier,
            error_tolerance=mass_tolerance)
        self.matched_fragments = list(get_fragments(self.assigned_peaks))

        self._structure_scorer = FragmentScorer(self.structure, self.matched_fragments)
        self._spectrum_scorer = IntensityRankScorer(self.peak_set, self.assigned_peaks)
        self.score = self._spectrum_scorer.final_score + self._structure_scorer.final_score
