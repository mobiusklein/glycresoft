cimport cython
from cpython cimport (
    PyTuple_GetItem, PyTuple_Size, PyTuple_GET_ITEM, PyTuple_GET_SIZE,
    PyList_GET_ITEM, PyList_GET_SIZE,
    PySet_Add, PySet_Contains)

from libc.math cimport log10, log, sqrt, exp

import numpy as np
cimport numpy as np
np.import_array()
from numpy.math cimport isnan, log2l

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycan_profiling._c.structure.fragment_match_map cimport (
    FragmentMatchMap, PeakFragmentPair)

from glypy.composition.ccomposition cimport CComposition

from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore
from glycopeptidepy._c.structure.fragment cimport PeptideFragment, FragmentBase, IonSeriesBase, ChemicalShiftBase, SimpleFragment, StubFragment
from glycopeptidepy.structure.fragment import IonSeries, ChemicalShift
from glycopeptidepy.structure.fragmentation_strategy import HCDFragmentationStrategy


cdef:
    IonSeriesBase IonSeries_b, IonSeries_y, IonSeries_c, IonSeries_z, IonSeries_stub_glycopeptide

IonSeries_b = IonSeries.b
IonSeries_y = IonSeries.y
IonSeries_c = IonSeries.c
IonSeries_z = IonSeries.z
IonSeries_stub_glycopeptide = IonSeries.stub_glycopeptide
IonSeries_precursor = IonSeries.precursor


cdef object zeros = np.zeros
cdef object np_float64 = np.float64

@cython.binding(True)
@cython.boundscheck(False)
cpdef _compute_coverage_vectors(self):
    cdef:
        np.ndarray[np.float64_t, ndim=1] n_term_ions, c_term_ions
        int stub_count
        long size
        set glycosylated_n_term_ions, glycosylated_c_term_ions
        FragmentMatchMap solution_map
        FragmentBase frag
        PeptideFragment pep_frag
        _PeptideSequenceCore target

    target = <_PeptideSequenceCore>self.target
    size = target.get_size()

    n_term_ions = zeros(size, dtype=np_float64)
    c_term_ions = zeros(size, dtype=np_float64)
    stub_count = 0
    glycosylated_n_term_ions = set()
    glycosylated_c_term_ions = set()

    solution_map = self.solution_map

    for obj in solution_map.fragments():
        frag = <FragmentBase>obj
        series = frag.get_series()
        if series in (IonSeries_b, IonSeries_c):
            pep_frag = <PeptideFragment>frag
            n_term_ions[pep_frag.position] = 1
            if frag.is_glycosylated:
                glycosylated_n_term_ions.add((series, pep_frag.position))
        elif series in (IonSeries_y, IonSeries_z):
            pep_frag = <PeptideFragment>frag
            c_term_ions[pep_frag.position] = 1
            if frag.is_glycosylated:
                glycosylated_c_term_ions.add((series, pep_frag.position))
        elif series == IonSeries_stub_glycopeptide:
            stub_count += 1
    return n_term_ions, c_term_ions, stub_count, len(glycosylated_n_term_ions), len(glycosylated_c_term_ions)


cdef set peptide_ion_series = {IonSeries_b, IonSeries_y, IonSeries_c, IonSeries_z}


@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
def calculate_peptide_score(self, double error_tolerance=2e-5, double coverage_weight=1.0, *args, **kwargs):
    cdef:
        double total, score, coverage_score, normalizer
        # set seen
        tuple coverage_result
        PeakFragmentPair peak_pair
        DeconvolutedPeak peak
        FragmentMatchMap solution_map
        long size
        np.ndarray[np.float64_t, ndim=1] n_term, c_term
        size_t i

    target = <_PeptideSequenceCore>self.target
    size = target.get_size()

    total = 0
    # seen = set()
    solution_map = <FragmentMatchMap>self.solution_map
    for obj in solution_map.members:
        peak_pair = <PeakFragmentPair>obj
        peak = peak_pair.peak
        if (<FragmentBase>peak_pair.fragment).get_series() in peptide_ion_series:
            # seen.add(peak._index.neutral_mass)
            total += log10(peak.intensity) * (1 - (abs(peak_pair.mass_accuracy()) / error_tolerance) ** 4)
    coverage_result = <tuple>_compute_coverage_vectors(self)
    n_term = <np.ndarray[np.float64_t, ndim=1]>PyTuple_GetItem(coverage_result, 0)
    c_term = <np.ndarray[np.float64_t, ndim=1]>PyTuple_GetItem(coverage_result, 1)
    normalizer = float((2 * size - 1))
    coverage_score = 0.0

    for i in range(size):
        coverage_score += n_term[i] + c_term[size - i - 1]
    coverage_score /= normalizer

    score = total * coverage_score ** coverage_weight
    if isnan(score):
        return 0
    return score


@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
def calculate_glycan_score(self, double error_tolerance=2e-5, double core_weight=0.4, double coverage_weight=0.5,
                           bint fragile_fucose=False, bint extended_glycan_search=False, *args, **kwargs):
    cdef:
        set seen, core_fragments, core_matches, extended_matches
        IonSeriesBase series
        list theoretical_set
        double total, score
        double glycan_prior, glycan_coverage
        FragmentMatchMap solution_map
        FragmentBase frag
        PeakFragmentPair peak_pair
        DeconvolutedPeak peak
        int side_group_count
        size_t i, m
        object target
    target = self.target
    seen = set()
    series = IonSeries_stub_glycopeptide
    if not extended_glycan_search:
        theoretical_set = list(target.stub_fragments(extended=True))
    else:
        theoretical_set = list(target.stub_fragments(extended=True, extended_fucosylation=True))
    core_fragments = set()
    for i in range(len(theoretical_set)):
        frag = <FragmentBase>theoretical_set[i]
        if not frag.is_extended:
            core_fragments.add(frag._name)

    total = 0
    core_matches = set()
    extended_matches = set()
    solution_map = <FragmentMatchMap>self.solution_map

    for obj in solution_map.members:
        peak_pair = <PeakFragmentPair>obj
        if (<FragmentBase>peak_pair.fragment).get_series() != series:
            continue
        fragment_name = (<FragmentBase>peak_pair.fragment).base_name()
        if fragment_name in core_fragments:
            core_matches.add(fragment_name)
        else:
            extended_matches.add(fragment_name)
        peak = peak_pair.peak
        if peak._index.neutral_mass not in seen:
            seen.add(peak._index.neutral_mass)
            total += log10(peak.intensity) * (1 - (abs(peak_pair.mass_accuracy()) / error_tolerance) ** 4)

    m = PyList_GET_SIZE(theoretical_set)
    glycan_coverage = _compute_glycan_coverage_from_metrics(
        self, fragile_fucose, len(core_matches), len(extended_matches),
        len(core_fragments), m, core_weight, coverage_weight)
    score = total * glycan_coverage
    glycan_prior = 0.0
    self._glycan_coverage = glycan_coverage
    if glycan_coverage > 0:
        glycan_prior = target.glycan_prior
        score += glycan_coverage * glycan_prior
    if isnan(score):
        return 0
    return score


@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double _calculate_glycan_coverage(self, double core_weight=0.4, double coverage_weight=0.5,
                                        bint fragile_fucose=False, bint extended_glycan_search=False):
    cdef:
        set seen, core_fragments, core_matches, extended_matches
        IonSeriesBase series
        list theoretical_set
        double total, n, k, d, core_coverage, extended_coverage, score
        FragmentMatchMap solution_map
        StubFragment frag
        PeakFragmentPair peak_pair
        size_t i, n_core_matches, n_extended_matches, m

    if self._glycan_coverage is not None:
            return self._glycan_coverage
    seen = set()
    series = IonSeries_stub_glycopeptide
    if not extended_glycan_search:
        theoretical_set = list(self.target.stub_fragments(extended=True))
    else:
        theoretical_set = list(self.target.stub_fragments(extended=True, extended_fucosylation=True))
    core_fragments = set()
    m = len(theoretical_set)
    for i in range(m):
        frag = <StubFragment>theoretical_set[i]
        if not frag.is_extended:
            core_fragments.add(frag._name)

    total = 0
    core_matches = set()
    extended_matches = set()
    solution_map = <FragmentMatchMap>self.solution_map

    for obj in solution_map.members:
        peak_pair = <PeakFragmentPair>obj
        if (<FragmentBase>peak_pair.fragment).get_series() != series:
            continue
        elif peak_pair.fragment_name in core_fragments:
            core_matches.add(peak_pair.fragment_name)
        else:
            extended_matches.add(peak_pair.fragment_name)

    coverage = self._compute_glycan_coverage_from_metrics(
        fragile_fucose, len(core_matches), len(extended_matches),
        len(core_fragments), m, core_weight, coverage_weight)

    self._glycan_coverage = coverage
    return coverage


@cython.binding(True)
@cython.cdivision(True)
cpdef double _compute_glycan_coverage_from_metrics(self, bint fragile_fucose, size_t n_core_matches,
                                                   size_t n_extended_matches, size_t n_core_fragments, size_t n_fragments,
                                                   double core_weight, double coverage_weight):
    cdef:
        double glycan_composition_size_normalizer
        double core_coverage, extended_coverage
        double approximate_size, extra_branch_factor

        double coverage


    glycan_composition = self.target.glycan_composition
    approximate_size = self._get_internal_size(glycan_composition)
    extra_branch_factor = 2.0
    if not fragile_fucose:
        side_group_count = self._glycan_side_group_count(glycan_composition)
        if side_group_count > 0:
            extra_branch_factor = 1.0

    glycan_composition_size_normalizer = min(
        max(approximate_size * log(approximate_size) / extra_branch_factor, approximate_size),
        n_fragments)

    core_coverage = ((n_core_matches * 1.0) / n_core_fragments) ** core_weight
    extended_coverage = min((n_core_matches + n_extended_matches) / glycan_composition_size_normalizer, 1.0) ** coverage_weight
    coverage = core_coverage * extended_coverage
    if isnan(coverage):
        coverage = 0.0
    return coverage

@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
def SimpleCoverageScorer_match_backbone_series(self, IonSeriesBase series, double error_tolerance=2e-5,
                                               set masked_peaks=None, strategy=None, bint include_neutral_losses=False):
    cdef:
        list frags, fragments
        tuple peaks
        bint glycosylated_position, previous_position_glycosylated
        PeptideFragment frag
        FragmentMatchMap solution_map
        long glycosylated_term_ions_count
        DeconvolutedPeak peak
        DeconvolutedPeakSet spectrum
        size_t i, n, j, m, i_peaks, n_peaks

    if strategy is None:
        strategy = HCDFragmentationStrategy
    # Assumes that fragmentation proceeds from the start of the ladder (series position 1)
    # which means that if the last fragment could be glycosylated then the next one will be
    # but if the last fragment wasn't the next one might be.
    previous_position_glycosylated = False
    glycosylated_term_ions_count = 0

    spectrum = self.spectrum
    obj = self.get_fragments(series, strategy=strategy, include_neutral_losses=include_neutral_losses)
    if not isinstance(obj, list):
        fragments = list(obj)
    else:
        fragments = <list>obj

    solution_map = <FragmentMatchMap>self.solution_map

    n = PyList_GET_SIZE(fragments)
    for i in range(n):
        frags = <list>PyList_GET_ITEM(fragments, i)
        glycosylated_position = previous_position_glycosylated
        m = PyList_GET_SIZE(frags)
        for j in range(m):
            frag = <PeptideFragment>PyList_GET_ITEM(frags, j)
            if not glycosylated_position:
                glycosylated_position |= frag._is_glycosylated()
            peaks = <tuple>spectrum.all_peaks_for(frag.mass, error_tolerance)
            for i_peaks in range(PyTuple_Size(peaks)):
                peak = <DeconvolutedPeak>PyTuple_GetItem(peaks, i_peaks)
                if peak._index.neutral_mass in masked_peaks:
                    continue
                solution_map.add(peak, frag)
        if glycosylated_position:
            glycosylated_term_ions_count += 1
        previous_position_glycosylated = glycosylated_position
    if series.direction > 0:
        self.glycosylated_n_term_ion_count += glycosylated_term_ions_count
    else:
        self.glycosylated_c_term_ion_count += glycosylated_term_ions_count



@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
def CoverageWeightedBinomialScorer_match_backbone_series(self, IonSeriesBase series, double error_tolerance=2e-5,
                                                         set masked_peaks=None, strategy=None,
                                                         bint include_neutral_losses=False):
    cdef:
        list frags, fragments
        tuple peaks
        bint glycosylated_position, previous_position_glycosylated
        PeptideFragment frag
        FragmentMatchMap solution_map
        long glycosylated_term_ions_count
        long n_theoretical
        DeconvolutedPeak peak
        DeconvolutedPeakSet spectrum
        size_t i, n, j, m, i_peaks, n_peaks


    if strategy is None:
        strategy = HCDFragmentationStrategy

    n_theoretical = 0
    previous_position_glycosylated = False
    glycosylated_term_ions_count = 0

    spectrum = self.spectrum
    obj = self.get_fragments(series, strategy=strategy, include_neutral_losses=include_neutral_losses)
    if not isinstance(obj, list):
        fragments = list(obj)
    else:
        fragments = <list>obj

    solution_map = <FragmentMatchMap>self.solution_map

    n = PyList_GET_SIZE(fragments)
    for i in range(n):
        frags = <list>PyList_GET_ITEM(fragments, i)

        glycosylated_position = previous_position_glycosylated
        n_theoretical += 1
        m = PyList_GET_SIZE(frags)
        for j in range(m):
            frag = <PeptideFragment>PyList_GET_ITEM(frags, j)
            if not glycosylated_position:
                glycosylated_position |= frag._is_glycosylated()
            peaks = spectrum.all_peaks_for(frag.mass, error_tolerance)
            n_peaks = PyTuple_Size(peaks)
            for i_peaks in range(n_peaks):
                peak = <DeconvolutedPeak>PyTuple_GetItem(peaks, i_peaks)
                if peak._index.neutral_mass in masked_peaks:
                    continue
                solution_map.add(peak, frag)
        if glycosylated_position:
            glycosylated_term_ions_count += 1
        previous_position_glycosylated = glycosylated_position

    self.n_theoretical += n_theoretical
    if series.direction > 0:
        self.glycosylated_n_term_ion_count += glycosylated_term_ions_count
    else:
        self.glycosylated_c_term_ion_count += glycosylated_term_ions_count


@cython.binding(True)
@cython.nonecheck(False)
def _match_oxonium_ions(self, double error_tolerance=2e-5, set masked_peaks=None):
    cdef:
        list fragments
        FragmentBase frag
        DeconvolutedPeak peak
        DeconvolutedPeakSet spectrum
        FragmentMatchMap solution_map
        object ix

    if masked_peaks is None:
        masked_peaks = set()

    fragments = list(self.target.glycan_fragments(
            all_series=False, allow_ambiguous=False,
            include_large_glycan_fragments=False,
            maximum_fragment_size=4))

    spectrum = <DeconvolutedPeakSet>self.spectrum
    solution_map = <FragmentMatchMap>self.solution_map

    for i in range(len(fragments)):
        frag = <FragmentBase>PyList_GET_ITEM(fragments, i)
        peak = spectrum.has_peak(frag.mass, error_tolerance)

        if peak is not None:
            ix = peak._index.neutral_mass
            if ix not in masked_peaks:
                solution_map.add(peak, frag)
                masked_peaks.add(ix)
    return masked_peaks


@cython.binding(True)
def _match_stub_glycopeptides(self, double error_tolerance=2e-5, set masked_peaks=None,
                              ChemicalShiftBase chemical_shift=None, bint extended_glycan_search=False):

    cdef:
        list fragments
        tuple peaks, shifted_peaks
        DeconvolutedPeak peak
        size_t i, n, j, m, k
        DeconvolutedPeakSet spectrum
        FragmentMatchMap solution_map


    if masked_peaks is None:
        masked_peaks = set()

    if not extended_glycan_search:
        obj = self.target.stub_fragments(extended=True)
    else:
        obj = self.target.stub_fragments(extended=True, extended_fucosylation=True)
    if isinstance(obj, list):
        fragments = <list>obj
    else:
        fragments = list(obj)

    spectrum = <DeconvolutedPeakSet>self.spectrum
    solution_map = <FragmentMatchMap>self.solution_map

    for i in range(PyList_GET_SIZE(fragments)):
        frag = <FragmentBase>PyList_GET_ITEM(fragments, i)

        peaks = spectrum.all_peaks_for(frag.mass, error_tolerance)
        for j in range(PyTuple_Size(peaks)):
            peak = <DeconvolutedPeak>PyTuple_GetItem(peaks, j)
            # should we be masking these? peptides which have amino acids which are
            # approximately the same mass as a monosaccharide unit at ther terminus
            # can produce cases where a stub ion and a backbone fragment match the
            # same peak.
            #
            masked_peaks.add(peak._index.neutral_mass)
            solution_map.add(peak, frag)
        if chemical_shift is not None:
            shifted_mass = frag.mass + chemical_shift.mass
            shifted_peaks = spectrum.all_peaks_for(shifted_mass, error_tolerance)
            for k in range(PyTuple_Size(shifted_peaks)):
                peak = <DeconvolutedPeak>PyTuple_GetItem(shifted_peaks, k)
                masked_peaks.add(peak.index.neutral_mass)

                shifted_frag = frag.clone()
                shifted_frag.set_chemical_shift(chemical_shift)
                solution_map.add(peak, shifted_frag)
    return masked_peaks


@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef base_peak(self):
    cdef:
        DeconvolutedPeak peak
        DeconvolutedPeakSet spectrum
        size_t i, n
    spectrum = <DeconvolutedPeakSet>self.spectrum
    if spectrum is None:
        return 0
    n = spectrum.get_size()
    max_peak = None
    max_intensity = 0
    for i in range(n):
        peak = spectrum.getitem(i)
        if peak.intensity > max_intensity:
            max_intensity = peak.intensity
            max_peak = peak
    return max_intensity


@cython.boundscheck(False)
cpdef DeconvolutedPeak base_peak_tuple(tuple peaks):
    cdef:
        size_t i, n
        DeconvolutedPeak peak, best_peak

    n = PyTuple_Size(peaks)
    if n == 0:
        return None
    else:
        best_peak = <DeconvolutedPeak>PyTuple_GetItem(peaks, 0)
    for i in range(1, n):
        peak = <DeconvolutedPeak>PyTuple_GetItem(peaks, i)
        if peak.intensity > best_peak.intensity:
            best_peak = peak
    return best_peak


cdef double sqrt2pi = sqrt(2 * np.pi)


ctypedef fused scalar_or_array:
    double
    np.ndarray


@cython.cdivision(True)
cpdef scalar_or_array gauss(scalar_or_array x, double mu, double sigma):
    if scalar_or_array is np.ndarray:
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * sqrt2pi)
    else:
        return exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * sqrt2pi)


@cython.binding(True)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef PeptideSpectrumMatcherBase_match_backbone_series(self, IonSeriesBase series, double error_tolerance=2e-5,
                                                       set masked_peaks=None, strategy=None, bint include_neutral_losses=False):
    cdef:
        list frags, fragments
        tuple peaks
        PeptideFragment frag
        FragmentMatchMap solution_map
        DeconvolutedPeak peak
        DeconvolutedPeakSet spectrum
        size_t i, n, j, m, i_peaks, n_peaks

    if strategy is None:
        strategy = HCDFragmentationStrategy

    spectrum = self.spectrum
    obj = self.get_fragments(series, strategy=strategy, include_neutral_losses=include_neutral_losses)
    if not isinstance(obj, list):
        fragments = list(obj)
    else:
        fragments = <list>obj

    solution_map = <FragmentMatchMap>self.solution_map

    n = PyList_GET_SIZE(fragments)
    for i in range(n):
        frags = <list>PyList_GET_ITEM(fragments, i)
        m = PyList_GET_SIZE(frags)
        for j in range(m):
            frag = <PeptideFragment>PyList_GET_ITEM(frags, j)
            peaks = <tuple>spectrum.all_peaks_for(frag.mass, error_tolerance)
            n_peaks = PyTuple_GET_SIZE(peaks)
            for i_peaks in range(n_peaks):
                peak = <DeconvolutedPeak>PyTuple_GET_ITEM(peaks, i_peaks)
                if PySet_Contains(masked_peaks, peak._index.neutral_mass):
                    continue
                solution_map.add(peak, frag)


@cython.binding(True)
@cython.boundscheck(False)
cpdef tuple _peptide_compute_coverage_vectors(self):
    cdef:
        np.ndarray[np.float64_t, ndim=1] n_term_ions, c_term_ions
        long size
        FragmentMatchMap solution_map
        FragmentBase frag
        PeptideFragment pep_frag
        _PeptideSequenceCore target

    target = <_PeptideSequenceCore>self.target
    size = target.get_size()

    n_term_ions = zeros(size, dtype=np_float64)
    c_term_ions = zeros(size, dtype=np_float64)

    solution_map = self.solution_map

    for obj in solution_map.fragments():
        frag = <FragmentBase>obj
        series = frag.get_series()
        if series.name in (IonSeries_b.name, IonSeries_c.name):
            pep_frag = <PeptideFragment>frag
            n_term_ions[pep_frag.position] = 1
        elif series.name in (IonSeries_y.name, IonSeries_z.name):
            pep_frag = <PeptideFragment>frag
            c_term_ions[pep_frag.position] = 1
    return n_term_ions, c_term_ions


cdef double log2_3 = log2l(3)

@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef compute_coverage(self):
    cdef:
        np.ndarray[np.float64_t, ndim=1] n_term_ions, c_term_ions
        long size, i
        _PeptideSequenceCore target
        double acc, mean_coverage
        tuple vectors

    target = <_PeptideSequenceCore>self.target
    size = target.get_size()
    vectors = <tuple>self._compute_coverage_vectors()
    n_term_ions = <np.ndarray>PyTuple_GetItem(vectors, 0)
    c_term_ions = <np.ndarray>PyTuple_GetItem(vectors, 1)
    acc = 0.0

    for i in range(size):
        acc += log2l(n_term_ions[i] + c_term_ions[size - (i + 1)] + 1) / log2_3
    mean_coverage = acc / size
    return mean_coverage


cpdef parse_float(str text):
    cdef:
        object value
    value = float(text)
    if isnan(value):
        return 0.0
    return value


cdef CComposition H2O_loss = -CComposition("H2O")
cdef CComposition NH3_loss = -CComposition("NH3")

cdef double NH3_loss_mass = NH3_loss.mass
cdef double H2O_loss_mass = H2O_loss.mass

cdef ChemicalShiftBase NH3_loss_shift = ChemicalShift("-NH3", NH3_loss)
cdef ChemicalShiftBase H2O_loss_shift = ChemicalShift("-H2O", H2O_loss)

@cython.binding(True)
cpdef _match_precursor(self, double error_tolerance=2e-5, set masked_peaks=None, bint include_neutral_losses=False):

    cdef:
        list frags, fragments
        tuple peaks
        SimpleFragment frag, shifted_frag
        FragmentMatchMap solution_map
        DeconvolutedPeak peak
        DeconvolutedPeakSet spectrum
        size_t i_peaks

    if masked_peaks is None:
        masked_peaks = set()

    mass = self.target.total_mass
    frag = SimpleFragment("M", mass, IonSeries_precursor, None)

    spectrum = self.spectrum
    solution_map = <FragmentMatchMap>self.solution_map
    peaks = <tuple>spectrum.all_peaks_for(frag.mass, error_tolerance)
    for i_peaks in range(PyTuple_Size(peaks)):
        peak = <DeconvolutedPeak>PyTuple_GET_ITEM(peaks, i_peaks)
        key = <object>peak._index.neutral_mass
        if PySet_Contains(masked_peaks, key):
            continue
        PySet_Add(masked_peaks, key)
        solution_map.add(peak, frag)
    if include_neutral_losses:
        peaks = <tuple>spectrum.all_peaks_for(frag.mass + NH3_loss_mass, error_tolerance)
        for i_peaks in range(PyTuple_Size(peaks)):
            peak = <DeconvolutedPeak>PyTuple_GET_ITEM(peaks, i_peaks)
            key = <object>peak._index.neutral_mass
            if PySet_Contains(masked_peaks, key):
                continue
            PySet_Add(masked_peaks, key)
            shifted_frag = frag.clone()
            shifted_frag.set_chemical_shift(NH3_loss_shift)
            solution_map.add(peak, shifted_frag)
        peaks = <tuple>spectrum.all_peaks_for(frag.mass + H2O_loss_mass, error_tolerance)
        for i_peaks in range(PyTuple_Size(peaks)):
            peak = <DeconvolutedPeak>PyTuple_GET_ITEM(peaks, i_peaks)
            key = <object>peak._index.neutral_mass
            if PySet_Contains(masked_peaks, key):
                continue
            PySet_Add(masked_peaks, key)
            shifted_frag = frag.clone()
            shifted_frag.set_chemical_shift(H2O_loss_shift)
            solution_map.add(peak, shifted_frag)



cdef long _factorial(long x):
    cdef:
        long i
        long acc

    acc = 1
    for i in range(x, 0, -1):
        acc *= i
    return acc


@cython.binding(True)
def _calculate_hyperscore(self, *args, **kwargs):
    cdef:
        FragmentMatchMap solution_map
        PeakFragmentPair pfp
        PeptideFragment pep_frag
        double n_term_intensity
        double c_term_intensity
        long n_term
        long c_term
        double hyper


    n_term_intensity = 0
    c_term_intensity = 0
    n_term = 0
    c_term = 0
    hyper = 0

    solution_map = self.solution_map

    for obj in solution_map.members:
        pfp = <PeakFragmentPair>obj
        series = (<FragmentBase>pfp.fragment).get_series()
        if series.name in (IonSeries_b.name, IonSeries_c.name):
            n_term += 1
            n_term_intensity += pfp.peak.intensity
        elif series.name in (IonSeries_y.name, IonSeries_z.name):
            c_term += 1
            c_term_intensity += pfp.peak.intensity

    hyper += log(n_term_intensity)
    hyper += log(_factorial(n_term))

    hyper += log(c_term_intensity)
    hyper += log(_factorial(c_term))
    return hyper


cpdef double fast_coarse_score(DeconvolutedPeakSet peak_set, _PeptideSequenceCore target, double error_tolerance=2e-5):
    cdef:
        size_t i, n, j, m, k, l
        list fragments, frags
        double acc
        object fragout
        PeptideFragment frag
        tuple peaks
        DeconvolutedPeak peak

    acc = 0.0
    fragout = target.get_fragments("b")
    if isinstance(fragout, list):
        fragments = <list>fragout
    else:
        fragments = list(fragout)
    n = PyList_GET_SIZE(fragments)
    for i in range(n):
        frags = <list>PyList_GET_ITEM(fragments, i)
        m = PyList_GET_SIZE(frags)
        for j in range(m):
            frag = <PeptideFragment>PyList_GET_ITEM(frags, j)
            peaks = peak_set.all_peaks_for(frag.mass, error_tolerance)
            l = PyTuple_GET_SIZE(peaks)
            for k in range(l):
                peak = <DeconvolutedPeak>PyTuple_GET_ITEM(peaks, k)
                acc += log10(peak.intensity)

    fragout = target.get_fragments("y")
    if isinstance(fragout, list):
        fragments = <list>fragout
    else:
        fragments = list(fragout)
    n = PyList_GET_SIZE(fragments)
    for i in range(n):
        frags = <list>PyList_GET_ITEM(fragments, i)
        m = PyList_GET_SIZE(frags)
        for j in range(m):
            frag = <PeptideFragment>PyList_GET_ITEM(frags, j)
            peaks = peak_set.all_peaks_for(frag.mass, error_tolerance)
            l = PyTuple_GET_SIZE(peaks)
            for k in range(l):
                peak = <DeconvolutedPeak>PyTuple_GET_ITEM(peaks, k)
                acc += log10(peak.intensity)
    return acc