from typing import DefaultDict, Dict, Tuple

from array import ArrayType as array

import numpy as np

from ms_deisotope.data_source import ProcessedScan
from glycresoft.structure import FragmentCachingGlycopeptide


from glycresoft.tandem.spectrum_match.spectrum_match import ScanMatchManagingMixin


class PeakFragmentVector(ScanMatchManagingMixin):
    scan: ProcessedScan
    target: FragmentCachingGlycopeptide
    error_tolerance: float
    max_charge: int
    charge_vectors: DefaultDict[int, array]

    def __init__(self, scan: ProcessedScan, target: FragmentCachingGlycopeptide,
                 error_tolerance: float=2e-5, max_charge: int=2):
        super().__init__(scan, target)
        self.error_tolerance = error_tolerance
        self.max_charge = max_charge
        self.charge_vectors = DefaultDict(lambda: array('d'))
        self.build()

    def build_for_series(self, ion_series):
        zs = np.ones(self.max_charge)
        for frags in self.target.get_fragments(ion_series):
            for frag in frags:
                zs[:] = 0
                for peak in self.scan.deconvoluted_peak_set.all_peaks_for(frag.mass, self.error_tolerance):
                    z = abs(peak.charge)
                    if z > self.max_charge:
                        continue
                    zs[z - 1] = peak.intensity
                for z, i in enumerate(zs, 1):
                    self.charge_vectors[z].append(i)

    def build(self):
        if self.is_hcd():
            self.build_for_series('b')
            self.build_for_series('y')
        if self.is_exd():
            self.build_for_series('c')
            self.build_for_series('z')

    def flatten(self):
        acc = array('d')
        for k, v in self.charge_vectors.items():
            acc.extend(v)
        return np.asarray(acc)

    def correlate(self, other: 'PeakFragmentVector') -> float:
        xy = 0
        xx = 0
        yy = 0
        for z in range(1, self.max_charge + 1):
            x = np.asarray(self.charge_vectors[z])
            y = np.asarray(other.charge_vectors[z])
            xy += x.dot(y)
            xx += x.dot(x)
            yy += y.dot(y)
        return xy / (np.sqrt(xx) * np.sqrt(yy))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target})"


class GlycanPeakFragmentVector(ScanMatchManagingMixin):
    scan: ProcessedScan
    target: FragmentCachingGlycopeptide
    error_tolerance: float
    max_charge: int
    fragment_map: Dict[Tuple[str, int], float]
    total: float

    def __init__(self, scan, target, error_tolerance: float=2e-5, max_charge: int = None):
        super().__init__(scan, target)
        if max_charge is None:
            max_charge = self.precursor_information.charge
        self.max_charge = max_charge
        self.error_tolerance = error_tolerance
        self.total = 0.0
        self.fragment_map = {}
        self.build()

    def build(self):
        total = 0.0
        for frag in self.target.stub_fragments(extended=True, extended_fucosylation=True):
            for peak in self.scan.deconvoluted_peak_set.all_peaks_for(frag.mass, self.error_tolerance):
                key = str(frag.glycosylation), peak.charge
                if key in self.fragment_map:
                    self.fragment_map[key] += peak.intensity
                else:
                    self.fragment_map[key] = peak.intensity
                total += peak.intensity
        self.total = total

    def correlate(self, other: 'GlycanPeakFragmentVector'):
        xy = 0
        xx = 0
        yy = 0
        keyspace = self.fragment_map.keys() | other.fragment_map.keys()
        for k in keyspace:
            x = self.fragment_map.get(k, 0)
            y = other.fragment_map.get(k, 0)
            xy += x * y
            xx += x ** 2
            yy += y ** 2
        return xy / (np.sqrt(xx) * np.sqrt(yy))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target})"


try:
    from glycresoft._c.tandem.tandem_scoring_helpers import correlate_fragment_map

    GlycanPeakFragmentVector.correlate = correlate_fragment_map
except ImportError:
    pass
