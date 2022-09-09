from typing import List, DefaultDict

from array import ArrayType as array

import numpy as np

from ms_deisotope.data_source import ProcessedScan
from glycan_profiling.structure import FragmentCachingGlycopeptide


from glycan_profiling.tandem.spectrum_match.spectrum_match import ScanMatchManagingMixin


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
