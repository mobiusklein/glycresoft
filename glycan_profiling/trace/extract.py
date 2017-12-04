from collections import defaultdict

import numpy as np

from glycan_profiling.task import TaskBase
from glycan_profiling.chromatogram_tree import (
    ChromatogramForest,
    SimpleChromatogram,
    find_truncation_points,
    ChromatogramFilter)


class ChromatogramExtractor(TaskBase):
    def __init__(self, peak_loader, truncate=False, minimum_mass=500, grouping_tolerance=1.5e-5,
                 minimum_intensity=250., min_points=3, delta_rt=0.25):
        self.peak_loader = peak_loader
        self.truncate = truncate
        self.minimum_mass = minimum_mass
        self.minimum_intensity = minimum_intensity
        self.grouping_tolerance = grouping_tolerance

        self.min_points = min_points
        self.delta_rt = delta_rt

        self.accumulated = None
        self.annotated_peaks = None
        self.peak_mapping = None

        self.chromatograms = None
        self.base_peak_chromatogram = None
        self.total_ion_chromatogram = None

    def get_scan_by_id(self, scan_id):
        return self.peak_loader.get_scan_by_id(scan_id)

    def get_scan_header_by_id(self, scan_id):
        return self.peak_loader.get_scan_header_by_id(scan_id)

    def get_index_information_by_scan_id(self, scan_id):
        return self.peak_loader.get_index_information_by_scan_id(scan_id)

    def scan_id_to_rt(self, scan_id):
        return self.peak_loader.convert_scan_id_to_retention_time(scan_id)

    def load_peaks(self):
        self.accumulated = self.peak_loader.ms1_peaks_above(min(500, self.minimum_mass), self.minimum_intensity)
        self.annotated_peaks = [x[:2] for x in self.accumulated]
        self.peak_mapping = {x[:2]: x[2] for x in self.accumulated}
        self.minimum_intensity = np.percentile([p[1].intensity for p in self.accumulated], 5)

    def aggregate_chromatograms(self):
        forest = ChromatogramForest([], self.grouping_tolerance, self.scan_id_to_rt)
        forest.aggregate_peaks(self.annotated_peaks, self.minimum_mass, self.minimum_intensity)
        chroma = list(forest)
        self.log("%d Chromatograms Extracted." % (len(chroma),))
        self.chromatograms = ChromatogramFilter.process(
            chroma, min_points=self.min_points, delta_rt=self.delta_rt)

    def summary_chromatograms(self):
        mapping = defaultdict(list)
        for scan_id, peak in self.annotated_peaks:
            mapping[scan_id].append(peak.intensity)
        bpc = SimpleChromatogram(self)
        tic = SimpleChromatogram(self)
        collection = sorted(mapping.items(), key=lambda b: self.scan_id_to_rt(b[0]))
        for scan_id, intensities in collection:
            bpc[scan_id] = max(intensities)
            tic[scan_id] = sum(intensities)
        self.base_peak_chromatogram = bpc
        self.total_ion_chromatogram = tic

    def run(self):
        self.log("... Begin Extracting Chromatograms")
        self.load_peaks()
        self.aggregate_chromatograms()
        self.summary_chromatograms()
        if self.truncate:
            self.chromatograms = ChromatogramFilter(
                self.truncate_chromatograms(self.chromatograms))
        return self.chromatograms

    def truncate_chromatograms(self, chromatograms):
        start, stop = find_truncation_points(*self.total_ion_chromatogram.as_arrays())
        out = []
        for c in chromatograms:
            if len(c) == 0:
                continue
            c.truncate_before(start)
            if len(c) == 0:
                continue
            c.truncate_after(stop)
            if len(c) == 0:
                continue
            out.append(c)
        return out

    def __iter__(self):
        if self.chromatograms is None:
            self.run()
        return iter(self.chromatograms)
