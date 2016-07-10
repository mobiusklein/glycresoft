import ms_peak_picker
import ms_deisotope


from ms_deisotope.processor import MzMLLoader


class ScanGenerator(object):
    def __init__(self, mzml_file, averagine=ms_deisotope.averagine.glycan):
        self.loader = MzMLLoader(mzml_file, use_index=True)
        self._iterator = None
        self.averagine_cache = ms_deisotope.averagine.AveragineCache(averagine)
        self.time_cache = {}

    def configure_iteration(self, start_scan=None, end_scan=None, max_scans=None):
        self._iterator = self.make_iterator(start_scan, end_scan, max_scans)

    def make_iterator(self, start_scan=None, end_scan=None, max_scans=None):
        if start_scan is not None:
            self.loader.start_from_scan(start_scan)
        i = 0
        for scan, _ in self.loader:
            self.time_cache[scan.id] = scan.scan_time
            if len(scan.arrays[0]):
                yield deconvolve(pick_peaks(scan), self.averagine_cache)
            if scan.id == end_scan or i == max_scans:
                break
            i += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = self.make_iterator()
        return next(self._iterator)

    def convert_scan_id_to_retention_time(self, scan_id):
        return self.time_cache[scan_id]

    next = __next__


def pick_peaks(scan):
    scan.pick_peaks(transforms=[
        ms_peak_picker.scan_filter.FTICRBaselineRemoval(scale=2.),
        ms_peak_picker.scan_filter.SavitskyGolayFilter()], start_mz=200)
    return scan


def deconvolve(scan, averagine=ms_deisotope.averagine.glycan):
    dp, _ = ms_deisotope.deconvolution.deconvolute_peaks(scan.peak_set, charge_range=(-1, -8), **{
        "averagine": averagine,
        "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(15)
    })
    scan.deconvoluted_peak_set = dp
    return scan


def to_csv(generator, out):
    writer = out
    writer.write(','.join(["scan_id", 'rt', "mz", "charge", "intensity", "fwhm", "score", "a_to_a2_ratio"]) + '\n')
    for scan in generator:
        for peak in scan.deconvoluted_peak_set:
            if peak.mz < 300:
                continue
            writer.write(','.join(map(str, [
                scan.id, scan.scan_time, peak.mz, peak.charge, peak.intensity,
                peak.full_width_at_half_max, peak.score, peak.a_to_a2_ratio])) + "\n")
    return out
