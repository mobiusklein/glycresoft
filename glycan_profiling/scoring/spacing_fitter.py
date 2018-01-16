import numpy as np

from .base import ScoringFeatureBase, epsilon


def total_intensity(peaks):
    return sum(p.intensity for p in peaks)


def binsearch(array, x):
    lo = 0
    hi = len(array)
    while hi != lo:
        mid = (hi + lo) // 2
        y = array[mid]
        err = y - x
        if abs(err) < 1e-4:
            return mid
        elif hi - 1 == lo:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


class TimeOffsetIndex(object):
    def __init__(self, array):
        self.array = np.array(array)
        self.average_delta = self.estimate_average_delta()

    def estimate_average_delta(self, weights=None):
        if weights is None:
            weights = np.ones(len(self) - 1)
        return np.average(self[1:] - self[:-1], weights=weights)

    def index_for(self, x):
        return binsearch(self.array, x)

    def __getitem__(self, i):
        return self.array[i]

    def __len__(self):
        return len(self.array)

    def delta(self, x):
        i = self.index_for(x)
        if i == 0:
            return self.average_delta
        y = self.array[i + 1]
        return y - x


def blunt(x):
    if x < 0.1:
        return x
    elif 0.1 < x < 0.5:
        return np.sqrt(x) / 3.5
    else:
        return x


class ChromatogramSpacingFitter(ScoringFeatureBase):
    feature_type = "spacing_fit"

    def __init__(self, chromatogram, *args, **kwargs):
        transform_fn = kwargs.get("transform_fn")
        if transform_fn is None:
            def transform_fn(x):
                return x
        self.chromatogram = chromatogram
        self.rt_deltas = []
        self.intensity_deltas = []
        self.score = None
        self.transform_fn = transform_fn

        if len(chromatogram) < 3:
            self.score = 1.0
        else:
            self.fit()

    def transform(self, d_rt):
        return self.transform_fn(d_rt)

    def fit(self):
        times, intensities = self.chromatogram.as_arrays()
        last_rt = times[0]
        last_int = intensities[0]

        for rt, inten in zip(times[1:], intensities[1:]):
            d_rt = rt - last_rt
            self.rt_deltas.append(self.transform(d_rt))
            self.intensity_deltas.append(abs(last_int - inten))
            last_rt = rt
            last_int = inten

        self.rt_deltas = np.array(self.rt_deltas, dtype=np.float16)
        self.intensity_deltas = np.array(self.intensity_deltas, dtype=np.float32) + 1

        self.score = np.average(self.rt_deltas, weights=self.intensity_deltas)

    def __repr__(self):
        return "ChromatogramSpacingFitter(%s, %0.4f)" % (self.chromatogram, self.score)

    @classmethod
    def score(cls, chromatogram, *args, **kwargs):
        return max(1 - 2 * cls(chromatogram, *args, **kwargs).score, epsilon)


class RelativeScaleChromatogramSpacingFitter(ChromatogramSpacingFitter):

    def __init__(self, chromatogram, index, *args, **kwargs):
        self.index = index
        super(RelativeScaleChromatogramSpacingFitter, self).__init__(
            chromatogram, *args, **kwargs)

    def fit(self):
        times, intensities = self.chromatogram.as_arrays()
        last_rt = times[0]
        last_int = intensities[0]

        for rt, inten in zip(times[1:], intensities[1:]):
            d_rt = rt - last_rt
            scale = d_rt / self.index.delta(rt)
            self.rt_deltas.append(self.transform(d_rt) * scale)
            self.intensity_deltas.append(abs(last_int - inten))
            last_rt = rt
            last_int = inten

        self.rt_deltas = np.array(self.rt_deltas, dtype=np.float16)
        self.intensity_deltas = np.array(self.intensity_deltas, dtype=np.float32) + 1

        self.score = np.average(self.rt_deltas, weights=self.intensity_deltas)


class PartitionAwareRelativeScaleChromatogramSpacingFitter(RelativeScaleChromatogramSpacingFitter):
    def __init__(self, chromatogram, index, gap_size=0.25, *args, **kwargs):
        self.gap_size = gap_size
        self.partitions = [0]
        self.intensities = []
        super(PartitionAwareRelativeScaleChromatogramSpacingFitter, self).__init__(
            chromatogram, index, *args, **kwargs)

    def best_partition(self):
        i = 0
        n = len(self.partitions) - 1
        abundance = 0
        best_score = 0
        for i in range(n):
            start = self.partitions[i]
            end = self.partitions[i + 1] - 1
            score = np.average(
                self.rt_deltas[start:end],
                weights=self.intensity_deltas[start:end])
            current_abundance = np.sum(self.intensities[start:end])
            if current_abundance > abundance:
                abundance = current_abundance
                best_score = score

        return best_score

    def fit(self):
        times, intensities = self.chromatogram.as_arrays()
        last_rt = times[0]
        last_int = intensities[0]

        i = 1
        for rt, inten in zip(times[1:], intensities[1:]):
            d_rt = rt - last_rt
            if d_rt > self.gap_size:
                self.partitions.append(i)
            scale = d_rt / self.index.delta(rt)
            self.rt_deltas.append(self.transform(d_rt) * scale)
            self.intensity_deltas.append(abs(last_int - inten))
            self.intensities.append(inten)
            last_rt = rt
            last_int = inten
            i += 1

        self.partitions.append(i - 1)

        self.rt_deltas = np.array(self.rt_deltas, dtype=np.float16)
        self.intensity_deltas = np.array(self.intensity_deltas, dtype=np.float32) + 1
        self.intensities = np.array(self.intensities, dtype=np.float32)

        self.score = self.best_partition()


class ChromatogramSpacingModel(ScoringFeatureBase):
    feature_type = 'spacing_fit'

    def __init__(self, index=None, gap_size=0.25):
        self.index = index
        self.gap_size = gap_size
        self.transform_fn = None

    def configure(self, analysis_data):
        peak_loader = analysis_data['peak_loader']
        gap_size = analysis_data['delta_rt']
        self.index = TimeOffsetIndex(peak_loader.ms1_scan_times())
        tic = peak_loader.extract_total_ion_current_chromatogram()
        self.index.average_delta = self.index.estimate_average_delta(tic[1:])
        self.gap_size = gap_size
        if self.index.average_delta > 0.2:
            def transform_fn(x):
                return x / (self.index.average_delta * 15)
        else:
            transform_fn = None
        self.transform_fn = transform_fn
        return {
            "index": self.index,
            "gap_size": self.gap_size,
            "transform_fn": self.transform_fn
        }

    def fit(self, chromatogram):
        if self.index is None:
            return ChromatogramSpacingFitter(chromatogram)
        else:
            return PartitionAwareRelativeScaleChromatogramSpacingFitter(
                chromatogram, index=self.index,
                gap_size=self.gap_size, transform_fn=self.transform_fn)

    def score(self, chromatogram, *args, **kwargs):
        if self.index is None:
            return ChromatogramSpacingFitter.score(chromatogram)
        else:
            return PartitionAwareRelativeScaleChromatogramSpacingFitter.score(
                chromatogram, index=self.index, gap_size=self.gap_size,
                transform_fn=self.transform_fn)
