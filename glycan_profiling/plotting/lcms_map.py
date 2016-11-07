from collections import defaultdict

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from glycan_profiling import serialize


def _make_color_map():
    # Derived from the seaborn.cubehelix_palette function
    start = .5
    rot = - .75
    gamma = 1.0
    hue = 0.8
    light = 0.85
    dark = 0.15
    cdict = matplotlib._cm.cubehelix(gamma, start, rot, hue)
    cmap = matplotlib.colors.LinearSegmentedColormap("cubehelix", cdict)
    x_256 = np.linspace(light, dark, 256)
    pal_256 = cmap(x_256)
    cmap = matplotlib.colors.ListedColormap(pal_256)
    return cmap


_color_map = _make_color_map()


def extract_intensity_array(peaks):
    mzs = []
    intensities = []
    rts = []
    current_mzs = []
    current_intensities = []
    last_time = None
    for peak, time in peaks:
        if time != last_time:
            if last_time is not None:
                mzs.append(current_mzs)
                intensities.append(current_intensities)
                assert len(current_mzs) == len(current_intensities)
                rts.append(time)
            last_time = time
            print(time)
            current_mzs = []
            current_intensities = []
        current_mzs.append(peak.mz)
        current_intensities.append(peak.intensity)
    mzs.append(current_mzs)
    intensities.append(current_intensities)
    rts.append(time)
    return mzs, intensities, rts


def binner(x):
    return np.floor(np.array(x) / 10.) * 10


def make_map(mzs, intensities):
    binned_mzs = [
        binner(mz_row) for mz_row in mzs
    ]
    unique_mzs = set()
    map(unique_mzs.update, binned_mzs)
    unique_mzs = np.array(sorted(unique_mzs))
    assigned_bins = []
    j = 0
    for mz, inten in zip(mzs, intensities):
        j += 1
        mz = binner(mz)
        bin_row = defaultdict(float)
        for i in range(len(mz)):
            k = mz[i]
            v = inten[i]
            bin_row[k] += v
        array_row = np.array([bin_row[m] for m in unique_mzs])
        assigned_bins.append(array_row)

    assigned_bins = np.vstack(assigned_bins)
    assigned_bins[assigned_bins == 0] = 1
    return assigned_bins, unique_mzs


def render_map(assigned_bins, rts, unique_mzs, ax=None, color_map=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    if color_map is None:
        color_map = _color_map
    ax.pcolormesh(np.array(np.sqrt(assigned_bins.T)), cmap=color_map)

    xticks = ax.get_xticks()
    newticks = xticks
    newlabels = np.array(ax.get_xticklabels())[np.arange(0, len(xticks))]
    n = len(newlabels)
    step = len(rts) / n
    interp = [rts[i * step] for i in range(n)]

    for i, label in enumerate(newlabels):
        num = round(interp[i], 1)
        label.set_rotation(90)
        label.set_text(str((num)))

    ax.set_xticks(newticks)
    ax.set_xticklabels(newlabels)

    yticks = ax.get_yticks()
    n = len(yticks)

    newticks = yticks[np.arange(0, len(yticks))]
    newlabels = np.array(ax.get_yticklabels())[np.arange(0, len(yticks))]
    va = unique_mzs
    n = len(newlabels)
    step = len(va) / n
    interp = [va[i * step] for i in range(n)]
    for i, label in enumerate(newlabels):
        num = interp[i]
        label.set_text(str((num)))
    ax.set_yticks(newticks)
    ax.set_yticklabels(newlabels)

    ax.set_xlabel("Retention Time")
    ax.set_ylabel("m/z")
    return ax


def get_peak_time_pairs(peak_loader):
    thresh = peak_loader.query(
        serialize.func.sum(
            serialize.FittedPeak.intensity) / serialize.func.count(
            serialize.FittedPeak.intensity)).join(serialize.MSScan).filter(
        serialize.MSScan.sample_run_id == peak_loader.sample_run_id,
        serialize.MSScan.ms_level == 1).scalar()
    peaks = peak_loader.query(serialize.FittedPeak, serialize.MSScan.scan_time).join(serialize.MSScan).filter(
        serialize.MSScan.sample_run_id == peak_loader.sample_run_id,
        serialize.MSScan.ms_level == 1,
        serialize.FittedPeak.intensity > thresh).order_by(
        serialize.MSScan.scan_time.asc()).yield_per(1000)
    return peaks


class LCMSMapArtist(object):

    def __init__(self, peak_time_pairs, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        self.peak_time_pairs = peak_time_pairs
        self.ax = ax

    def draw(self):
        mzs, intensities, rts = extract_intensity_array(self.peak_time_pairs)
        binned_intensities, unique_mzs = make_map(mzs, intensities)
        render_map(binned_intensities, rts, unique_mzs, self.ax)
        return self

    @classmethod
    def from_peak_loader(cls, peak_loader, ax=None):
        return cls(get_peak_time_pairs(peak_loader), ax=ax)
