'''A collection of odds-and-ends that are not heavily used or optimized.
'''

from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_filter1d


class ChromatogramDeltaNode(object):
    '''Represent a sub-region of a chromatogram to determine whether to truncate
    the chromatogram or not based upon whether or not they show large gaps in time
    or significant change in intensity over time.
    '''

    def __init__(self, retention_times, delta_intensity, start_time, end_time, is_below_threshold=True):
        self.retention_times = retention_times
        self.delta_intensity = delta_intensity
        self.start_time = start_time
        self.end_time = end_time
        self.mean_change = np.mean(delta_intensity)
        self.is_below_threshold = is_below_threshold

    def __repr__(self):
        return "ChromatogramDeltaNode(%f, %f, %f)" % (
            self.mean_change, self.start_time, self.end_time)

    @classmethod
    def partition(cls, rt, delta_smoothed, window_size=.5):
        last_rt = rt[1]
        last_index = 1
        nodes = []
        i = 0
        for i, rt_i in enumerate(rt[2:]):
            if (rt_i - last_rt) >= window_size:
                nodes.append(
                    cls(
                        rt[last_index:i],
                        delta_smoothed[last_index:i + 1],
                        last_rt, rt[i]))
                last_index = i
                last_rt = rt_i
        nodes.append(
            cls(
                rt[last_index:i],
                delta_smoothed[last_index:i + 1],
                last_rt, rt[i]))
        return nodes


def build_chromatogram_nodes(rt, signal, sigma=3):
    rt = np.array(rt)
    smoothed = gaussian_filter1d(signal, sigma)
    delta_smoothed = np.gradient(smoothed, rt)
    change = delta_smoothed[:-1] - delta_smoothed[1:]
    avg_change = change.mean()
    std_change = change.std()

    lo = avg_change - std_change
    hi = avg_change + std_change

    nodes = ChromatogramDeltaNode.partition(rt, delta_smoothed)

    for node in nodes:
        if lo > node.mean_change or node.mean_change > hi:
            node.is_below_threshold = False

    return nodes


def find_truncation_points(rt, signal, sigma=3, pad=3):
    nodes = build_chromatogram_nodes(rt, signal, sigma=3)

    leading = 0
    ending = len(nodes)

    for node in nodes:
        if not node.is_below_threshold:
            break
        leading += 1
    leading -= 3
    leading = max(leading - pad, 0)

    for node in reversed(nodes):
        if not node.is_below_threshold:
            break
        ending -= 1

    ending = min(ending + pad, len(nodes) - 1)
    if len(nodes) == 1:
        return nodes[0].start_time, nodes[0].end_time
    elif len(nodes) == 2:
        return nodes[0].start_time, nodes[-1].end_time
    return nodes[leading].start_time, nodes[ending].end_time


class SimpleChromatogram(OrderedDict):
    '''A simplified Chromatogram-like object which supports :meth:`as_arrays`
    and :meth:`get_chromatogram`, but otherwise acts as a mapping from retention
    time to intensity.
    '''
    def __init__(self, *args):
        super(SimpleChromatogram, self).__init__(*args)

    composition = None
    glycan_composition = None

    def as_arrays(self):
        return (
            np.array(list(self.keys())),
            np.array(list(self.values()))
        )

    def get_chromatogram(self):
        return self

    def _new(self):
        return self.__class__()

    def slice(self, start, end):
        pairs = []
        for t, v in self.items():
            if start <= t <= end:
                pairs.append((t, v))
        dup = self._new()
        dup.update(pairs)
        return dup

    def split_sparse(self, delta_rt=1.):
        parts = []
        start = 0
        last = None
        for i, t in enumerate(self.keys()):
            if last is None:
                last = t
                start = t
            if t - last >= delta_rt:
                parts.append(self.slice(start, last))
                start = t
            last = t
        if last != start:
            parts.append(self.slice(start, last))
        return parts

    @property
    def start_time(self):
        return next(iter(self.keys()))

    @property
    def end_time(self):
        return list(self.keys())[-1]

    @property
    def apex_time(self):
        time, intensity = self.as_arrays()
        i = np.argmax(intensity)
        return time[i]


class SimpleEntityChromatogram(SimpleChromatogram):

    def __init__(self, entity=None, glycan_composition=None):
        self.entity = entity
        self.composition = entity
        self.glycan_composition = glycan_composition
        super(SimpleEntityChromatogram, self).__init__()

    def _new(self):
        return self.__class__(self.entity, self.glycan_composition)


class PairedArray(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_arrays(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x)
