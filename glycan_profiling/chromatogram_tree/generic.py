from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_filter1d


class ChromatogramDeltaNode(object):
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
    def __init__(self, time_converter):
        self.time_converter = time_converter
        super(SimpleChromatogram, self).__init__()

    composition = None
    glycan_composition = None

    def as_arrays(self):
        return (
            np.array(map(self.time_converter.scan_id_to_rt, self)),
            np.array(self.values()))

    def get_chromatogram(self):
        return self


class PairedArray(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_arrays(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x)
