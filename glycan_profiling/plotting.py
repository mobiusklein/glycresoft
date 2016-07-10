from collections import defaultdict, OrderedDict
from itertools import cycle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
from scipy.ndimage import gaussian_filter1d


import glypy
from glycresoft_sqlalchemy.report import colors

from .trace import count_charge_states, total_intensity


def label_count_charge_states(chromatogram, **kwargs):
    return "%s-%d-charges" % (
        default_label_extractor(chromatogram), count_charge_states(chromatogram.peaks))


def default_label_extractor(chromatogram, **kwargs):
    if chromatogram.composition:
        return str(chromatogram.composition)
    else:
        return str(chromatogram.neutral_mass)


class ColorCycler(object):
    def __init__(self, colors=None):
        if colors is None:
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey', 'black', "orange"]
        self.color_cycler = cycle(colors)

    def __call__(self, *args, **kwargs):
        return next(self.color_cycler)


class NGlycanChromatogramColorizer(object):
    def __call__(self, chromatogram, default_color='black'):
        if chromatogram.composition is None:
            return default_color
        else:
            try:
                return colors.NGlycanCompositionColorizer(chromatogram.composition)
            except:
                return default_color

n_glycan_colorizer = NGlycanChromatogramColorizer()


class LabelProducer(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, chromatogram, *args, **kwargs):
        return default_label_extractor(chromatogram)


class NGlycanLabelProducer(LabelProducer):
    def __init__(self, monosaccharides=("HexNAc", "Hex", "Fuc", "NeuAc")):
        self.monosaccharides = monosaccharides
        self.stub = glypy.GlycanComposition()
        for x in monosaccharides:
            self.stub[x] = -99

    def __call__(self, chromatogram, *args, **kwargs):
        if chromatogram.composition is not None:
            return list(colors.GlycanLabelTransformer(
                [chromatogram.composition, self.stub], colors.NGlycanCompositionOrderer))[0]
        else:
            return chromatogram.key

n_glycan_labeler = NGlycanLabelProducer()


class AbundantLabeler(LabelProducer):
    def __init__(self, labeler, threshold):
        self.labeler = labeler
        self.threshold = threshold

    def __call__(self, chromatogram, *args, **kwargs):
        if chromatogram.total_signal > self.threshold:
            return self.labeler(chromatogram, *args, **kwargs), True
        else:
            return self.labeler(chromatogram, *args, **kwargs), False


class ChromatogramArtist(object):
    def __init__(self, chromatograms, ax=None, colorizer=None):
        if colorizer is None:
            colorizer = ColorCycler()
        if ax is None:
            fig, ax = plt.subplots(1)
        self.chromatograms = chromatograms
        self.minimum_ident_time = float("inf")
        self.maximum_ident_time = 0
        self.maximum_intensity = 0
        self.scan_id_to_intensity = {}
        self.ax = ax
        self.default_colorizer = colorizer

    def draw_generic_chromatogram(self, label, rt, heights, color, fill=False):
        if fill:
            s = self.ax.fill_between(
                rt,
                heights,
                alpha=0.25,
                color=color,
                label=label
            )

        else:
            s = self.ax.plot(rt, heights, color=color, label=label, alpha=0.5)[0]

        s.set_gid(str(label) + "-area")
        s = self.ax.scatter(
            rt,
            heights,
            color=color,
            s=1)
        s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = heights.index(apex)
        rt_apex = rt[apex_ind]

        if label is not None:
            self.ax.text(rt_apex, apex + 1200, label, ha='center', fontsize=10)

    def draw_group(self, label, rt, heights, color, label_peak=True):
        s = self.ax.fill_between(
            rt,
            heights,
            alpha=0.25,
            color=color,
            label=label
        )
        s.set_gid(str(label) + "-area")
        s = self.ax.scatter(
            rt,
            heights,
            color=color,
            s=1)
        s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None and label_peak:
            self.ax.text(rt_apex, apex + 1200, label, ha='center', fontsize=10)

    def process_group(self, composition, chromatogram, label_function=default_label_extractor):
        part = slice(None)
        peaks = chromatogram.peaks[part]
        ids = chromatogram.scan_ids[part]

        color = self.default_colorizer(chromatogram)

        # rt = chromatogram.retention_times
        # heights = [total_intensity(peak) for peak in peaks]
        rt, heights = chromatogram.as_arrays()

        self.scan_id_to_intensity = dict(zip(ids, heights))

        self.maximum_ident_time = max(max(rt), self.maximum_ident_time)
        self.minimum_ident_time = min(min(rt), self.minimum_ident_time)

        self.maximum_intensity = max(max(heights), self.maximum_intensity)

        label = label_function(
            chromatogram, rt=rt, heights=heights, peaks=peaks)
        if isinstance(label, basestring):
            label = label
            label_peak = True
        else:
            label, label_peak = label

        self.draw_group(label, rt, heights, color, label_peak)

    def layout_axes(self, legend=True):
        self.ax.set_xlim(self.minimum_ident_time - 0.02,
                         self.maximum_ident_time + 0.02)
        self.ax.set_ylim(0, self.maximum_intensity * 1.1)
        if legend:
            self.ax.legend(bbox_to_anchor=(1.7, 1.), ncol=2, fontsize=10)
        self.ax.axes.spines['right'].set_visible(False)
        self.ax.axes.spines['top'].set_visible(False)
        self.ax.yaxis.tick_left()
        self.ax.xaxis.tick_bottom()
        self.ax.set_xlabel("Retention Time", fontsize=28)
        self.ax.set_ylabel("Relative Abundance", fontsize=28)
        [t.set(fontsize=20) for t in self.ax.get_xticklabels()]
        [t.set(fontsize=20) for t in self.ax.get_yticklabels()]

    def draw(self, filter_function=lambda x, y: False, label_function=default_label_extractor,
             legend=True):
        for chroma in self.chromatograms:
            composition = chroma.composition
            if composition is not None:
                gc = glypy.GlycanComposition.parse(composition)
            else:
                gc = None
            if filter_function(gc, chroma):
                continue

            self.process_group(composition, chroma, label_function)
        self.layout_axes(legend=legend)
        return self


class SmoothingChromatogramArtist(ChromatogramArtist):
    def draw_group(self, label, rt, heights, color, label_peak=True):
        heights = gaussian_filter1d(heights, 1.0)
        s = self.ax.fill_between(
            rt,
            heights,
            alpha=0.25,
            color=color,
            label=label
        )
        s.set_gid(str(label) + "-area")
        s = self.ax.scatter(
            rt,
            heights,
            color=color,
            s=1)
        s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None and label_peak:
            self.ax.text(rt_apex, apex + 1200, label, ha='center', fontsize=10)

    def draw_generic_chromatogram(self, label, rt, heights, color, fill=False):
        heights = gaussian_filter1d(heights, 1.0)
        if fill:
            s = self.ax.fill_between(
                rt,
                heights,
                alpha=0.25,
                color=color,
                label=label
            )

        else:
            s = self.ax.plot(rt, heights, color=color, label=label, alpha=0.5)[0]

        s.set_gid(str(label) + "-area")
        s = self.ax.scatter(
            rt,
            heights,
            color=color,
            s=1)
        s.set_gid(str(label) + "-points")
        apex = max(heights)
        apex_ind = np.argmax(heights)
        rt_apex = rt[apex_ind]

        if label is not None:
            self.ax.text(rt_apex, apex + 1200, label, ha='center', fontsize=10)


class ChargeSeparatingChromatogramArtist(ChromatogramArtist):

    def process_group(self, composition, chroma):
        charge_map = defaultdict(OrderedDict)
        for scan, peaks in zip(chroma.scan_ids, chroma.peaks):
            for peak in peaks:
                charge_map[peak.charge][scan] = peak

        for charge, obs in charge_map.items():

            part = slice(None)
            peaks = obs.values()[part]
            ids = obs.keys()[part]

            color = next(self.color_generator)

            rt = [self.tracker.scan_generator.time_cache[id] for id in ids]
            heights = [(peak).intensity for peak in peaks]
            self.scan_id_to_intensity = dict(zip(ids, heights))

            self.maximum_ident_time = max(max(rt), self.maximum_ident_time)
            self.minimum_ident_time = min(min(rt), self.minimum_ident_time)
            self.maximum_intensity = max(max(heights), self.maximum_intensity)

            self.draw_group(
                ''.join(map(str, (composition, charge))), rt, heights, color)


class AggregatedAbundanceArtist(object):
    def __init__(self, chromatograms, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        self.ax = ax
        self.chromatograms = [c for c in chromatograms if c.composition is not None]

    def draw(self):
        chroma = sorted(
            self.chromatograms, lambda x, y: colors.NGlycanCompositionOrderer(x.composition, y.composition))
        indices = np.arange(len(chroma))
        keys = [c.composition for c in chroma]
        include_classes = set(map(colors.NGlycanCompositionColorizer.classify, keys))
        bar_width = 0.5
        alpha = 0.5
        xtick_labeler = colors.GlycanLabelTransformer(keys, colors.NGlycanCompositionOrderer)

        color = map(colors.NGlycanCompositionColorizer, keys)
        self.bars = self.ax.bar(
            indices + bar_width, [c.total_signal for c in chroma],
            width=bar_width, color=color, alpha=alpha, lw=0)

        ax = self.ax
        ax.set_xticks(indices + bar_width * 1.5)
        font_size = max((200. / (len(indices) / 2.)), 3)

        ax.set_xlabel(xtick_labeler.label_key, fontsize=28)
        ax.set_xticklabels(tuple(xtick_labeler), rotation=90, ha='center', size=font_size)
        ax.xaxis.set_ticks_position('none')
        ax.set_title("Glycan Composition Total Abundances", fontsize=28)
        ax.set_ylabel("Relative Intensity", fontsize=28)
        handles = colors.NGlycanCompositionColorizer.make_legend(
            include_classes, alpha=alpha)
        if handles:
            ax.legend(handles=handles, bbox_to_anchor=(1.20, 1.0))

        self.ax.axes.spines['right'].set_visible(False)
        self.ax.axes.spines['top'].set_visible(False)
        self.ax.yaxis.tick_left()
        self.ax.xaxis.set_ticks_position('none')
        return self


class LCMSSurfaceArtist(object):
    def __init__(self, chromatograms):
        self.chromatograms = chromatograms
        self.times = []
        self.masses = []
        self.heights = []

    def build_map(self):
        self.times = []
        self.masses = []
        self.heights = []

        for chroma in self.chromatograms:
            x, z = chroma.as_arrays()
            y = chroma.neutral_mass
            self.times.append(x)
            self.masses.append(y)

        rt = set()
        map(rt.update, self.times)
        rt = np.array(list(rt))
        rt.sort()
        self.times = rt

        self.heights = list(map(self.make_z_array, self.chromatograms))

    def make_z_array(self, chroma):
        z = []
        next_time_i = 0
        next_time = chroma.retention_times[next_time_i]

        for i in self.times:
            if np.allclose(i, next_time):
                z.append(total_intensity(chroma.peaks[next_time_i]))
                next_time_i += 1
                if next_time_i == len(chroma):
                    break
                next_time = chroma.retention_times[next_time_i]
            else:
                z.append(0)
        z = gaussian_filter1d(np.concatenate((z, np.zeros(len(self.times) - len(z)))), 1)
        return z

    def draw(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.ax = ax

        self.build_map()

        X, Y = np.meshgrid(self.times, self.masses)
        ax.plot_surface(X, Y, self.heights, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=True,
                        alpha=1)
        ax.view_init()
        ax.azim += 20
        ax.set_xlim(self.times.min(), self.times.max())
        ax.set_ylim(min(self.masses) - 100, max(self.masses))
        ax.set_xlabel("Retention Time", fontsize=18)
        ax.set_ylabel("Neutral Mass", fontsize=18)
        ax.set_zlabel("Relative Abundance", fontsize=18)
        return self
