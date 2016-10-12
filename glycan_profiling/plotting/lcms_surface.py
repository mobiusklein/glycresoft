from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
import numpy as np


from ..scoring import total_intensity


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
        scaler = max(map(max, self.heights)) / 100.
        for height in self.heights:
            height /= scaler

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

    def make_sparse(self, width=0.05):
        i = 0
        masses = []
        heights = []

        flat = self.heights[0] * 0

        masses.append(self.masses[0] - 200)
        heights.append(flat)

        while i < len(self.masses):
            mass = self.masses[i]
            masses.append(mass - width)
            heights.append(flat)
            masses.append(mass)
            heights.append(self.heights[i])
            masses.append(mass + width)
            heights.append(flat)
            i += 1

        self.masses = masses
        self.heights = heights

    def draw(self, alpha=0.8, **kwargs):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.ax = ax

        self.build_map()
        self.make_sparse()

        X, Y = np.meshgrid(self.times, self.masses)
        ax.plot_surface(X, Y, self.heights, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=True,
                        alpha=alpha)
        ax.view_init()
        ax.azim += 20
        ax.set_xlim3d(self.times.min(), self.times.max())
        ax.set_ylim3d(min(self.masses) - 100, max(self.masses))
        ax.set_xlabel("Retention Time (Min)", fontsize=18)
        ax.set_ylabel("Neutral Mass", fontsize=18)
        ax.set_zlabel("Relative Abundance (%)", fontsize=18)
        return self
