from matplotlib import pyplot as plt
import numpy as np


def target_decoy_score_separation(target_hits, decoy_hits, binwidth=5, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    bins = np.arange(min([s.score for s in target_hits]), max([s.score for s in target_hits]) + binwidth, binwidth)
    ax.hist([s.score for s in target_hits], bins=bins, alpha=0.4, lw=0.5, ec='white', label='Target PSMs')
    ax.hist([s.score for s in decoy_hits], bins=bins, alpha=0.4, lw=0.5, ec='white', label='Decoy PSMs')
    ax.legend().get_frame().set_linewidth(0)
    return ax
