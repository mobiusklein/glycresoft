from ms_peak_picker.utils import draw_peaklist


def annotate_matched_deconvoluted_peaks(solution_map, ax):
    upper = max(ax.get_ylim())
    for fragment, peak in solution_map:
        draw_peaklist([peak], alpha=0.8, ax=ax, color='red')
        y = peak.intensity
        y = min(y + 100, upper * 0.8)
        label = "%s" % fragment.name
        if peak.charge > 1:
            label += "$^{%d}$" % peak.charge
        ax.text(peak.mz, y, label, rotation=90, va='bottom', ha='center', fontsize=12)
    return ax
