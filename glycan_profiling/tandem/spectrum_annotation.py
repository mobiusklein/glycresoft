from ms_peak_picker.utils import draw_peaklist


def annotate_matched_deconvoluted_peaks(solution_map, ax, labeler=None, **kwargs):
    if labeler is None:
        def labeler(fragment, peak, ax, fontsize, y=None):
            if y is None:
                y = peak.intensity
            label = "%s" % fragment.name
            if peak.charge > 1:
                label += "$^{%d}$" % peak.charge
            rotation = 90
            ax.text(peak.mz, y, label, rotation=rotation, va='bottom', ha='center', fontsize=fontsize)
            return label

    fontsize = kwargs.get('fontsize', 9)
    upper = max(ax.get_ylim())
    for fragment, peak in solution_map:
        draw_peaklist([peak], alpha=0.8, ax=ax, color='red')
        y = peak.intensity
        y = min(y + 100, upper * 0.95)
        labeler(fragment, peak, ax, fontsize, y)
    return ax
