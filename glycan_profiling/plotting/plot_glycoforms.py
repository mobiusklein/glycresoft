import matplotlib
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.textpath import TextPath

from .svg_utils import ET, BytesIO, IDMapper
from .colors import lighten, darken, get_color


font_options = font_manager.FontProperties(family='monospace')


def span_overlap(a, b):
    if a.end_position == b.start_position or b.end_position == a.start_position:
        return False
    return (a.spans(b.start_position + 1) or a.spans(b.end_position) or
            b.spans(a.start_position + 1) or b.spans(a.end_position))


def layout_layers(gpms):
    '''
    Produce a non-overlapping stacked layout of individual peptide-like
    identifications across a protein sequence.
    '''
    layers = [[]]
    gpms.sort(key=lambda x: x.ms2_score, reverse=True)
    for gpm in gpms:
        placed = False
        for layer in layers:
            collision = False
            for member in layer:
                if span_overlap(gpm, member):
                    collision = True
                    break
            if not collision:
                layer.append(gpm)
                placed = True
                break
        if not placed:
            layers.append([gpm])
    # import IPython
    # IPython.embed()
    return layers


def draw_layers(layers, protein, scale_factor=1.0, ax=None, row_width=50, **kwargs):
    '''
    Render fixed-width stacked peptide identifications across
    a protein. Each shape is rendered with a unique identifier.
    '''
    if ax is None:
        figure, ax = plt.subplots(1, 1)
    id_mapper = IDMapper()
    i = 0

    layer_height = 0.56 * scale_factor
    y_step = (layer_height + 0.15) * -scale_factor
    cur_y = -3

    cur_position = 0

    mod_text_x_offset = 0.50 * scale_factor
    sequence_font_size = 6. * scale_factor
    mod_font_size = 2.08 * scale_factor
    mod_text_y_offset = 0.1 * scale_factor
    mod_width = 0.5 * scale_factor
    mod_x_offset = 0.60 * scale_factor
    total_length = len(protein.protein_sequence or '')
    protein_pad = -0.365 * scale_factor
    peptide_pad = protein_pad * (1.2)
    peptide_end_pad = 0.35 * scale_factor

    glycosites = set(protein.n_glycan_sequon_sites)
    for layer in layers:
        layer.sort(key=lambda x: x.start_position)

    while cur_position < total_length:
        next_row = cur_position + row_width
        i = -2
        text_path = TextPath(
            (protein_pad + i, layer_height + .2 + cur_y),
            str(cur_position + 1), size=sequence_font_size / 7.5, prop=font_options, stretch=1000)
        patch = mpatches.PathPatch(text_path, facecolor='grey', lw=0.04)
        ax.add_patch(patch)

        i = row_width + 2
        text_path = TextPath(
            (protein_pad + i, layer_height + .2 + cur_y),
            str(next_row), size=sequence_font_size / 7.5, prop=font_options, stretch=1000)
        patch = mpatches.PathPatch(text_path, facecolor='grey', lw=0.04)
        ax.add_patch(patch)

        for i, aa in enumerate(protein.protein_sequence[cur_position:next_row]):
            text_path = TextPath(
                (protein_pad + i, layer_height + .2 + cur_y),
                aa, size=sequence_font_size / 7.5, prop=font_options, stretch=1000)
            color = 'red' if any(
                (((i + cur_position) in glycosites),
                 ((i + cur_position - 1) in glycosites),
                 ((i + cur_position - 2) in glycosites))
            ) else 'black'
            patch = mpatches.PathPatch(text_path, facecolor=color, lw=0.04)
            ax.add_patch(patch)

        for layer in layers:
            c = 0
            for gpm in layer:
                if gpm.start_position < cur_position and gpm.end_position < cur_position:
                    continue
                elif gpm.start_position >= next_row:
                    break
                c += 1

                color = "lightblue"
                alpha = min(max(gpm.ms2_score * 2, 0.2), 0.8)

                interval_start = max(
                    gpm.start_position - cur_position,
                    0)
                interval_end = min(
                    len(gpm.structure) + gpm.start_position - cur_position,
                    row_width)

                rect = mpatches.Rectangle(
                    (interval_start + peptide_pad, cur_y),
                    width=(interval_end - interval_start) - peptide_end_pad,
                    height=layer_height,
                    facecolor=color, edgecolor='none',
                    alpha=alpha)

                id_mapper.add("glycopeptide-%d", rect, {
                    "sequence": str(gpm.structure),
                    "start-position": gpm.start_position,
                    "end-position": gpm.end_position,
                    "ms2-score": gpm.ms2_score,
                    "q-value": gpm.q_value,
                    "record-id": gpm.id if hasattr(gpm, 'id') else None,
                    "calculated-mass": gpm.structure.total_mass,
                    "spectra-count": len(gpm.spectrum_matches)
                })
                ax.add_patch(rect)

                # Compute offsets into the peptide sequence to select
                # PTMs to draw for this row
                if (cur_position) > gpm.start_position:
                    start_index = cur_position - gpm.start_position
                    if gpm.end_position - start_index > row_width:
                        end_index = min(
                            row_width,
                            len(gpm.structure))
                    else:
                        end_index = gpm.end_position - start_index
                else:
                    start_index = min(0, gpm.start_position - cur_position)
                    end_index = min(
                        gpm.end_position - cur_position,
                        row_width - (gpm.start_position - cur_position))

                # Extract PTMs from the peptide sequence to draw over the
                # peptide rectangle
                seq = gpm.structure

                for i, pos in enumerate(seq[start_index:end_index]):
                    if len(pos[1]) > 0:
                        color = get_color(pos[1][0].name)
                        facecolor, edgecolor = lighten(
                            color), darken(color, 0.6)

                        mod_patch = mpatches.Rectangle(
                            (gpm.start_position - cur_position +
                             i - mod_x_offset + 0.3 + start_index, cur_y),
                            width=mod_width, height=layer_height, alpha=0.4,
                            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5,
                        )

                        id_mapper.add(
                            "modification-%d", mod_patch,
                            {
                                "modification-type": pos[1][0].name,
                                "parent": gpm.id
                            })
                        ax.add_patch(mod_patch)
                        text_path = TextPath(
                            (gpm.start_position - cur_position + i -
                             mod_text_x_offset + 0.3 + start_index, cur_y + mod_text_y_offset),
                            str(pos[1][0])[0], size=mod_font_size / 4.5, prop=font_options)
                        patch = mpatches.PathPatch(
                            text_path, facecolor='black', lw=0.04)
                        ax.add_patch(patch)
            if c > 0:
                cur_y += y_step
        cur_y += y_step * 3
        cur_position = next_row

    ax.set_ylim(cur_y - 5, 5)
    ax.set_xlim(-5, row_width + 5)
    ax.axis('off')
    return ax, id_mapper


def plot_glycoforms(protein, identifications, **kwargs):
    layers = layout_layers(identifications)
    ax, id_mapper = draw_layers(layers, protein, **kwargs)
    return ax, id_mapper


def plot_glycoforms_svg(protein, identifications, scale=1.5, ax=None,
                        margin_left=80, margin_top=0, height_padding_scale=1.2,
                        **kwargs):
    '''
    A specialization of :func:`plot_glycoforms` which adds additional features to SVG images, such as
    adding shape metadata to XML tags and properly configuring the viewport and canvas for the figure's
    dimensions.
    '''
    ax, id_mapper = plot_glycoforms(protein, identifications, ax=ax, **kwargs)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.autoscale()

    x_size = sum(map(abs, xlim))
    y_size = sum(map(abs, ylim))

    aspect_ratio = x_size / y_size
    canvas_x = 8.
    canvas_y = canvas_x / aspect_ratio

    fig = ax.get_figure()
    # fig.tight_layout(pad=0.2)
    fig.tight_layout(pad=0)
    fig.patch.set_visible(False)
    fig.set_figwidth(canvas_x)
    fig.set_figheight(canvas_y)

    ax.patch.set_visible(False)
    buff = BytesIO()
    fig.savefig(buff, format='svg')
    root, ids = ET.XMLID(buff.getvalue())
    root.attrib['class'] = 'plot-glycoforms-svg'
    for id, attributes in id_mapper.items():
        element = ids[id]
        element.attrib.update({("data-" + k): str(v)
                               for k, v in attributes.items()})
        element.attrib['class'] = id.rsplit('-')[0]
    min_x, min_y, max_x, max_y = map(int, root.attrib["viewBox"].split(" "))
    min_x += margin_left
    min_y += margin_top
    max_x += 200
    view_box = ' '.join(map(str, (min_x, min_y, max_x, max_y)))
    root.attrib["viewBox"] = view_box
    width = float(root.attrib["width"][:-2]) * 1.75
    root.attrib["width"] = "100%"

    height = width / (aspect_ratio)

    root.attrib["height"] = "%dpt" % (height * height_padding_scale)
    root.attrib["preserveAspectRatio"] = "xMinYMin meet"
    root[1].attrib["transform"] = "scale(%f)" % scale
    svg = ET.tostring(root)
    plt.close()

    return svg
