from collections import defaultdict

import glycopeptidepy
from glycopeptidepy.structure import sequence, modification
from glycopeptidepy.utils import simple_repr
from glypy.plot import plot as draw_tree

from .colors import darken, cnames, hex2color

from matplotlib import pyplot as plt, patches as mpatches, textpath, font_manager


font_options = font_manager.FontProperties(family='monospace')


class SequencePositionGlyph(object):

    def __init__(self, position, index, x, y, patch=None, ax=None, **kwargs):
        self.position = position
        self.index = index
        self.x = x
        self.y = y
        self._patch = patch
        self._ax = ax
        self.options = kwargs

    __repr__ = simple_repr

    def render(self, ax=None):
        if ax is None:
            ax = self._ax
        else:
            self._ax = ax

        symbol = self.position[0].symbol
        tpath = textpath.TextPath(
            (self.x, self.y), symbol, size=self.options.get('size'), prop=font_options)
        tpatch = mpatches.PathPatch(
            tpath, color=self.options.get(
                'color', 'black'), lw=0.25)
        self._patch = tpatch
        ax.add_patch(tpatch)
        return ax


class SequenceGlyph(object):

    def __init__(self, peptide, ax=None, size=1, step_coefficient=1.0, **kwargs):
        if not isinstance(peptide, sequence.PeptideSequenceBase):
            peptide = sequence.PeptideSequence(peptide)
        self.sequence = peptide
        self.ax = ax
        self.size = size
        self.step_coefficient = 1.0
        self.sequence_position_glyphs = []
        self.x = kwargs.get('x', 1)
        self.y = kwargs.get('y', 1)
        self.options = kwargs

        self.render()

    def render(self):
        ax = self.ax
        if ax is None:
            fig, ax = plt.subplots(1)
            self.ax = ax
        else:
            ax = self.ax

        size = self.size
        x = self.x
        y = self.y
        glyphs = self.sequence_position_glyphs = []
        i = 0
        for position in self.sequence:
            glyph = SequencePositionGlyph(position, i, x, y, size=size)
            glyph.render(ax)
            glyphs.append(glyph)
            x += size * self.step_coefficient
            i += 1
        return ax

    def next_between(self, index):
        for i, position in enumerate(self.sequence_position_glyphs, 1):
            if i == index:
                break
        return position.x + (self.step_coefficient * self.size) / 1.3

    def draw_bar_at(self, index, height=0.25, color='red', **kwargs):
        x = self.next_between(index)
        y = self.y
        rect = mpatches.Rectangle(
            (x, y - height), 0.05, 1 + height, color=color, **kwargs)
        self.ax.add_patch(rect)

    def draw_n_term_label(
            self, index, label, height=0.25, length=0.75, size=0.45,
            color='black', **kwargs):
        x = self.next_between(index)
        y = self.y - height - size
        length *= self.step_coefficient
        label_x = x - length * 0.9
        label_y = y
        tpath = textpath.TextPath(
            (label_x, label_y), label, size=size, prop=font_options)
        tpatch = mpatches.PathPatch(
            tpath, color='black', lw=0.25)
        self.ax.add_patch(tpatch)

    def draw_n_term_annotation(
            self, index, height=0.25, length=0.75, color='red', **kwargs):
        x = self.next_between(index)
        y = self.y - height
        length *= self.step_coefficient
        rect = mpatches.Rectangle(
            (x - length, y), length, 0.05, color=color, **kwargs)
        self.ax.add_patch(rect)

    def draw_c_term_label(
            self, index, label, height=0.25, length=0.75, size=0.45,
            color='black', **kwargs):
        x = self.next_between(index)
        y = (self.y * 2) + height
        length *= self.step_coefficient
        label_x = x + length / 10.
        label_y = y + 0.2
        tpath = textpath.TextPath(
            (label_x, label_y), label, size=size, prop=font_options)
        tpatch = mpatches.PathPatch(
            tpath, color='black', lw=0.25)
        self.ax.add_patch(tpatch)

    def draw_c_term_annotation(
            self, index, height=0., length=0.75, color='red', **kwargs):
        x = self.next_between(index)
        y = (self.y * 2) + height
        length *= self.step_coefficient
        rect = mpatches.Rectangle((x, y), length, 0.05, color=color, **kwargs)
        self.ax.add_patch(rect)

    def layout(self):
        ax = self.ax
        ax.set_xlim(self.x -
                    1, self.size *
                    self.step_coefficient *
                    len(self.sequence) +
                    1)
        ax.set_ylim(self.y - 1, self.y + 2)
        ax.axis("off")

    def annotate_from_fragments(self, fragments, **kwargs):
        index = {}
        for i in range(1, len(self.sequence)):
            for series in self.sequence.break_at(i):
                for f in series:
                    index[f.name] = i
        n_annotations = []
        c_annotations = []

        for fragment in fragments:
            key = fragment.name
            if key in index:
                is_glycosylated = fragment.is_glycosylated
                if key.startswith('b') or key.startswith('c'):
                    n_annotations.append((index[key], is_glycosylated, key))
                elif key.startswith('y') or key.startswith('z'):
                    c_annotations.append((index[key], is_glycosylated, key))

        kwargs_with_greater_height = kwargs.copy()
        kwargs.setdefault("height", 0.25)
        kwargs_with_greater_height["height"] = kwargs["height"] * 2
        kwargs.setdefault('color', 'red')

        try:
            kwargs.pop("glycosylated_color")
            kwargs_with_greater_height[
                'color'] = kwargs_with_greater_height['glycosylated_color']
            kwargs_with_greater_height.pop("glycosylated_color")
        except KeyError:
            color = kwargs.get("color", 'red')
            try:
                color = cnames.get(color, color)
                rgb = hex2color(color)
            except:
                rgb = color
            kwargs_with_greater_height['color'] = darken(rgb)

        heights_at = defaultdict(float)

        for n_annot, is_glycosylated, key in n_annotations:
            self.draw_bar_at(n_annot, color=kwargs['color'])
            if is_glycosylated:
                self.draw_n_term_annotation(
                    n_annot, **kwargs_with_greater_height)
                if heights_at[n_annot] < kwargs_with_greater_height['height']:
                    heights_at[n_annot] = kwargs_with_greater_height['height']
            else:
                self.draw_n_term_annotation(n_annot, label=key, **kwargs)
                if heights_at[n_annot] < kwargs['height']:
                    heights_at[n_annot] = kwargs['height']

        labeled = set()
        for n_annot, is_glycosylated, key in n_annotations:
            label = key.split("+")[0]
            if label not in labeled:
                self.draw_n_term_label(n_annot, label=label, height=heights_at[n_annot])
                labeled.add(label)

        kwargs_with_greater_height['height'] = kwargs.get("height", 0.25)
        kwargs['height'] = 0

        heights_at = defaultdict(float)

        for c_annot, is_glycosylated, key in c_annotations:
            self.draw_bar_at(c_annot, color=kwargs['color'])
            if is_glycosylated:
                self.draw_c_term_annotation(
                    c_annot, **kwargs_with_greater_height)
                if heights_at[c_annot] < kwargs_with_greater_height['height']:
                    heights_at[c_annot] = kwargs_with_greater_height['height']
            else:
                self.draw_c_term_annotation(c_annot, label=key, **kwargs)
                if heights_at[c_annot] < kwargs['height']:
                    heights_at[c_annot] = kwargs['height']

        labeled = set()
        for c_annot, is_glycosylated, key in c_annotations:
            label = key.split("+")[0]
            if label not in labeled:
                self.draw_c_term_label(c_annot, label=label, height=heights_at[c_annot])
                labeled.add(label)

    @classmethod
    def from_spectrum_match(cls, spectrum_match, ax=None, **kwargs):
        annotation_options = kwargs.pop("annotation_options", {})
        inst = cls(spectrum_match.target, ax=ax, **kwargs)
        fragments = [f for f in spectrum_match.solution_map.fragments()]
        inst.annotate_from_fragments(fragments, **annotation_options)
        inst.layout()
        return inst


def glycopeptide_match_logo(glycopeptide_match, ax=None, color='red', glycosylated_color='forestgreen', **kwargs):
    annotation_options = kwargs.get("annotation_options", {})
    annotation_options['color'] = color
    annotation_options['glycosylated_color'] = glycosylated_color
    kwargs['annotation_options'] = annotation_options
    inst = SequenceGlyph.from_spectrum_match(
        glycopeptide_match, color=color,
        glycosylated_color=glycosylated_color, ax=ax, **kwargs)
    return inst.ax


def draw_proteoform(proteoform, width=60):
    '''Draw a sequence logo for a proteoform
    '''
    fig, ax = plt.subplots(1)
    i = 0
    n = len(proteoform)
    y = 1
    x = 1
    rows = []
    # partion the protein sequence into chunks of width positions
    while i < n:
        rows.append(glycopeptidepy.PeptideSequence.from_iterable(proteoform[i:i + width]))
        i += width
    # draw each row, starting from the last row and going up
    # so that if the current row has a glycan on it, we can
    # compute its height and start drawing the next row further
    # up the plot
    for row in rows[::-1]:
        sg = SequenceGlyph(row, x=x, y=y, ax=ax)
        sg.layout()
        drawn_trees = []
        for position in sg.sequence_position_glyphs:
            # check for glycosylation at each position
            if position.position[1] and position.position[1][0].is_a(
                    modification.ModificationCategory.glycosylation):
                mod = position.position[1][0]
                dtree, ax = draw_tree(
                    mod.rule._original, at=(position.x, position.y + 1),
                    orientation='vertical', ax=sg.ax, center=False)
                # re-center tree because draw_tree's layout only sets the starting
                # coordinates of the tree root, and later transforms move it around.
                # this sequence of translates will re-center it above the residue
                dtree.transform.translate(-dtree.x, 0).translate(0.3, 0)
                drawn_trees.append(dtree)
        # get the height of the tallest glycan
        if drawn_trees:
            tree_height = max(dtree.extrema()[3] for dtree in drawn_trees) + 0.5
        else:
            tree_height = 0
        ymax = tree_height + 1
        y += ymax
    ax.set_xlim(0, width + 10)
    ax.set_ylim(-1, y + 1)
    return ax
