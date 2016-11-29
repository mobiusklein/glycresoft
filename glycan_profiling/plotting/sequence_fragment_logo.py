from glycopeptidepy.structure import sequence
from glycopeptidepy.utils import simple_repr

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
                'color', 'black'))
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

    def draw_n_term_annotation(
            self, index, height=0.25, length=0.5, color='red', **kwargs):
        x = self.next_between(index)
        y = self.y - height
        length *= self.step_coefficient
        rect = mpatches.Rectangle(
            (x - length, y), length, 0.05, color=color, **kwargs)
        self.ax.add_patch(rect)

    def draw_c_term_annotation(
            self, index, height=0., length=0.5, color='red', **kwargs):
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
                    n_annotations.append((index[key], is_glycosylated))
                elif key.startswith('y') or key.startswith('z'):
                    c_annotations.append((index[key], is_glycosylated))
        kwargs_with_greater_height = kwargs.copy()
        kwargs_with_greater_height["height"] = kwargs.get("height", 0.25) * 2
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
        for n_annot, is_glycosylated in n_annotations:
            self.draw_bar_at(n_annot, color=kwargs['color'])
            if is_glycosylated:
                self.draw_n_term_annotation(
                    n_annot, **kwargs_with_greater_height)
            else:
                self.draw_n_term_annotation(n_annot, **kwargs)

        kwargs_with_greater_height['height'] = kwargs.get("height", 0.25)
        kwargs['height'] = 0
        for c_annot, is_glycosylated in c_annotations:
            self.draw_bar_at(c_annot, color=kwargs['color'])
            if is_glycosylated:
                self.draw_c_term_annotation(
                    c_annot, **kwargs_with_greater_height)
            else:
                self.draw_c_term_annotation(c_annot, **kwargs)

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
