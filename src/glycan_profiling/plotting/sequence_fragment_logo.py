from collections import defaultdict
from typing import Any, Dict, List, Tuple, NamedTuple, Optional

import glypy

import glycopeptidepy
from glycopeptidepy.structure import sequence, modification, SequencePosition
from glycopeptidepy.structure.fragment import IonSeries
from glycopeptidepy.utils import simple_repr

from glypy.plot import plot as draw_tree, SNFGNomenclature
from glypy.plot.common import MonosaccharidePatch

from .colors import darken, cnames, hex2color, get_color

import numpy as np

from matplotlib import (
    pyplot as plt,
    patches as mpatches,
    textpath,
    font_manager,
    path as mpath,
    transforms as mtransform,
    collections as mcollection
)


font_options = font_manager.FontProperties(family='monospace')
glycan_symbol_grammar = SNFGNomenclature()


class BBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def expand(self, xmin, ymin, xmax, ymax) -> 'BBox':
        return self.__class__(
            min(xmin, self.xmin),
            min(ymin, self.ymin),
            max(xmax, self.xmax),
            max(ymax, self.ymax),
        )


def bbox_path(path):
    nodes = path.vertices
    if nodes.size == 0:
        return BBox(0, 0, 0, 0)
    xmin = nodes[:, 0].min()
    xmax = nodes[:, 0].max()
    ymin = nodes[:, 1].min()
    ymax = nodes[:, 1].max()
    return BBox(xmin, ymin, xmax, ymax)


class SequencePositionGlyph(object):
    position: SequencePosition
    index: int
    _ax: plt.Axes
    x: float
    y: float
    options: Dict[str, Any]

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

        symbol = self.position.amino_acid.symbol

        color = self.options.get("color", 'black')
        if self.position.modifications:
            color = darken(get_color(self.position.modifications[0]))
            label = self.position.modifications[0].name
        else:
            label = None

        tpath = textpath.TextPath(
            (self.x, self.y), symbol, size=self.options.get('size'), prop=font_options)
        tpatch = mpatches.PathPatch(
            tpath, color=color, lw=self.options.get('lw', 0), label=label)
        self._patch = tpatch
        ax.add_patch(tpatch)
        return ax

    def set_transform(self, transform):
        self._patch.set_transform(transform)

    def centroid(self):
        path = self._patch.get_path()
        point = np.zeros_like(path.vertices[0])
        c = 0.
        for p in path.vertices:
            point += p
            c += 1
        return point / c

    def bbox(self):
        return bbox_path(self._patch.get_path())


class GlycanCompositionGlyphs(object):
    glycan_composition: glypy.GlycanComposition
    ax: plt.Axes
    x: float
    xend: float
    y: float
    options: Dict[str, Any]

    patches: List[MonosaccharidePatch]

    def __init__(self, glycan_composition, x, y, ax, **kwargs):
        self.glycan_composition = glycan_composition
        self.ax = ax
        self.x = x
        self.xend = x
        self.y = y
        self.options = kwargs
        self.patches = []

    def render(self):
        x = self.x
        y = self.y
        glyphs = []
        for mono, count in self.glycan_composition.items():
            glyph = glycan_symbol_grammar.draw(mono, x, y, ax=self.ax, scale=(0.4, 0.4))
            x += 0.75
            glyphs.extend(glyph.shape_patches)
            glyph = glycan_symbol_grammar.draw_text(
                self.ax, x, y - 0.17, r'$\times %d$' % count, center=False, fontsize=22)
            if isinstance(glyph, tuple):
                for g in glyph:
                    g.set_lw(0)
            else:
                glyph.set_lw(0)
            x += 1.75
            glyphs.extend(glyph)

        self.xend = x
        self.patches = glyphs

    def set_transform(self, transform):
        for patch in self.patches:
            patch.set_transform(transform)

    def bbox(self):
        if not self.patches:
            return self.x, self.y, self.xend, self.y

        bbox = bbox_path(self.patches[0].get_path())
        for patch in self.patches:
            bbox = bbox.expand(*bbox_path(patch.get_path()))
        return bbox


class FragmentStroke:
    index: int

    linewidth: float
    x: float
    top: float
    bottom: float

    n_length: float
    c_length: float

    v_step: float
    stroke_slope: float

    c_labels: List[str]
    n_labels: List[str]

    main_color: str
    c_colors: List[str]
    n_colors: List[str]

    main_stroke: mpatches.PathPatch
    c_strokes: List[mpatches.PathPatch]
    n_strokes: List[mpatches.PathPatch]

    use_collection: bool = True
    _collection: mcollection.PatchCollection
    ax: plt.Axes
    options: Dict[str, Any]

    def __init__(
        self,
        index: int,
        x: float,
        top: float,
        bottom: float,
        n_labels=None,
        c_labels=None,
        main_color=None,
        n_colors=None,
        c_colors=None,
        ax=None,
        v_step: float=0.2,
        stroke_slope: float=0.2,
        n_length: float=0.8,
        c_length: float=0.8,
        options: Dict[str, Any] = None,
        **kwargs
    ):
        if options is None:
            options = {}
        options.update(kwargs)

        self.index = index
        self.x = x
        self.top = top
        self.bottom = bottom

        self.v_step = v_step
        self.stroke_slope = stroke_slope

        self.c_length = c_length
        self.n_length = n_length

        self.c_labels = c_labels or []
        self.n_labels = n_labels or []
        self.main_color = main_color or "black"
        self.c_colors = c_colors or ["black"] * len(self.c_labels)
        self.n_colors = n_colors or ["black"] * len(self.n_labels)

        self.main_stroke = None
        self.c_strokes = []
        self.n_strokes = []

        self.ax = ax
        self.options = options
        self.options.setdefault('lw', 2)
        self._collection = None

    def has_annotations(self):
        return self.has_n_annotations() or self.has_c_annotations()

    def has_n_annotations(self):
        return bool(self.n_labels)

    def has_c_annotations(self):
        return bool(self.c_labels)

    def add_n_label(self, label: str, color='black'):
        self.n_labels.append(label)
        self.n_colors.append(color)

    def add_c_label(self, label: str, color='black'):
        self.c_labels.append(label)
        self.c_colors.append(color)

    def draw_main_stroke(self):
        p = mpath.Path(
            [[self.x, self.top], [self.x, self.bottom], [self.x, self.bottom]],
            [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.STOP],
        )

        pp = mpatches.PathPatch(
            p,
            facecolor=self.main_color,
            edgecolor=self.main_color,
            lw=self.options['lw'],
            capstyle="round" if self.stroke_slope != 0 else 'projecting',
        )
        if not self.use_collection:
            self.ax.add_patch(pp)
        self.main_stroke = pp

    def draw_c_strokes(self):
        v_step = self.v_step
        stroke_slope = self.stroke_slope
        for i, (c_label, color) in enumerate(zip(self.c_labels, self.c_colors), 1):
            if color is None or color == 'none':
                continue
            p = mpath.Path(
                [
                    [self.x, self.top + v_step * (i - 1)],
                    [self.x + self.c_length, self.top + v_step * (i - 1) + stroke_slope],
                    [self.x + self.c_length, self.top + v_step * (i - 1) + stroke_slope],
                ],
                [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.STOP],
            )
            pp = mpatches.PathPatch(
                p, facecolor=color, edgecolor=color, lw=self.options['lw'],
                capstyle="round" if self.stroke_slope != 0 else 'projecting'
            )
            if not self.use_collection:
                self.ax.add_patch(pp)
            self.c_strokes.append(pp)

    def draw_n_strokes(self):
        v_step = self.v_step
        for i, (n_label, color) in enumerate(zip(self.n_labels, self.n_colors), 1):
            if color is None or color == 'none':
                continue
            p = mpath.Path(
                [
                    [self.x, self.bottom - v_step * (i - 1)],
                    [self.x - self.n_length, self.bottom - v_step * (i - 1) - self.stroke_slope],
                    [self.x - self.n_length, self.bottom -
                        v_step * (i - 1) - self.stroke_slope],
                ],
                [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.STOP],
            )
            pp = mpatches.PathPatch(
                p, facecolor=color, edgecolor=color, lw=self.options['lw'],
                capstyle="round" if self.stroke_slope != 0 else 'projecting'
            )
            if not self.use_collection:
                self.ax.add_patch(pp)
            self.n_strokes.append(pp)

    def draw(self):
        self.draw_main_stroke()
        self.draw_c_strokes()
        self.draw_n_strokes()
        if self.c_labels:
            self.draw_c_label()
        if self.n_labels:
            self.draw_n_label()
        if self.use_collection:
            collect = self.as_patch_collection()
            self.ax.add_collection(collect)
            self._collection = collect

    def draw_c_label(self):
        if self.c_strokes:
            p = self.c_strokes[-1]
            base = bbox_path(p.get_path()).ymax
            label_y = base + 0.2
        else:
            i = len(self.c_strokes)
            label_y = self.top + (0.2) * i
        skipped = 0
        seen = set()
        for j, text in enumerate(self.c_labels):
            text = str(text)
            if text in seen or not text:
                skipped += 1
                continue
            seen.add(text)
            tpath = textpath.TextPath(
                (self.x, label_y + (j - skipped) * (self.v_step or 0.2)),
                text,
                size=0.45,
                prop=font_options,
            )
            # bbox = bbox_path(tpath)
            # dy = (bbox.ymax - bbox.ymin) / 2 + 0.2 * (j - skipped)
            dy = (0.2) * (j - skipped)
            tpath = tpath.transformed(mtransform.Affine2D().translate(0, dy))
            tpatch = mpatches.PathPatch(tpath, color="black", lw=0)
            if not self.use_collection:
                self.ax.add_patch(tpatch)
            self.c_strokes.append(tpatch)

    def draw_n_label(self):
        if self.n_strokes:
            p = self.n_strokes[-1]
            base = bbox_path(p.get_path()).ymin
            label_y = base - 0.1
        else:
            i = len(self.n_strokes)
            label_y = self.bottom - (self.v_step or 0.2) * i
        skipped = 0
        seen = set()
        for j, text in enumerate(self.n_labels):
            text = str(text)
            if text in seen or not text:
                skipped += 1
                continue
            seen.add(text)
            tpath = textpath.TextPath(
                (self.x - self.c_length, label_y -
                 (j - skipped) * (self.v_step or 0.2)),
                text,
                size=0.45,
                prop=font_options,
            )
            bbox = bbox_path(tpath)
            dy = (bbox.ymax - bbox.ymin) + 0.16 * (j - skipped)
            tpath = tpath.transformed(mtransform.Affine2D().translate(0, -dy))

            tpatch = mpatches.PathPatch(tpath, color="black", lw=0)
            if not self.use_collection:
                self.ax.add_patch(tpatch)
            self.n_strokes.append(tpatch)

    def bbox(self):
        bbox = bbox_path(self.main_stroke.get_path())
        for stroke in self.n_strokes + self.c_strokes:
            bbox = bbox.expand(
                *bbox_path(stroke.get_path()))
        return bbox

    def as_patch_collection(self):
        strokes = [self.main_stroke] + self.n_strokes + self.c_strokes
        collect = mcollection.PatchCollection(
            strokes,
            match_original=True,
            capstyle="round" if self.stroke_slope != 0 else 'projecting'
        )
        return collect

    def set_transform(self, transform: mtransform.Affine2D):
        if self.main_stroke:
            self.main_stroke.set_transform(transform)
        for stroke in self.c_strokes + self.n_strokes:
            stroke.set_transform(transform)
        if self._collection is not None:
            self._collection.set_transform(transform)


class SequenceGlyph(object):
    sequence: glycopeptidepy.PeptideSequence
    sequence_position_glyphs: List[SequencePositionGlyph]
    glycan_composition_glyphs: Optional[GlycanCompositionGlyphs]

    step_coefficient: float
    x: float
    y: float
    size: float
    draw_glycan: bool

    ax: plt.Axes

    # Old and possibly unused
    multi_tier_annotation: bool
    label_fragments: bool

    def __init__(self, peptide, ax=None, size=1, step_coefficient=1.0, draw_glycan=False,
                 label_fragments=True, **kwargs):
        if not isinstance(peptide, sequence.PeptideSequenceBase):
            peptide = sequence.PeptideSequence(peptide)
        self.sequence = peptide
        self._fragment_index = None
        self.ax = ax
        self.size = size
        self.step_coefficient = step_coefficient
        self.sequence_position_glyphs = []
        self.annotations = []
        self.x = kwargs.get('x', 1)
        self.y = kwargs.get('y', 1)
        self.options = kwargs
        self.next_at_height = dict(c_term=defaultdict(float), n_term=defaultdict(float))
        self.label_fragments = label_fragments
        self.multi_tier_annotation = False
        self.draw_glycan = draw_glycan
        self.glycan_composition_glyphs = None
        self.render()

    def transform(self, transform):
        for glyph in self.sequence_position_glyphs:
            glyph.set_transform(transform + self.ax.transData)
        for annot in self.annotations:
            annot.set_transform(transform + self.ax.transData)
        if self.draw_glycan:
            self.glycan_composition_glyphs.set_transform(transform + self.ax.transData)
        return self

    def make_position_glyph(self, position, i, x, y, size, lw=0):
        glyph = SequencePositionGlyph(position, i, x, y, size=size, lw=lw)
        return glyph

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
            glyph = self.make_position_glyph(position, i, x, y, size=size, lw=self.options.get("lw", 0))
            glyph.render(ax)
            glyphs.append(glyph)
            x += size * self.step_coefficient
            i += 1
        if self.draw_glycan:
            self.glycan_composition_glyphs = GlycanCompositionGlyphs(
                self.sequence.glycan_composition, x + size * self.step_coefficient, y + 0.35, ax)
            self.glycan_composition_glyphs.render()
        return ax

    def __getitem__(self, i):
        return self.sequence_position_glyphs[i]

    def __len__(self):
        return len(self.sequence_position_glyphs)

    def next_between(self, index):
        for i, position in enumerate(self.sequence_position_glyphs, 1):
            if i == index:
                break
        mid_point_offset = (self.step_coefficient * self.size) / 1.3
        return position.x + mid_point_offset

    def draw_bar_at(self, index, height=0.25, color='red', **kwargs):
        x = self.next_between(index)
        y = self.y
        rect = mpatches.Rectangle(
            (x, y - height), 0.05, 1 + height, color=color, **kwargs)
        self.ax.add_patch(rect)
        self.annotations.append(rect)

    def draw_n_term_label(
            self, index, label, height=0.25, length=0.75, size=0.45,
            color='black', **kwargs):
        kwargs.setdefault('lw', self.options.get("lw", 0))
        x = self.next_between(index)
        y = self.y - height - size
        length *= self.step_coefficient
        label_x = x - length * 0.9
        label_y = y

        if self.next_at_height['n_term'][index] != 0:
            label_y = self.next_at_height['n_term'][index] - 0.4
            self.multi_tier_annotation = True
        else:
            label_y = y

        if self.label_fragments:
            tpath = textpath.TextPath(
                (label_x, label_y), label, size=size, prop=font_options)
            tpatch = mpatches.PathPatch(
                tpath, color=color, lw=kwargs.get('lw'))
            self.ax.add_patch(tpatch)
            self.annotations.append(tpatch)
            self.next_at_height['n_term'][index] = tpath.vertices[:, 1].min()

    def draw_n_term_annotation(
            self, index, height=0.25, length=0.75, color='red', **kwargs):
        x = self.next_between(index)
        y = self.y - height
        length *= self.step_coefficient
        rect = mpatches.Rectangle(
            (x - length, y), length, 0.07, color=color, **kwargs)
        self.ax.add_patch(rect)
        self.annotations.append(rect)

    def draw_c_term_label(
            self, index, label, height=0.25, length=0.75, size=0.45,
            color='black', **kwargs):
        kwargs.setdefault('lw', self.options.get("lw", 0))
        x = self.next_between(index)
        y = (self.y * 2) + height
        length *= self.step_coefficient
        label_x = x + length / 10.
        if self.next_at_height['c_term'][index] != 0:
            label_y = self.next_at_height['c_term'][index]
            self.multi_tier_annotation = True
        else:
            label_y = y + 0.2

        if self.label_fragments:
            tpath = textpath.TextPath(
                (label_x, label_y), label, size=size, prop=font_options)
            tpatch = mpatches.PathPatch(
                tpath, color=color, lw=kwargs.get('lw'))
            self.ax.add_patch(tpatch)
            self.annotations.append(tpatch)
            self.next_at_height['c_term'][index] = tpath.vertices[:, 1].max() + 0.1

    def draw_c_term_annotation(
            self, index, height=0., length=0.75, color='red', **kwargs):
        x = self.next_between(index)
        y = (self.y * 2) + height
        length *= self.step_coefficient
        rect = mpatches.Rectangle((x, y), length, 0.07, color=color, **kwargs)
        self.ax.add_patch(rect)
        self.annotations.append(rect)

    def bbox(self) -> BBox:
        bbox: BBox = None
        for glyph in self.sequence_position_glyphs:
            if bbox is None:
                bbox = glyph.bbox()
            else:
                bbox = bbox.expand(*glyph.bbox())
        if self.glycan_composition_glyphs is not None:
            if bbox is None:
                bbox = self.glycan_composition_glyphs.bbox()
            else:
                bbox = bbox.expand(*self.glycan_composition_glyphs.bbox())
        for glyph in self.annotations:
            if bbox is None:
                bbox = glyph.bbox()
            else:
                bbox = bbox.expand(*glyph.bbox())
        return bbox

    def layout(self):
        ax = self.ax
        # xmax = self.x + self.size * self.step_coefficient * len(self.sequence) + 1
        # if self.draw_glycan:
        #     xmax += self.glycan_composition_glyphs.xend - self.glycan_composition_glyphs.x + 1
        # ax.set_xlim(self.x - 1, xmax)
        # ax.set_ylim(self.y - 1, self.y + 2)
        # if self.multi_tier_annotation:
        #     ax.set_ylim(self.y - 2, self.y + 3)
        bbox = self.bbox()
        ax.set_xlim(bbox.xmin - 0.5, bbox.xmax + 0.5)
        ax.set_ylim(bbox.ymin - 0.5, bbox.ymax + 0.5)
        ax.axis("off")

    def draw(self):
        self.layout()

    def break_at(self, idx):
        if self._fragment_index is None:
            self._build_fragment_index()
        return self._fragment_index[idx]

    def _build_fragment_index(self, types=tuple('bycz')):
        self._fragment_index = [[] for i in range(len(self) + 1)]
        for series in types:
            series = IonSeries(series)
            if series.direction > 0:
                g = self.sequence.get_fragments(
                    series)
                for frags in g:
                    position = self._fragment_index[frags[0].position]
                    position.append(frags)
            else:
                g = self.sequence.get_fragments(
                    series)
                for frags in g:
                    position = self._fragment_index[
                        len(self) - frags[0].position]
                    position.append(frags)

    def build_annotations(self,
                          fragments: List[glycopeptidepy.PeptideFragment]) -> Tuple[List[
                                                            Tuple[glycopeptidepy.PeptideFragment, bool, str]
                                                        ]
                                                    ]:
        index = {}
        for i in range(1, len(self.sequence)):
            for series in self.break_at(i):
                for f in series:
                    index[f.name] = i
        n_annotations = []
        c_annotations = []

        for fragment in fragments:
            if hasattr(fragment, 'base_name'):
                key = fragment.base_name()
            else:
                key = fragment.name
            if key in index:
                is_glycosylated = fragment.is_glycosylated
                if key.startswith('b') or key.startswith('c'):
                    n_annotations.append((index[key], is_glycosylated, key))
                elif key.startswith('y') or key.startswith('z'):
                    c_annotations.append((index[key], is_glycosylated, key))
        return n_annotations, c_annotations

    def annotate_from_fragments_stroke(self, fragments: List[glycopeptidepy.PeptideFragment], **kwargs):
        n = len(self.sequence)
        kwargs.setdefault('lw', 2)
        kwargs.setdefault('color', 'red')
        kwargs.setdefault('glycosylated_color', 'green')
        kwargs.setdefault('v_step', 0.3)
        kwargs.setdefault('stroke_slope', 0)
        kwargs.setdefault('stroke_length', .8)
        kwargs.setdefault('glyph_padding', 0.2)

        lw = kwargs['lw']
        color = kwargs['color']
        glycosylated_color = kwargs['glycosylated_color']
        v_step = kwargs['v_step']
        stroke_slope = kwargs['stroke_slope']
        stroke_length = kwargs['stroke_length']
        glyph_padding = kwargs['glyph_padding']

        annotations = [
            FragmentStroke(
                i,
                self.next_between(i),
                self.y + 0.8 + glyph_padding,
                self.y - 0.1 - glyph_padding,
                n_length=stroke_length,
                c_length=stroke_length,
                v_step=v_step,
                stroke_slope=stroke_slope,
                ax=self.ax,
                main_color=color,
                lw=lw
            )
            for i in range(1, n)
        ]

        for frag in sorted(fragments, key=lambda x: x.mass):
            if frag.series.includes_peptide and frag.series.direction:
                if frag.series.direction > 0:
                    stroke = annotations[frag.position - 1]
                    if frag.is_glycosylated:
                        if not stroke.has_n_annotations():
                            stroke.add_n_label('', 'none')
                        stroke.add_n_label(frag.base_name(), glycosylated_color)
                    else:
                        stroke.add_n_label(frag.base_name(), color)
                else:
                    stroke = annotations[n - (frag.position + 1)]
                    if frag.is_glycosylated:
                        if not stroke.has_c_annotations():
                            stroke.add_c_label('', 'none')
                        stroke.add_c_label(frag.base_name(), glycosylated_color)
                    else:
                        stroke.add_c_label(frag.base_name(), color)

        annotations = [
            annot for annot in annotations if annot.has_annotations()
        ]
        for annot in annotations:
            annot.draw()
        self.annotations.extend(annotations)
        return annotations

    def annotate_from_fragments(self, fragments: List[glycopeptidepy.PeptideFragment], **kwargs):
        n_annotations, c_annotations = self.build_annotations(fragments)

        kwargs.setdefault("height", 0.25)
        kwargs_with_greater_height = kwargs.copy()
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
            except Exception:
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
    def from_spectrum_match(cls, spectrum_match, ax=None, set_layout=True, color='red',
                            glycosylated_color='forestgreen', **kwargs):
        annotation_options = kwargs.get("annotation_options", {})
        annotation_options['color'] = color
        annotation_options['glycosylated_color'] = glycosylated_color
        kwargs['annotation_options'] = annotation_options
        inst = cls(spectrum_match.target, ax=ax, **kwargs)
        fragments = [f for f in spectrum_match.solution_map.fragments()]
        # inst.annotate_from_fragments(fragments, **annotation_options)
        inst.annotate_from_fragments_stroke(fragments, **annotation_options)
        if set_layout:
            inst.layout()
        return inst


def glycopeptide_match_logo(glycopeptide_match, ax=None, color='red', glycosylated_color='forestgreen',
                            return_artist: bool=True, annotation_options: Optional[dict]=None, **kwargs):
    if annotation_options is None:
        annotation_options = {}
    annotation_options['color'] = color
    annotation_options['glycosylated_color'] = glycosylated_color
    kwargs['annotation_options'] = annotation_options
    inst = SequenceGlyph.from_spectrum_match(
        glycopeptide_match, color=color,
        glycosylated_color=glycosylated_color, ax=ax, **kwargs)
    if return_artist:
        return inst
    return inst.ax


class ProteoformDisplay(object):
    def __init__(self, proteoform, ax=None, width=60):
        self.proteoform = proteoform
        self.glyph_rows = []
        self.peptide_rows = []
        self.width = width
        self.x = 1
        self.y = 1


def draw_proteoform(proteoform, width=60):
    '''Draw a sequence logo for a proteoform
    '''
    fig, ax = plt.subplots(1)
    i = 0
    n = len(proteoform)
    y = 1
    x = 1
    rows = []
    if width > n:
        width = n + 1
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
    ax.set_xlim(0, width + 1)
    ax.set_ylim(-1, y + 1)
    return ax
