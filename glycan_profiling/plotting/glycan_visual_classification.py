from collections import OrderedDict

from matplotlib import patches as mpatches

from glypy.structure.glycan_composition import FrozenGlycanComposition, FrozenMonosaccharideResidue
from glycopeptidepy.utils import simple_repr

from glycan_profiling.database.composition_network import (
    CompositionRangeRule,
    CompositionRuleClassifier,
    CompositionRatioRule,
    normalize_composition)


def _degree_monosaccharide_alteration(x):
    try:
        if not isinstance(x, FrozenMonosaccharideResidue):
            x = FrozenMonosaccharideResidue.from_iupac_lite(str(x))
        return (len(x.modifications), len(x.substituent_links))
    except Exception:
        return (float('inf'), float('inf'))


class GlycanCompositionOrderer(object):
    def __init__(self, priority_residues=None, sorter=None):
        self.priority_residues = priority_residues or []
        self.sorter = sorter or _degree_monosaccharide_alteration

        self.priority_residues = [
            residue if isinstance(
                residue, FrozenMonosaccharideResidue) else FrozenMonosaccharideResidue.from_iupac_lite(
                residue)
            for residue in self.priority_residues
        ]

        outer = self

        class _ComparableProxy(object):
            def __init__(self, composition, obj=None):
                if isinstance(composition, basestring):
                    composition = FrozenGlycanComposition.parse(composition)
                if obj is None:
                    obj = composition
                self.composition = composition
                self.obj = obj

            def __iter__(self):
                return iter(self.composition)

            def __getitem__(self, key):
                return self.composition[key]

            def __lt__(self, other):
                return outer(self, other) < 0

            def __gt__(self, other):
                return outer(self, other) > 0

            def __eq__(self, other):
                return outer(self, other) == 0

            def __le__(self, other):
                return outer(self, other) <= 0

            def __ge__(self, other):
                return outer(self, other) >= 0

            def __ne__(self, other):
                return outer(self, other) != 0

        self._comparable_proxy = _ComparableProxy

    def sort(self, compositions, key=None, reverse=False):
        if key is None:

            def key(x):
                return x

        proxies = [self._comparable_proxy(key(c), c) for c in compositions]
        proxies = sorted(proxies, reverse=reverse)
        out = [p.obj for p in proxies]
        return out

    def key_order(self, keys):
        keys = list(keys)
        for r in reversed(self.priority_residues):
            try:
                i = keys.index(r)
                keys.pop(i)
                keys = [r] + keys
            except ValueError:
                pass
        return keys

    def __call__(self, a, b):
        if isinstance(a, basestring):
            a = normalize_composition(a)
            b = normalize_composition(b)
        keys = self.key_order(sorted(set(a) | set(b), key=self.sorter))

        for key in keys:
            if a[key] < b[key]:
                return -1
            elif a[key] > b[key]:
                return 1
            else:
                continue
        return 0

    __repr__ = simple_repr


class allset(frozenset):
    def __contains__(self, k):
        return True


class GlycanCompositionClassifierColorizer(object):
    def __init__(self, rule_color_map=None, default=None):
        self.rule_color_map = rule_color_map or {}
        self.default = default

    def __call__(self, obj):
        obj = normalize_composition(obj)
        for rule, color in self.rule_color_map.items():
            if rule(obj):
                return color
        if self.default:
            return self.default
        raise ValueError("Could not classify %r" % obj)

    def classify(self, obj):
        for rule, color in self.rule_color_map.items():
            if rule(obj):
                return rule.name
        return None

    __repr__ = simple_repr

    def make_legend(self, included=allset(), alpha=0.5):
        return [
            mpatches.Patch(
                label=rule.name, color=color, alpha=alpha) for rule, color in self.rule_color_map.items()
            if rule.name in included
        ]


NGlycanCompositionColorizer = GlycanCompositionClassifierColorizer(OrderedDict([
    (CompositionRuleClassifier("Paucimannose", [
        CompositionRangeRule("HexNAc", 2, 2) & CompositionRangeRule("Hex", 0, 4)]), "#f05af0"),
    (CompositionRuleClassifier("High Mannose", [CompositionRangeRule("HexNAc", 2, 2)]), '#1f77b4'),
    (CompositionRuleClassifier("Hybrid", [CompositionRangeRule("HexNAc", 3, 3)]), '#ff7f0e'),
    (CompositionRuleClassifier("Bi-Antennary", [CompositionRangeRule("HexNAc", 4, 4)]), '#2ca02c'),
    (CompositionRuleClassifier("Tri-Antennary", [CompositionRangeRule("HexNAc", 5, 5)]), '#d62728'),
    (CompositionRuleClassifier("Tetra-Antennary", [CompositionRangeRule("HexNAc", 6, 6)]), '#9467bd'),
    (CompositionRuleClassifier("Penta-Antennary", [CompositionRangeRule("HexNAc", 7, 7)]), '#8c564b'),
    (CompositionRuleClassifier("Supra-Penta-Antennary", [CompositionRangeRule("HexNAc", 8)]), 'brown'),
    (CompositionRuleClassifier("Low Sulfate GAG", [CompositionRatioRule("HexN", "@sulfate", (0, 2))]), "#2aaaaa"),
    (CompositionRuleClassifier("High Sulfate GAG", [CompositionRatioRule("HexN", "@sulfate", (2, 4))]), "#88faaa")
]), default="slateblue")

NGlycanCompositionOrderer = GlycanCompositionOrderer(["HexNAc", "Hex", "Fucose", "NeuAc"])

_null_color_chooser = GlycanCompositionClassifierColorizer({}, default='blue')


class GlycanLabelTransformer(object):
    def __init__(self, label_series, order_chooser):
        self._input_series = label_series
        self.order_chooser = order_chooser
        self.residues = None
        self._infer_compositions()

    def _infer_compositions(self):
        residues = set()
        for item in self._input_series:
            if isinstance(item, basestring):
                item = normalize_composition(item)
            residues.update(item)

        residues = sorted(residues, key=_degree_monosaccharide_alteration)
        self.residues = self.order_chooser.key_order(residues)

    def transform(self):
        for item in self._input_series:
            if isinstance(item, basestring):
                item = normalize_composition(item)
            counts = [str(item[r]) for r in self.residues]
            yield '[%s]' % '; '.join(counts)

    __iter__ = transform

    __repr__ = simple_repr

    @property
    def label_key(self):
        return "Key: [%s]" % '; '.join(map(str, self.residues))
