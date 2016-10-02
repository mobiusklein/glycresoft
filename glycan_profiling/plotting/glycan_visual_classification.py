from collections import OrderedDict
from itertools import cycle

from glypy.composition.glycan_composition import FrozenGlycanComposition, FrozenMonosaccharideResidue
from glycopeptidepy.utils import simple_repr

from matplotlib import patches as mpatches


def _degree_monosaccharide_alteration(x):
    try:
        if not isinstance(x, FrozenMonosaccharideResidue):
            x = FrozenMonosaccharideResidue.from_iupac_lite(str(x))
        return (len(x.modifications), len(x.substituent_links))
    except:
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
            a = FrozenGlycanComposition.parse(a)
            b = FrozenGlycanComposition.parse(b)
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


class CompositionRangeRule(object):
    def __init__(self, name, low=None, high=None, required=True):
        self.name = name
        self.low = low
        self.high = high
        self.required = required

    __repr__ = simple_repr

    def get_composition(self, obj):
        try:
            composition = obj.glycan_composition
        except:
            composition = FrozenGlycanComposition.parse(obj)
        return composition

    def __call__(self, obj):
        composition = self.get_composition(obj)
        if self.name in composition:
            if self.low is None:
                return composition[self.name] <= self.high
            elif self.high is None:
                return self.low <= composition[self.name]
            return self.low <= composition[self.name] <= self.high
        else:
            return not self.required


class CompositionRuleClassifier(object):
    def __init__(self, name, rules):
        self.name = name
        self.rules = rules

    def __iter__(self):
        return iter(self.rules)

    def __call__(self, obj):
        for rule in self:
            if not rule(obj):
                return False
        return True

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    __repr__ = simple_repr


class allset(frozenset):
    def __contains__(self, k):
        return True


class GlycanCompositionClassifierColorizer(object):
    def __init__(self, rule_color_map=None, default=None):
        self.rule_color_map = rule_color_map or {}
        self.default = default

    def __call__(self, obj):
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
    (CompositionRuleClassifier("High Mannose", [CompositionRangeRule("HexNAc", 2, 2)]), '#1f77b4'),
    (CompositionRuleClassifier("Hybrid", [CompositionRangeRule("HexNAc", 3, 3)]), '#ff7f0e'),
    (CompositionRuleClassifier("Bi-Antennerary", [CompositionRangeRule("HexNAc", 4, 4)]), '#2ca02c'),
    (CompositionRuleClassifier("Tri-Antennerary", [CompositionRangeRule("HexNAc", 5, 5)]), '#d62728'),
    (CompositionRuleClassifier("Tetra-Antennerary", [CompositionRangeRule("HexNAc", 6, 6)]), '#9467bd'),
    (CompositionRuleClassifier("Penta-Antennerary", [CompositionRangeRule("HexNAc", 7, 7)]), '#8c564b'),
    (CompositionRuleClassifier("Supra-Penta-Antennerary", [CompositionRangeRule("HexNAc", 8)]), 'brown')
]))
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
                item = FrozenGlycanComposition.parse(item)
            residues.update(item)

        residues = sorted(residues, key=_degree_monosaccharide_alteration)
        self.residues = self.order_chooser.key_order(residues)

    def transform(self):
        for item in self._input_series:
            if isinstance(item, basestring):
                item = FrozenGlycanComposition.parse(item)
            counts = [str(item[r]) for r in self.residues]
            yield '[%s]' % '; '.join(counts)

    __iter__ = transform

    __repr__ = simple_repr

    @property
    def label_key(self):
        return "Key: [%s]" % '; '.join(map(str, self.residues))
