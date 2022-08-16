import json
from six import add_metaclass

from glycopeptidepy.io import uniprot


class AnnotationMeta(type):
    _cache = {}

    def __new__(cls, name, parents, attrs):
        new_type = type.__new__(cls, name, parents, attrs)
        try:
            cls._cache[name] = new_type
        except AttributeError:
            pass
        return new_type

    def _type_for_name(self, feature_type):
        return self._cache[feature_type]

    def from_dict(self, d):
        name = d.pop('__class__')
        impl = self._type_for_name(name)
        return impl(**d)


@add_metaclass(AnnotationMeta)
class AnnotationBase(object):
    def __init__(self, feature_type, description):
        self.feature_type = feature_type
        self.description = description

    def to_dict(self):
        d = {}
        d['feature_type'] = self.feature_type
        d['description'] = self.description
        d['__class__'] = self.__class__.__name__
        return d

    def __eq__(self, other):
        if other is None:
            return False
        return self.feature_type == other.feature_type and self.description == other.description

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.feature_type, self.description))


class AnnotatedResidue(AnnotationBase):
    def __init__(self, position, feature_type, description):
        super(AnnotatedResidue, self).__init__(feature_type, description)
        self.position = position

    @property
    def start(self):
        return self.position

    @property
    def end(self):
        return self.position + 1

    def to_dict(self):
        d = super(AnnotatedResidue, self).to_dict()
        d['position'] = self.position
        return d

    def __eq__(self, other):
        res = super(AnnotatedResidue, self).__eq__(other)
        if not res:
            return res
        return self.position == other.position

    def __hash__(self):
        return hash((self.feature_type, self.description, self.position))


class AnnotatedInterval(AnnotationBase):
    def __init__(self, start, end, feature_type, description):
        super(AnnotatedInterval, self).__init__(feature_type, description)
        self.start = start
        self.end = end

    def to_dict(self):
        d = super(AnnotatedInterval, self).to_dict()
        d['start'] = self.start
        d['end'] = self.end
        return d

    def __eq__(self, other):
        res = super(AnnotatedInterval, self).__eq__(other)
        if not res:
            return res
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.feature_type, self.description, self.start, self.end))


class PeptideBase(AnnotatedInterval):
    feature_type = None

    def __init__(self, start, end, **kwargs):
        super(PeptideBase, self).__init__(
            start, end, self.feature_type, self.feature_type)

    def __repr__(self):
        template = '{self.__class__.__name__}({self.start}, {self.end})'
        return template.format(self=self)


class SignalPeptide(PeptideBase):
    feature_type = 'signal peptide'


class Propeptide(PeptideBase):
    feature_type = 'propeptide'


class TransitPeptide(PeptideBase):
    feature_type = "transit peptide"


class Peptide(PeptideBase):
    feature_type = 'peptide'


class MatureProtein(PeptideBase):
    feature_type = 'mature protein'


class ProteolyticSite(AnnotatedResidue):
    feature_type = 'proteolytic site'

    def __init__(self, position, **kwargs):
        super(ProteolyticSite, self).__init__(
            self.position, self.feature_type, self.feature_type)


class ModifiedResidue(AnnotatedResidue):
    feature_type = 'modified residue'

    def __init__(self, position, modification, **kwargs):
        super(ModifiedResidue, self).__init__(position, self.feature_type, modification)

    @property
    def modification(self):
        return self.description

    def to_dict(self):
        d = super(ModifiedResidue, self).to_dict()
        d['modification'] = d.pop('description')
        return d

    def __repr__(self):
        return "{self.__class__.__name__}({self.position}, {self.modification!r})".format(self=self)


class AnnotationCollection(object):
    def __init__(self, items):
        self.items = list(items)

    def append(self, item):
        self.items.append(item)

    def __getitem__(self, i):
        return self.items[i]

    def __setitem__(self, i, item):
        self.items[i] = item

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return "{self.__class__.__name__}({self.items})".format(self=self)

    def to_json(self):
        return [a.to_dict() for a in self]

    @classmethod
    def from_json(cls, d):
        return cls([AnnotationBase.from_dict(di) for di in d])

    def dump(self, fp):
        json.dump(self.to_json())

    @classmethod
    def load(cls, fp):
        d = json.load(fp)
        inst = cls.from_json(d)
        return inst

    def __eq__(self, other):
        return self.items == list(other)

    def __ne__(self, other):
        return not (self == other)


def from_uniprot(record):
    mapping = {
        uniprot.SignalPeptide.feature_type: SignalPeptide,
        uniprot.Propeptide.feature_type: Propeptide,
        uniprot.TransitPeptide.feature_type: TransitPeptide,
        uniprot.MatureProtein.feature_type: MatureProtein,
        uniprot.ModifiedResidue.feature_type: ModifiedResidue,
    }
    annotations = []
    for feature in record.features:
        if not feature.is_defined:
            continue
        try:
            annotation_tp = mapping[feature.feature_type]
            if issubclass(annotation_tp, PeptideBase):
                annotations.append(
                    annotation_tp(feature.start, feature.end))
            elif issubclass(annotation_tp, AnnotatedResidue):
                annotations.append(
                    annotation_tp(feature.position, feature.description))
        except KeyError:
            continue
    return AnnotationCollection(annotations)
