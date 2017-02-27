from glypy.composition import formula as _formulastr, Composition
from ms_deisotope.utils import dict_proxy
from collections import defaultdict


@dict_proxy("composition")
class Formula(object):
    """Represent a hashable composition wrapper and a backing object supplying
    other attributes and properties. Used to make handling of objects which posses
    composition-like behavior easier.

    This class implements the Mapping interface, supplying its composition as key-value data.

    Attributes
    ----------
    composition : Composition
        The elemental composition to proxy
    data : object
        An arbitrary object to proxy attribute look ups to.
        Usually the source of :attr:`composition`
    name : str
        The formula for the given composition. Used for hashing and equality testing
    """
    def __init__(self, composition, data):
        self.composition = composition
        self.name = _formulastr(composition)
        self.data = data
        self._hash = hash(self.name)

    def __getattr__(self, attr):
        if attr == "data":
            return None
        return getattr(self.data, attr)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self._hash

    def total_composition(self):
        return Composition(self.composition)


@dict_proxy("composition")
class CommonComposition(object):
    """A type to represent multiple objects which share the same elemental composition,
    holding a list of all the objects which it shares formulae with.

    This class implements the Mapping interface, supplying its composition as key-value data.

    Attributes
    ----------
    base_data : list
        All the objects which are bundled together.
    composition : Composition
        The composition being represented.
    """
    def __init__(self, composition, base_data):
        self.composition = composition
        self.base_data = base_data

    @property
    def name(self):
        return '|'.join(x.name for x in self.base_data)

    @classmethod
    def aggregate(cls, iterable):
        """Combine an iterable of objects which are hashable and function as
        composition mappings into a list of :class:`CommonComposition` instances such
        that each composition is represented exactly once.

        Parameters
        ----------
        iterable : Iterable
            An iterable of any type of object which is both hashable and provides a Composition-like
            interface

        Returns
        -------
        list
        """
        agg = defaultdict(list)
        for item in iterable:
            agg[item].append(item.data)
        results = []
        for key, collected in agg.items():
            results.append(cls(key, collected))
        return results

    def __repr__(self):
        return "CommonComposition(%s)" % dict(self.composition)

    def total_composition(self):
        return Composition(self.composition)
