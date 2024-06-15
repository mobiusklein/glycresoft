import operator
import logging

from typing import Callable, List, Optional, TypeVar, Generic
from abc import ABCMeta, abstractmethod, abstractproperty
from collections.abc import Sequence as SequenceABC

try:
    from itertools import imap
except ImportError:
    imap = map

import dill

from six import add_metaclass

from sqlalchemy.orm.session import Session

from glycopeptidepy import HashableGlycanComposition

from .composition_network import CompositionGraph, n_glycan_distance

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


T = TypeVar("T")


@add_metaclass(ABCMeta)
class SearchableMassCollection(Generic[T]):
    def __len__(self):
        return len(self.structures)

    def __iter__(self):
        return iter(self.structures)

    def __getitem__(self, index) -> T:
        return self.structures[index]

    def _convert(self, bundle):
        return bundle

    @abstractproperty
    def lowest_mass(self) -> float:
        raise NotImplementedError()

    def reset(self, **kwargs):
        pass

    @abstractproperty
    def highest_mass(self) -> float:
        raise NotImplementedError()

    def search_mass_ppm(self, mass: float, error_tolerance: float) -> List[T]:
        """Search for the set of all items in :attr:`structures` within `error_tolerance` PPM
        of the queried `mass`.

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The range of mass errors (in Parts-Per-Million Error) to allow

        Returns
        -------
        list
            The list of instances which meet the criterion
        """
        tol = mass * error_tolerance
        return self.search_mass(mass, tol)

    @abstractmethod
    def search_mass(self, mass: float, error_tolerance: float=0.1) -> List[T]:
        raise NotImplementedError()

    def __repr__(self):
        size = len(self)
        return "{self.__class__.__name__}({size})".format(self=self, size=size)

    def mark_hit(self, match):
        # no-op to be taken when a structure
        # gets matched.
        return match

    def mark_batch(self):
        # no-op to be taken when a batch of
        # structures is to be searched. This hint
        # helps finalize any intermediary operations.
        pass

SequenceABC.register(SearchableMassCollection)


class MassDatabase(SearchableMassCollection[T]):
    """A quick-to-search database of :class:`HashableGlycanComposition` instances
    stored in memory.

    Implements the Sequence interface, with `__iter__`, `__len__`, and `__getitem__`.

    Attributes
    ----------
    structures : list
        A list of :class:`HashableGlycanComposition` instances, sorted by mass
    """
    def __init__(self, structures, network=None, distance_fn=n_glycan_distance,
                 glycan_composition_type=HashableGlycanComposition, sort=True):
        self.glycan_composition_type = glycan_composition_type
        if not isinstance(structures[0], glycan_composition_type):
            structures = list(map(glycan_composition_type, structures))
        self.structures = structures
        if sort:
            self.structures.sort(key=lambda x: x.mass())
        if network is None:
            self.network = CompositionGraph(self.structures)
            if distance_fn is not None:
                self.network.create_edges(1, distance_fn=distance_fn)
        else:
            self.network = network

    @classmethod
    def from_network(cls, network):
        structures = [node.composition for node in network.nodes]
        return cls(structures, network)

    @property
    def lowest_mass(self):
        return self.structures[0].mass()

    @property
    def highest_mass(self):
        return self.structures[-1].mass()

    @property
    def glycan_composition_network(self):
        return self.network

    def search_binary(self, mass, error_tolerance=1e-6):
        """Search within :attr:`structures` for the index of a structure
        with a mass nearest to `mass`, within `error_tolerance`

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The approximate error tolerance to accept

        Returns
        -------
        int
            The index of the structure with the nearest mass
        """
        lo = 0
        n = hi = len(self)

        while hi != lo:
            mid = (hi + lo) // 2
            x = self[mid]
            err = x.mass() - mass
            if abs(err) <= error_tolerance:
                best_index = mid
                best_error = abs(err)
                i = mid - 1
                while i >= 0:
                    x = self.structures[i]
                    err = abs((x.mass() - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    x = self.structures[i]
                    err = abs((x.mass() - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif (hi - lo) == 1:
                best_index = mid
                best_error = err
                i = mid - 1
                while i >= 0:
                    x = self.structures[i]
                    err = abs((x.mass() - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    x = self.structures[i]
                    err = abs((x.mass() - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif err > 0:
                hi = mid
            elif err < 0:
                lo = mid

    def search_mass(self, mass, error_tolerance=0.1):
        """Search for the set of all items in :attr:`structures` within `error_tolerance` Da
        of the queried `mass`.

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The range of mass errors (in Daltons) to allow

        Returns
        -------
        list
            The list of items instances which meet the mass criterion
        """
        if len(self) == 0:
            return []
        lo_mass = mass - error_tolerance
        hi_mass = mass + error_tolerance
        lo = self.search_binary(lo_mass)
        hi = self.search_binary(hi_mass) + 1
        return [structure for structure in self[lo:hi] if lo_mass <= structure.mass() <= hi_mass]

    def search_between(self, lower, higher):
        if len(self) == 0:
            return []
        lo = self.search_binary(lower)
        hi = self.search_binary(higher) + 1
        return self.__class__(
            [structure for structure in self[lo:hi] if lower <= structure.mass() <= higher], sort=False)


class MassObject(object):
    __slots__ = ('obj', 'mass')

    def __init__(self, obj, mass):
        self.obj = obj
        self.mass = mass

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def __reduce__(self):
        return self.__class__, (self.obj, self.mass)

    def __repr__(self):
        return "MassObject(%r, %r)" % (self.obj, self.mass)

    def __lt__(self, other):
        return self.mass < other.mass

    def __gt__(self, other):
        return self.mass > other.mass

    def __eq__(self, other):
        return abs(self.mass - other.mass) < 1e-3

    def __ne__(self, other):
        return abs(self.mass - other.mass) >= 1e-3


def identity(x):
    return x


class _NeutralMassDatabase(SearchableMassCollection[T]):
    def __init__(self, structures, mass_getter=operator.attrgetter("calculated_mass"), sort=True):
        self.mass_getter = mass_getter
        self.structures = self._prepare(structures, sort)

    def _prepare(self, structures, sort=True):
        temp = []
        for obj in structures:
            mass = self.mass_getter(obj)
            temp.append(MassObject(obj, mass))
        if sort:
            temp.sort()
        return temp

    def __reduce__(self):
        return self.__class__, ([], identity, False), self.__getstate__()

    def __getstate__(self):
        return {
            'mass_getter': dill.dumps(self.mass_getter),
            'structures': self.structures
        }

    def __setstate__(self, state):
        self.mass_getter = dill.loads(state['mass_getter'])
        self.structures = state['structures']

    def __iter__(self):
        for obj in self.structures:
            yield obj.obj

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [mo.obj for mo in self.structures[i]]
        return self.structures[i].obj

    @property
    def lowest_mass(self):
        return self.structures[0].mass

    @property
    def highest_mass(self):
        return self.structures[-1].mass

    def search_binary(self, mass, error_tolerance=1e-6):
        """Search within :attr:`structures` for the index of a structure
        with a mass nearest to `mass`, within `error_tolerance`

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The approximate error tolerance to accept

        Returns
        -------
        int
            The index of the structure with the nearest mass
        """
        lo = 0
        n = hi = len(self.structures)
        while hi != lo:
            mid = (hi + lo) // 2
            x = self.structures[mid]
            err = x.mass - mass
            if abs(err) <= error_tolerance:
                best_index = mid
                best_error = err
                i = mid - 1
                while i >= 0:
                    x = self.structures[i]
                    err = abs((x.mass - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    x = self.structures[i]
                    err = abs((x.mass - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif (hi - lo) == 1:
                best_index = mid
                best_error = err
                i = mid - 1
                while i >= 0:
                    x = self.structures[i]
                    err = abs((x.mass - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    x = self.structures[i]
                    err = abs((x.mass - mass) / mass)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif err > 0:
                hi = mid
            elif err < 0:
                lo = mid

    def search_mass(self, mass, error_tolerance=0.1):
        """Search for the set of all items in :attr:`structures` within `error_tolerance` Da
        of the queried `mass`.

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The range of mass errors (in Daltons) to allow

        Returns
        -------
        list
            The list of instances which meet the criterion
        """
        if len(self) == 0:
            return []
        lo_mass = mass - error_tolerance
        hi_mass = mass + error_tolerance
        lo = self.search_binary(lo_mass)
        hi = self.search_binary(hi_mass) + 1
        return [structure.obj for structure in self.structures[lo:hi] if lo_mass <= structure.mass <= hi_mass]

    def search_between(self, lower, higher):
        lo = self.search_binary(lower)
        hi = self.search_binary(higher) + 1
        return self.__class__(
            [structure.obj for structure in self.structures[lo:hi] if lower <= structure.mass <= higher],
            self.mass_getter, sort=False)

    def _merge_neutral_mass_database(self, other, id_fn=id):
        new = {id_fn(x.obj): x for x in self.structures}
        new.update({id_fn(x.obj): x for x in other.structures})
        structures = list(new.values())
        structures.sort()
        self.structures = structures
        return self

    def _merge_other(self, other, id_fn=id):
        new = {id_fn(x): x for x in self}
        new.update({id_fn(x): x for x in other})
        structures = list(new.values())
        self.structures = self._pack(structures, True)
        return self

    def merge(self, other, id_fn=id):
        if isinstance(other, NeutralMassDatabase):
            return self._merge_neutral_mass_database(other, id_fn)
        else:
            return self._merge_other(other, id_fn)


try:
    from glycresoft._c.database.mass_collection import (
        MassObject, NeutralMassDatabaseImpl as _NeutralMassDatabaseImpl)

    class NeutralMassDatabase(_NeutralMassDatabaseImpl, _NeutralMassDatabase[T]):
        pass
except ImportError:
    NeutralMassDatabase = _NeutralMassDatabase


class ConcatenatedDatabase(SearchableMassCollection[T]):
    def __init__(self, databases):
        self.databases = list(databases)

    def reset(self, **kwargs):
        return [d.reset(**kwargs) for d in self.databases]

    @property
    def lowest_mass(self):
        return min(db.lowest_mass for db in self.databases)

    @property
    def highest_mass(self):
        return max(db.highest_mass for db in self.databases)

    def search_mass(self, mass, error_tolerance=0.1):
        """Search for the set of all items in :attr:`structures` within `error_tolerance` Da
        of the queried `mass`.

        Parameters
        ----------
        mass : float
            The neutral mass to search for
        error_tolerance : float, optional
            The range of mass errors (in Daltons) to allow

        Returns
        -------
        list
            The list of instances which meet the criterion
        """
        hits = []
        for database in self.databases:
            hits += database.search_mass(mass, error_tolerance)
        return hits

    def search_between(self, lower, higher):
        result = []
        for database in self.databases:
            subset = database.search_between(lower, higher)
            if len(subset) > 0:
                result.append(subset)
        return self.__class__(result)

    def add(self, database):
        self.databases.append(database)
        return self

    def __len__(self):
        return sum(map(len, self.databases))


class SearchableMassCollectionWrapper(SearchableMassCollection[T]):
    searchable_mass_collection: SearchableMassCollection

    @property
    def highest_mass(self):
        return self.searchable_mass_collection.highest_mass

    @property
    def lowest_mass(self):
        return self.searchable_mass_collection.lowest_mass

    def reset(self, **kwargs):
        return self.searchable_mass_collection.reset(**kwargs)

    # Fake the disk-backed interface

    @property
    def session(self) -> Session:
        return self.searchable_mass_collection.session

    @property
    def hypothesis_id(self):
        return self.searchable_mass_collection.hypothesis_id

    @property
    def hypothesis(self):
        return self.searchable_mass_collection.hypothesis

    def __len__(self):
        return len(self.searchable_mass_collection)

    @property
    def peptides(self):
        return self.searchable_mass_collection.peptides

    @property
    def proteins(self):
        return self.searchable_mass_collection.proteins

    def query(self, *args, **kwargs):
        return self.searchable_mass_collection.query(*args, **kwargs)


class TransformingMassCollectionAdapter(SearchableMassCollectionWrapper[T]):
    def __init__(self, searchable_mass_collection, transformer):
        self.searchable_mass_collection = searchable_mass_collection
        self.transformer = transformer

    def search_mass_ppm(self, mass: float, error_tolerance: float=1e-5) -> List[T]:
        result = self.searchable_mass_collection.search_mass_ppm(mass, error_tolerance)
        return [self.transformer(r) for r in result]

    def __reduce__(self):
        return self.__class__, (self.searchable_mass_collection, None), self.__getstate__()

    def __getstate__(self):
        return {
            "transformer": dill.dumps(self.transformer)
        }

    def __setstate__(self, state):
        self.transformer = dill.loads(state['transformer'])

    def search_mass(self, mass, error_tolerance):  # pylint: disable=signature-differs
        result = self.searchable_mass_collection.search_mass(mass, error_tolerance)
        return [self.transformer(r) for r in result]

    def __getitem__(self, i):
        return self.transformer(self.searchable_mass_collection[i])

    def __iter__(self):
        return imap(self.transformer, self.searchable_mass_collection)

    def search_between(self, lower, higher):
        return self.__class__(
            self.searchable_mass_collection.search_between(lower, higher), self.transformer)


class MassCollectionProxy(SearchableMassCollectionWrapper[T]):
    _session: Optional[Session]
    session_resolver: Callable[[], Session]

    def __init__(self, resolver, session_resolver=None, hypothesis_id=None, hypothesis_resolver=None):
        self._searchable_mass_collection = None
        self._session = None
        self._hypothesis_id = hypothesis_id
        self._hypothesis = None
        self.resolver = resolver
        self.session_resolver = session_resolver
        self.hypothesis_resolver = hypothesis_resolver

    @property
    def hypothesis_id(self):
        if self._hypothesis_id is not None:
            return self._hypothesis_id
        else:
            logger.info("Loading hypothesis ID from resolver")
            self._hypothesis_id = self.searchable_mass_collection.hypothesis_id
            return self._hypothesis_id

    @property
    def session(self) -> Session:
        if self._session is None and self.session_resolver is not None:
            self._session = self.session_resolver()
        if self._session is not None:
            return self._session
        else:
            logger.info("Loading session from primary resolver")
            self._session = self.searchable_mass_collection.session
            return self._session

    @property
    def hypothesis(self):
        if self._hypothesis is not None:
            return self._hypothesis
        elif self.hypothesis_resolver:
            self._hypothesis = self.hypothesis_resolver()
            return self._hypothesis
        else:
            return super().hypothesis

    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    @property
    def searchable_mass_collection(self):
        if self._searchable_mass_collection is None:
            self._searchable_mass_collection = self.resolver()
        return self._searchable_mass_collection

    def search_mass_ppm(self, mass: float, error_tolerance: float=1e-5) -> List[T]:
        result = self.searchable_mass_collection.search_mass_ppm(mass, error_tolerance)
        return result

    def __reduce__(self):
        return self.__class__, (None,), self.__getstate__()

    def __getstate__(self):
        return {
            "resolver": dill.dumps(self.resolver),
            "session_resolver": dill.dumps(self.session_resolver),
            "hypothesis_resolver": dill.dumps(self.hypothesis_resolver),
            "hypothesis_id": self._hypothesis_id
        }

    def _has_resolved(self):
        return self._searchable_mass_collection is not None

    def __repr__(self):
        if self._has_resolved():
            size = len(self)
        else:
            size = "?"
        return "{self.__class__.__name__}({size})".format(self=self, size=size)

    def __setstate__(self, state):
        self.resolver = dill.loads(state['resolver'])
        self.session_resolver = dill.loads(state['session_resolver'])
        self.hypothesis_resolver = dill.loads(state['hypothesis_resolver'])
        self._hypothesis_id = state['hypothesis_id']

    def search_mass(self, mass: float, error_tolerance: float) -> List[T]:  # pylint: disable=signature-differs
        result = self.searchable_mass_collection.search_mass(mass, error_tolerance)
        return result

    def __getitem__(self, i) -> T:
        return self.searchable_mass_collection[i]

    def __iter__(self):
        return iter(self.searchable_mass_collection)

    def search_between(self, lower: float, higher: float) -> List[T]:
        return self.searchable_mass_collection.search_between(lower, higher)
