import operator

from glycopeptidepy import HashableGlycanComposition

from .composition_network import CompositionGraph, n_glycan_distance


class SearchableMassCollection(object):
    def __len__(self):
        return len(self.structures)

    def __iter__(self):
        return iter(self.structures)

    def __getitem__(self, index):
        return self.structures[index]

    def _convert(self, bundle):
        return bundle

    @property
    def lowest_mass(self):
        raise NotImplementedError()

    @property
    def highest_mass(self):
        raise NotImplementedError()

    def search_mass_ppm(self, mass, error_tolerance):
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

    def search_mass(self, mass, error_tolerance=0.1):
        raise NotImplementedError()

    def __repr__(self):
        size = len(self)
        return "{self.__class__.__name__}({size})".format(self=self, size=size)


class MassDatabase(SearchableMassCollection):
    """A quick-to-search database of :class:`HashableGlycanComposition` instances
    stored in memory.

    Implements the Sequence interface, with `__iter__`, `__len__`, and `__getitem__`.

    Attributes
    ----------
    structures : list
        A list of :class:`HashableGlycanComposition` instances, sorted by mass
    """
    def __init__(self, structures, network=None, distance_fn=n_glycan_distance,
                 glycan_composition_type=HashableGlycanComposition):
        self.glycan_composition_type = glycan_composition_type
        if not isinstance(structures[0], glycan_composition_type):
            structures = list(map(glycan_composition_type, structures))
        self.structures = structures
        self.structures.sort(key=lambda x: x.mass())
        if network is None:
            self.network = CompositionGraph(self.structures)
            if distance_fn is not None:
                self.network._create_edges(1, distance_fn=distance_fn)
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
        hi = len(self)

        while hi != lo:
            mid = (hi + lo) / 2
            x = self[mid]
            err = x.mass() - mass
            if abs(err) <= error_tolerance:
                return mid
            elif (hi - lo) == 1:
                return mid
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
            The list of :class:`HashableGlycanComposition` instances which meet the criterion
        """
        if len(self) == 0:
            return []
        lo_mass = mass - error_tolerance
        hi_mass = mass + error_tolerance
        lo = self.search_binary(lo_mass)
        hi = self.search_binary(hi_mass) + 1
        return [structure for structure in self[lo:hi] if lo_mass <= structure.mass() <= hi_mass]


class NeutralMassDatabase(SearchableMassCollection):
    def __init__(self, structures, mass_getter=operator.attrgetter("calculated_mass")):
        self.structures = sorted(structures, key=mass_getter)
        self.mass_getter = mass_getter

    @property
    def lowest_mass(self):
        return self.mass_getter(self.structures[0])

    @property
    def highest_mass(self):
        return self.mass_getter(self.structures[-1])

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
        hi = len(self)

        while hi != lo:
            mid = (hi + lo) / 2
            x = self[mid]
            err = self.mass_getter(x) - mass
            if abs(err) <= error_tolerance:
                return mid
            elif (hi - lo) == 1:
                return mid
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
        return [structure for structure in self[lo:hi] if lo_mass <= self.mass_getter(structure) <= hi_mass]


class ConcatenatedDatabase(SearchableMassCollection):
    def __init__(self, databases):
        self.databases = list(databases)

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
