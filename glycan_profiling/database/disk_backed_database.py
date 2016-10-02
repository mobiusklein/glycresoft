import operator


from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from sqlalchemy import select
from glycresoft_sqlalchemy.data_model import (
    DatabaseManager, TheoreticalGlycopeptideComposition, func,
    Protein, Hypothesis, InformedPeptide)

from .mass_collection import SearchableMassCollection, NeutralMassDatabase
from .lru import LRUCache

try:
    basestring
except:
    basestring = (str, bytes)


_empty_interval = NeutralMassDatabase([])


class DiskBackedStructureDatabase(SearchableMassCollection):
    # Good for 10 ppm error intervals on structures up to 1e5 Da
    loading_interval = 1.
    threshold_cache_total_count = 1e3

    def __init__(self, connection, hypothesis_id, cache_size=5, loading_interval=1.,
                 threshold_cache_total_count=1e3,
                 model_type=TheoreticalGlycopeptideComposition):
        if isinstance(connection, basestring):
            connection = DatabaseManager(connection)
        self.manager = connection
        self.session = self.manager()
        self.hypothesis_id = hypothesis_id
        self.model_type = model_type
        self.loading_interval = loading_interval
        self.threshold_cache_total_count = threshold_cache_total_count
        self._intervals = LRUIntervalSet([], cache_size)
        self._ignored_intervals = IntervalSet([])
        self.proteins = _ProteinIndex(self.session, self.hypothesis_id)
        self.peptides = _PeptideIndex(self.session, self.hypothesis_id)

    def __reduce__(self):
        return self.__class__, (
            self.manager, self.hypothesis_id, self.cache_size, self.loading_interval,
            self.threshold_cache_total_count, self.model_type)

    def _upkeep_memory_intervals(self):
        n = len(self._intervals)
        if n > 1:
            while (len(self._intervals) > 1 and
                   self._intervals.total_count >
                   self.threshold_cache_total_count):
                self._intervals.remove_lru_interval()

    @property
    def hypothesis(self):
        return self.session.query(Hypothesis).get(self.hypothesis_id)

    @property
    def structures(self):
        return self.session.query(self.model_type).filter(
            self.model_type.hypothesis_id == self.hypothesis_id).order_by(
            self.model_type.calculated_mass)

    def __len__(self):
        return self.structures.count()

    def ignore_interval(self, interval):
        self._ignored_intervals.insert_interval(interval)

    def is_interval_ignored(self, interval):
        is_ignored = self._ignored_intervals.find_interval(interval)
        return is_ignored is not None and is_ignored.contains_interval(interval)

    def insert_interval(self, mass):
        tree = self.make_memory_interval(mass)
        # We won't insert this node.
        if len(tree) == 0:
            self.ignore_interval(FixedQueryInterval(mass))
            return tree
        node = MassIntervalNode(tree)
        nearest_interval = self._intervals.find_interval(node)
        # Should an insert be performed if the query just didn't overlap well
        # with the database?
        if nearest_interval is None:
            self._intervals.insert_interval(node)
            return node.group
        elif not nearest_interval.contains_interval(node):
            self._intervals.insert_interval(node)
            return node.group
        else:
            return nearest_interval.group

    def has_interval(self, mass, ppm_error_tolerance):
        q = PPMQueryInterval(mass, ppm_error_tolerance)
        match = self._intervals.find_interval(q)
        if match is not None:
            # We are completely contained in an existing interval, so just
            # use that one.
            if match.contains_interval(q):
                return match.group
            # We overlap with an extending interval, so we should populate
            # the new one and merge them.
            elif match.overlaps(q):
                self._intervals.extend_interval(
                    match, self.make_memory_interval(mass))
                return match.group
            # We might need to insert a new interval
            else:
                if self.is_interval_ignored(q):
                    return match.group
                else:
                    return self.insert_interval(mass)
        else:
            is_ignored = self._ignored_intervals.find_interval(q)
            if is_ignored is not None and is_ignored.contains_interval(q):
                return _empty_interval
            else:
                return self.insert_interval(mass)

    def search_mass_ppm(self, mass, error_tolerance):
        self._upkeep_memory_intervals()
        return self.has_interval(mass, error_tolerance).search_mass_ppm(mass, error_tolerance)

    def search_mass(self, mass, error_tolerance):
        model = self.model_type
        return self.session.query(self.model_type).filter(
            model.hypothesis_id == self.hypothesis_id,
            model.calculated_mass.between(
                mass - error_tolerance, mass + error_tolerance)).all()

    def make_memory_interval(self, mass, error=None):
        if error is None:
            error = self.loading_interval
        out = self.search_mass(mass, error)
        return NeutralMassDatabase(out, operator.attrgetter("calculated_mass"))

    @property
    def lowest_mass(self):
        return self.session.query(func.min(self.model_type.calculated_mass)).filter(
            self.model_type.hypothesis_id == self.hypothesis_id)

    @property
    def highest_mass(self):
        return self.session.query(func.max(self.model_type.calculated_mass)).filter(
            self.model_type.hypothesis_id == self.hypothesis_id)

    def get_object_by_id(self, id):
        return self.session.query(self.model_type).get(id)

    def get_object_by_reference(self, reference):
        return self.session.query(self.model_type).get(reference.id)


class _ProteinIndex(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id

    def _get_by_id(self, id):
        return self.session.query(Protein).get(id)

    def _get_by_name(self, name):
        return self.session.query(Protein).filter(
            Protein.hypothesis_id == self.hypothesis_id,
            Protein.name == name).one()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_id(key)
        else:
            return self._get_by_name(key)


class _PeptideIndex(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id

    def _get_by_id(self, id):
        return self.session.query(InformedPeptide).get(id)

    def _get_by_sequence(self, modified_peptide_sequence, protein_id):
        return self.session.query(InformedPeptide).filter(
            InformedPeptide.hypothesis_id == self.hypothesis_id,
            InformedPeptide.modified_peptide_sequence == modified_peptide_sequence,
            InformedPeptide.protein_id == protein_id).one()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_id(key)
        else:
            return self._get_by_name(*key)


class GlycopeptideDiskBackedStructureDatabase(DiskBackedStructureDatabase):

    def _get_record_properties(self):
        table = self.model_type.__table__
        return [
            table.c.id, table.c.calculated_mass, table.c.glycopeptide_sequence,
            table.c.protein_id, table.c.start_position, table.c.end_position,
            table.c.hypothesis_id
        ]

    @property
    def structures(self):
        table = self.model_type.__table__
        return self.session.query(*self._get_record_properties()).filter(
            table.c.hypothesis_id == self.hypothesis_id).order_by(
            table.c.calculated_mass)

    def search_mass(self, mass, error_tolerance):
        table = self.model_type.__table__
        conn = self.session.connection()

        stmt = select(self._get_record_properties()).where(
            (table.c.hypothesis_id == self.hypothesis_id) & table.c.calculated_mass.between(
                mass - error_tolerance, mass + error_tolerance))
        return conn.execute(stmt).fetchall()


class MassIntervalNode(SpanningMixin):
    """Contains a NeutralMassDatabase object and provides an interval-like
    API. Intended for use with IntervalSet.

    Attributes
    ----------
    center : float
        The central point of mass for the interval
    end : float
        The upper-most mass in the wrapped collection
    group : NeutralMassDatabase
        The collection of massable objects wrapped
    growth : int
        The number of times the interval had been expanded
    size : int
        The number of items in :attr:`group`
    start : float
        The lower-most mass in the wrapped collection
    """
    def __init__(self, interval):
        self.wrap(interval)
        self.growth = 0

    def __hash__(self):
        return hash((self.start, self.center, self.end))

    def __eq__(self, other):
        return (self.start, self.center, self.end) == (other.start, other.center, other.end)

    def wrap(self, interval):
        """Updates the internal state of the interval to wrap

        Parameters
        ----------
        interval : NeutralMassDatabase
            The new collection to wrap.
        """
        self.group = interval
        self.start = interval.lowest_mass
        self.end = interval.highest_mass
        self.center = (self.start + self.end) / 2.
        self.size = len(self.group)

    def __repr__(self):
        return "MassIntervalNode(%0.4f, %0.4f)" % (self.start, self.end)

    def extend(self, new_data):
        """Add the components of `new_data` to `group` and update
        the interval's internal state

        Parameters
        ----------
        new_data : NeutralMassDatabase
            Iterable of massable objects
        """
        new = {x.id: x for x in (self.group)}
        new.update({x.id: x for x in (new_data)})
        self.wrap(NeutralMassDatabase(list(new.values()), self.group.mass_getter))
        self.growth += 1


class QueryIntervalBase(SpanningMixin):

    def __hash__(self):
        return hash((self.start, self.center, self.end))

    def __eq__(self, other):
        return (self.start, self.center, self.end) == (other.start, other.center, other.end)

    def __repr__(self):
        return "QueryInterval(%0.4f, %0.4f)" % (self.start, self.end)

    def extend(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.center = (self.start + self.end) / 2.


class PPMQueryInterval(QueryIntervalBase):

    def __init__(self, mass, error_tolerance=2e-5):
        self.center = mass
        self.start = mass - (mass * error_tolerance)
        self.end = mass + (mass * error_tolerance)


class FixedQueryInterval(QueryIntervalBase):

    def __init__(self, mass, width=3):
        self.center = mass
        self.start = mass - width
        self.end = mass + width


class IntervalSet(object):

    def __init__(self, intervals=None):
        if intervals is None:
            intervals = list()
        self.intervals = sorted(intervals, key=lambda x: x.center)
        self._total_count = None
        self.compute_total_count()

    def _invalidate(self):
        self._total_count = None

    @property
    def total_count(self):
        if self._total_count is None:
            self.compute_total_count()
        return self._total_count

    def compute_total_count(self):
        self._total_count = 0
        for interval in self:
            self._total_count += interval.size
        return self._total_count

    def __iter__(self):
        return iter(self.intervals)

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, i):
        return self.intervals[i]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.intervals)

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.intervals)

    def find_insertion_point(self, mass):
        lo = 0
        hi = len(self)
        if hi == 0:
            return 0, False
        while hi != lo:
            mid = (hi + lo) / 2
            x = self[mid]
            err = x.center - mass
            if abs(err) <= 1e-9:
                return mid, True
            elif (hi - lo) == 1:
                return mid, False
            elif err > 0:
                hi = mid
            else:
                lo = mid
        raise ValueError((hi, lo, err, len(self)))

    def extend_interval(self, target, expansion):
        self._invalidate()
        target.extend(expansion)

    def insert_interval(self, interval):
        center = interval.center
        if len(self) != 0:
            index, matched = self.find_insertion_point(center)
            index += 1
            if matched and self[index - 1].overlaps(interval):
                self.extend_interval(self[index - 1], interval)
                return
            if index == 1:
                if self[index - 1].center > center:
                    index -= 1
        else:
            index = 0
        self._insert_interval(index, interval)

    def _insert_interval(self, index, interval):
        self._invalidate()
        self.intervals.insert(index, interval)

    def find_interval(self, query):
        lo = 0
        hi = len(self)
        while hi != lo:
            mid = (hi + lo) / 2
            x = self[mid]
            err = x.center - query.center
            if err == 0 or x.contains_interval(query):
                return x
            elif (hi - 1) == lo:
                return x
            elif err > 0:
                hi = mid
            else:
                lo = mid

    def find(self, mass, ppm_error_tolerance):
        return self.find_interval(PPMQueryInterval(mass, ppm_error_tolerance))

    def remove_interval(self, center):
        self._invalidate()
        ix, match = self.find_insertion_point(center)
        assert match
        self.intervals.pop(ix)


class LRUIntervalSet(IntervalSet):

    def __init__(self, intervals=None, max_size=1000):
        super(LRUIntervalSet, self).__init__(intervals)
        self.max_size = max_size
        self.current_size = len(self)
        self.lru = LRUCache()
        for item in self:
            self.lru.add_node(item)

    def insert_interval(self, interval):
        if self.current_size == self.max_size:
            self.remove_lru_interval()
        super(LRUIntervalSet, self).insert_interval(interval)

    def _insert_interval(self, index, interval):
        super(LRUIntervalSet, self)._insert_interval(index, interval)
        self.lru.add_node(interval)
        self.current_size += 1

    def extend_interval(self, target, expansion):
        self.lru.remove_node(target)
        try:
            super(LRUIntervalSet, self).extend_interval(target, expansion)
            self.lru.add_node(target)
        except:
            self.lru.add_node(target)
            raise

    def find_interval(self, query):
        match = super(LRUIntervalSet, self).find_interval(query)
        if match is not None:
            try:
                self.lru.hit_node(match)
            except KeyError:
                self.lru.add_node(match)
        return match

    def remove_lru_interval(self):
        lru_interval = self.lru.get_least_recently_used()
        self.lru.remove_node(lru_interval)
        self.remove_interval(lru_interval.center)
        self.current_size -= 1
