import operator

try:
    from itertools import imap
except ImportError:
    imap = map

import logging

from ms_deisotope.peak_dependency_network.intervals import SpanningMixin
from glycan_profiling.serialize import (
    GlycanComposition, Glycopeptide, Peptide,
    func, Protein, GlycopeptideHypothesis, GlycanHypothesis,
    DatabaseBoundOperation)

from sqlalchemy import select, join

from .mass_collection import SearchableMassCollection, NeutralMassDatabase
from glycan_profiling.structure.lru import LRUCache
from glycan_profiling.structure.structure_loader import CachingGlycanCompositionParser
from .composition_network import CompositionGraph, n_glycan_distance


logger = logging.getLogger("glycresoft.database")

_empty_interval = NeutralMassDatabase([])


# The maximum total number of items across all loaded intervals that a database
# is allowed to hold in memory at once before it must start pruning intervals.
# This is to prevent the program from running out of memory because it loaded
# too many large intervals into memory while not exceeding the maximum number of
# intervals allowed. 200,000 is reasonable given the current memory consumption
# patterns.
DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT = 2e5

# The maximum number of mass intervals to hold in memory at once.
# The current algorithms used elsewhere order queries such that this
# number could even be set to 1 and not suffer runtime penalties becase
# overlapping intervals are merged and only consume a single slot, but
# this is retained at >1 for convenience.
DEFAULT_CACHE_SIZE = 2

# Controls the mass interval loaded from the database when a
# query misses the alreadly loaded intervals. Uses the assumption
# that if a mass is asked for, the next mass asked for will be
# close to it, so by pre-emptively loading more data now, less
# time will be spent searching the disk later. The larger this
# number, the more data that will be preloaded. If this number is
# too large, memory and disk I/O is wasted. If too small, then not
# only does it not help solve the pre-loading problem, but it also
# truncates the range that the in-memory interval search actually covers
# and may lead to missing matches. This means this number must chosen
# carefully given the non-constant error criterion we're using, PPM error.
#
# The default value of 1 is good for 10 ppm error intervals on structures
# up to 1e5 Da
DEFAULT_LOADING_INTERVAL = 1.


class DiskBackedStructureDatabaseBase(SearchableMassCollection, DatabaseBoundOperation):

    def __init__(self, connection, hypothesis_id, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT,
                 model_type=Glycopeptide):
        DatabaseBoundOperation.__init__(self, connection)
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
            self._original_connection, self.hypothesis_id, self.cache_size,
            self.loading_interval, self.threshold_cache_total_count,
            self.model_type)

    def _upkeep_memory_intervals(self):
        n = len(self._intervals)
        if n > 1:
            while (len(self._intervals) > 1 and
                   self._intervals.total_count >
                   self.threshold_cache_total_count):
                logger.info("Upkeep Memory Intervals %d %d", self._intervals.total_count, len(self._intervals))
                self._intervals.remove_lru_interval()

    def _get_record_properties(self):
        return self.fields

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(Peptide.__table__.c.hypothesis_id == self.hypothesis_id)

    def ignore_interval(self, interval):
        self._ignored_intervals.insert_interval(interval)

    def is_interval_ignored(self, interval):
        is_ignored = self._ignored_intervals.find_interval(interval)
        return is_ignored is not None and is_ignored.contains_interval(interval)

    def insert_interval(self, mass):
        tree = self.make_memory_interval(mass)
        # We won't insert this node.
        if len(tree) == 0:
            # Ignore seems to be not-working.
            # self.ignore_interval(FixedQueryInterval(mass))
            return tree
        node = MassIntervalNode(tree)
        nearest_interval = self._intervals.find_interval(node)
        # Should an insert be performed if the query just didn't overlap well
        # with the database?
        if nearest_interval is None:
            # No nearby interval, so we should insert
            self._intervals.insert_interval(node)
            return node.group
        elif not nearest_interval.contains_interval(node):
            # Nearby interval didn't contain this interval
            self._intervals.insert_interval(node)
            return node.group
        else:
            # Situation unclear.
            # Not worth inserting, so just return the group
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

    def on_memory_interval(self, mass, interval):
        if len(interval) > self.threshold_cache_total_count:
            logger.info("Interval Length %d for %f", len(interval), mass)

    def make_memory_interval(self, mass, error=None):
        if error is None:
            error = self.loading_interval
        out = self.search_mass(mass, error)
        self.on_memory_interval(mass, out)
        return NeutralMassDatabase(out, operator.attrgetter("calculated_mass"))

    @property
    def lowest_mass(self):
        return self.session.query(func.min(self.model_type.calculated_mass)).filter(
            self.model_type.hypothesis_id == self.hypothesis_id).first()

    @property
    def highest_mass(self):
        return self.session.query(func.max(self.model_type.calculated_mass)).filter(
            self.model_type.hypothesis_id == self.hypothesis_id).first()

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

    def __iter__(self):
        q = self.session.query(Protein).filter(Protein.hypothesis_id == self.hypothesis_id)
        return iter(q)


class _PeptideIndex(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id

    def _get_by_id(self, id):
        return self.session.query(Peptide).get(id)

    def _get_by_sequence(self, modified_peptide_sequence, protein_id):
        return self.session.query(Peptide).filter(
            Peptide.hypothesis_id == self.hypothesis_id,
            Peptide.modified_peptide_sequence == modified_peptide_sequence,
            Peptide.protein_id == protein_id).one()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_id(key)
        else:
            return self._get_by_name(*key)


class DeclarativeDiskBackedDatabase(DiskBackedStructureDatabaseBase):
    def __init__(self, connection, hypothesis_id, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        super(DeclarativeDiskBackedDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count, None)
        self._glycan_composition_network = None

    @property
    def glycan_composition_network(self):
        return self._glycan_composition_network

    @property
    def lowest_mass(self):
        q = select(
            [func.min(self.mass_field)]).select_from(
            self.selectable)
        q = self._limit_to_hypothesis(q)
        return self.session.execute(q).scalar()

    @property
    def highest_mass(self):
        q = select(
            [func.max(self.mass_field)]).select_from(
            self.selectable)
        q = self._limit_to_hypothesis(q)
        return self.session.execute(q).scalar()

    def _get_record_properties(self):
        return self.fields

    def _limit_to_hypothesis(self, selectable):
        raise NotImplementedError()

    @property
    def structures(self):
        stmt = self._limit_to_hypothesis(
            select(self._get_record_properties()).select_from(
                self.selectable)).order_by(
            self.mass_field)
        return imap(self._convert, self.session.execute(stmt))

    def __getitem__(self, i):
        stmt = self._limit_to_hypothesis(
            select(self._get_record_properties()).select_from(
                self.selectable)).order_by(
            self.mass_field).offset(i)
        return self._convert(self.session.execute(stmt).fetchone())

    def search_mass(self, mass, error_tolerance):
        conn = self.session.connection()
        stmt = self._limit_to_hypothesis(
            select(self._get_record_properties()).select_from(self.selectable)).where(
            self.mass_field.between(
                mass - error_tolerance, mass + error_tolerance))
        return conn.execute(stmt).fetchall()

    def get_object_by_id(self, id):
        return self._convert(self.session.execute(select(self._get_record_properties()).select_from(
            self.selectable).where(
            self.identity_field == id)).first())

    def get_object_by_reference(self, reference):
        return self._convert(self.get_object_by_id(reference.id))

    def __len__(self):
        stmt = select([func.count(self.identity_field)]).select_from(
            self.selectable)
        stmt = self._limit_to_hypothesis(stmt)
        return self.session.execute(stmt).scalar()


class GlycopeptideDiskBackedStructureDatabase(DeclarativeDiskBackedDatabase):
    selectable = join(Glycopeptide.__table__, Peptide.__table__)
    fields = [
        Glycopeptide.__table__.c.id,
        Glycopeptide.__table__.c.calculated_mass,
        Glycopeptide.__table__.c.glycopeptide_sequence,
        Glycopeptide.__table__.c.protein_id,
        Peptide.__table__.c.start_position,
        Peptide.__table__.c.end_position,
        Glycopeptide.__table__.c.hypothesis_id,
    ]
    mass_field = Glycopeptide.__table__.c.calculated_mass
    identity_field = Glycopeptide.__table__.c.id

    @property
    def hypothesis(self):
        return self.session.query(GlycopeptideHypothesis).get(self.hypothesis_id)

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(Glycopeptide.__table__.c.hypothesis_id == self.hypothesis_id)


class GlycopeptideOnlyDiskBackedStructureDatabase(DeclarativeDiskBackedDatabase):
    selectable = Glycopeptide.__table__
    fields = [
        Glycopeptide.__table__.c.id,
        Glycopeptide.__table__.c.calculated_mass,
        Glycopeptide.__table__.c.glycopeptide_sequence,
        Glycopeptide.__table__.c.peptide_id,
        Glycopeptide.__table__.c.protein_id,
        Glycopeptide.__table__.c.hypothesis_id,
    ]
    mass_field = Glycopeptide.__table__.c.calculated_mass
    identity_field = Glycopeptide.__table__.c.id

    @property
    def hypothesis(self):
        return self.session.query(GlycopeptideHypothesis).get(self.hypothesis_id)

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(Glycopeptide.__table__.c.hypothesis_id == self.hypothesis_id)


class GlycanCompositionDiskBackedStructureDatabase(DeclarativeDiskBackedDatabase):
    selectable = (GlycanComposition.__table__)
    fields = [
        GlycanComposition.__table__.c.id, GlycanComposition.__table__.c.calculated_mass,
        GlycanComposition.__table__.c.composition, GlycanComposition.__table__.c.formula,
        GlycanComposition.__table__.c.hypothesis_id
    ]
    mass_field = GlycanComposition.__table__.c.calculated_mass
    identity_field = GlycanComposition.__table__.c.id

    def __init__(self, connection, hypothesis_id, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        super(GlycanCompositionDiskBackedStructureDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count)
        self._convert_cache = CachingGlycanCompositionParser()

    def _convert(self, bundle):
        return self._convert_cache(bundle)

    @property
    def glycan_composition_network(self):
        if self._glycan_composition_network is None:
            self._glycan_composition_network = CompositionGraph(tuple(self.structures))
            self._glycan_composition_network.create_edges(1, n_glycan_distance)
        return self._glycan_composition_network

    @property
    def hypothesis(self):
        return self.session.query(GlycanHypothesis).get(self.hypothesis_id)

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(GlycanComposition.__table__.c.hypothesis_id == self.hypothesis_id)


class PeptideDiskBackedStructureDatabase(DeclarativeDiskBackedDatabase):
    selectable = Peptide.__table__
    fields = [
        Peptide.__table__.c.id,
        Peptide.__table__.c.calculated_mass,
        Peptide.__table__.c.modified_peptide_sequence,
        Peptide.__table__.c.protein_id,
        Peptide.__table__.c.start_position,
        Peptide.__table__.c.end_position,
        Peptide.__table__.c.hypothesis_id,
    ]
    mass_field = Peptide.__table__.c.calculated_mass
    identity_field = Peptide.__table__.c.id

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(Peptide.__table__.c.hypothesis_id == self.hypothesis_id)


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
        # if interval in self.intervals:
        #     raise ValueError("Duplicate Insertion")
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
        except Exception:
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
