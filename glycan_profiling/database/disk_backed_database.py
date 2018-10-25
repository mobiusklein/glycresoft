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
    DatabaseBoundOperation, GlycanClass, GlycanTypes,
    GlycanCompositionToClass)

from sqlalchemy import select, join

from .mass_collection import SearchableMassCollection, NeutralMassDatabase
from .intervals import (
    PPMQueryInterval, FixedQueryInterval, LRUIntervalSet,
    IntervalSet)
from glycan_profiling.structure.lru import LRUCache
from glycan_profiling.structure.structure_loader import (
    CachingGlycanCompositionParser, CachingGlycopeptideParser,
    GlycopeptideDatabaseRecord)
from .composition_network import CompositionGraph, n_glycan_distance


logger = logging.getLogger("glycresoft.database")

_empty_interval = NeutralMassDatabase([])


# The maximum total number of items across all loaded intervals that a database
# is allowed to hold in memory at once before it must start pruning intervals.
# This is to prevent the program from running out of memory because it loaded
# too many large intervals into memory while not exceeding the maximum number of
# intervals allowed.
DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT = 6e5

# The maximum number of mass intervals to hold in memory at once.
DEFAULT_CACHE_SIZE = 3

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

    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
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
        logger.debug("Calling insert_interval with mass %f", mass)
        node = self.make_memory_interval(mass)
        logger.debug("New Node: %r", node)
        # We won't insert this node if it is empty.
        if len(node.group) == 0:
            # Ignore seems to be not-working.
            # self.ignore_interval(FixedQueryInterval(mass))
            return node.group
        nearest_interval = self._intervals.find_interval(node)
        # Should an insert be performed if the query just didn't overlap well
        # with the database?
        if nearest_interval is None:
            # No nearby interval, so we should insert
            logger.debug("No nearby interval for %r", node)
            self._intervals.insert_interval(node)
            return node.group
        elif nearest_interval.overlaps(node):
            logger.debug("Nearest interval %r overlaps this %r", nearest_interval, node)
            nearest_interval = self._intervals.extend_interval(nearest_interval, node)
            return nearest_interval.group
        elif not nearest_interval.contains_interval(node):
            logger.debug("Nearest interval %r didn't contain this %r", nearest_interval, node)
            # Nearby interval didn't contain this interval
            self._intervals.insert_interval(node)
            return node.group
        else:
            # Situation unclear.
            # Not worth inserting, so just return the group
            logger.info("Unknown Condition Overlap %r / %r" % (node, nearest_interval))
            return nearest_interval.group

    def clear_cache(self):
        self._intervals.clear()

    def has_interval(self, mass, ppm_error_tolerance):
        q = PPMQueryInterval(mass, ppm_error_tolerance)
        match = self._intervals.find_interval(q)
        if match is not None:
            q2 = FixedQueryInterval(mass, self.loading_interval)
            logger.debug("Nearest interval %r", match)
            # We are completely contained in an existing interval, so just
            # use that one.
            if match.contains_interval(q):
                logger.debug("Query interval %r was completely contained in %r", q, match)
                return match.group
            # We overlap with an extending interval, so we should populate
            # the new one and merge them.
            elif match.overlaps(q2):
                q3 = q2.difference(match).scale(1.05)
                logger.debug("Query interval partially overlapped, creating disjoint interval %r", q3)
                match = self._intervals.extend_interval(
                    match,
                    # self.make_memory_interval(mass),
                    self.make_memory_interval_from_mass_interval(q3.start, q3.end)
                )
                return match.group
            # We might need to insert a new interval
            else:
                logger.debug("Query interval %r did not overlap with %r", q, match)
                if self.is_interval_ignored(q):
                    return match.group
                else:
                    return self.insert_interval(mass)
        else:
            logger.debug("No existing interval contained %r", q)
            is_ignored = self._ignored_intervals.find_interval(q)
            if is_ignored is not None and is_ignored.contains_interval(q):
                return _empty_interval
            else:
                return self.insert_interval(mass)

    def search_mass_ppm(self, mass, error_tolerance):
        self._upkeep_memory_intervals()
        return self.has_interval(mass, error_tolerance).search_mass_ppm(mass, error_tolerance)

    def search_mass(self, mass, error_tolerance):
        return self._search_mass_interval(mass - error_tolerance, mass + error_tolerance)

    def _search_mass_interval(self, start, end):
        model = self.model_type
        return self.session.query(self.model_type).filter(
            model.hypothesis_id == self.hypothesis_id,
            model.calculated_mass.between(
                start, end)).all()

    def on_memory_interval(self, mass, interval):
        if len(interval) > self.threshold_cache_total_count:
            logger.info("Interval Length %d for %f", len(interval), mass)

    def make_memory_interval(self, mass, error=None):
        if error is None:
            error = self.loading_interval
        node = self.make_memory_interval_from_mass_interval(mass - error, mass + error)
        return node

    def make_memory_interval_from_mass_interval(self, start, end):
        logger.debug("Querying the database for masses between %f and %f", start, end)
        out = self._search_mass_interval(start, end)
        logger.debug("Retrieved %d records", len(out))
        self.on_memory_interval((start + end) / 2.0, out)
        mass_db = self._prepare_interval(out)
        # bind the bounds of the returned dataset to the bounds of the query
        node = MassIntervalNode(mass_db)
        node.start = start
        node.end = end
        return node

    def _prepare_interval(self, query_results):
        return NeutralMassDatabase(query_results, operator.attrgetter("calculated_mass"))

    @property
    def lowest_mass(self):
        return self.session.query(func.min(self.model_type.calculated_mass)).filter(
            self.model_type.hypothesis_id == self.hypothesis_id).first()

    @property
    def highest_mass(self):
        return self.session.query(func.max(self.model_type.calculated_mass)).filter(
            self.model_type.hypothesis_id == self.hypothesis_id).first()

    def get_record(self, id):
        return self.session.query(self.model_type).get(id)

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


class _GlycopeptideSequenceCache(object):
    def __init__(self, session, cache_size=1e6):
        self.session = session
        self.cache_size = int(cache_size)
        self.cache = dict()
        self.lru = LRUCache()

    def _fetch_sequence_by_id(self, id):
        return self.session.query(Glycopeptide.glycopeptide_sequence).filter(
            Glycopeptide.id == id).scalar()

    def _check_cache_valid(self):
        lru = self.lru
        while len(self.cache) > self.cache_size:
            self.churn += 1
            key = lru.get_least_recently_used()
            lru.remove_node(key)
            self.cache.pop(key)

    def _populate_cache(self, id):
        self._check_cache_valid()
        value = self._fetch_sequence_by_id(id)
        self.cache[id] = value
        self.lru.add_node(id)
        return value

    def _get_sequence_by_id(self, id):
        try:
            return self.cache[id]
        except KeyError:
            return self._populate_cache(id)

    def __getitem__(self, key):
        return self._get_sequence_by_id(id)

    def _fetch_batch(self, ids, chunk_size=500):
        n = len(ids)
        i = 0
        acc = []
        while i < n:
            batch = ids[i:(i + chunk_size)]
            i += chunk_size
            seqs = self.session.query(Glycopeptide.glycopeptide_sequence).filter(
                Glycopeptide.id.in_(batch)).all()
            acc.extend(s[0] for s in seqs)
        return acc

    def _process_batch(self, ids, chunk_size=500):
        result = dict()
        missing = []
        for i in ids:
            try:
                result[i] = self.cache[i]
            except KeyError:
                missing.append(i)
        fetched = self._fetch_batch(missing, chunk_size)
        for i, v in zip(missing, fetched):
            self.cache[i] = v
            result[i] = v
        return result

    def batch(self, ids, chunk_size=500):
        self._check_cache_valid()
        return self._process_batch(ids, chunk_size)


class _GlycopeptideBatchManager(object):
    def __init__(self, cache):
        self.cache = cache
        self.batch = {}

    def mark_hit(self, match):
        # print("Marking ", match)
        self.batch[match.id] = match
        return match

    def process_batch(self):
        ids = [m for m, v in self.batch.items() if v.glycopeptide_sequence is None]
        seqs = self.cache.batch(ids)
        for k, v in seqs.items():
            self.batch[k].glycopeptide_sequence = v

    def clear(self):
        self.batch.clear()


class DeclarativeDiskBackedDatabase(DiskBackedStructureDatabaseBase):
    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
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
        return self._search_mass_interval(mass - error_tolerance, mass + error_tolerance)

    def _search_mass_interval(self, start, end):
        conn = self.session.connection()
        stmt = self._limit_to_hypothesis(
            select(self._get_record_properties()).select_from(self.selectable)).where(
            self.mass_field.between(
                start, end))
        return conn.execute(stmt).fetchall()

    def get_record(self, id):
        record = self.session.execute(select(self._get_record_properties()).select_from(
            self.selectable).where(
            self.identity_field == id)).first()
        return record

    def get_all_records(self):
        records = self.session.execute(
            select(
                self._get_record_properties()).select_from(self.selectable))
        return list(records)

    def get_object_by_id(self, id):
        return self._convert(self._get_record(id))

    def get_object_by_reference(self, reference):
        return self._convert(self.get_object_by_id(reference.id))

    def __len__(self):
        try:
            return self.hypothesis.parameters['database_size']
        except KeyError:
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
        Peptide.__table__.c.calculated_mass.label("peptide_mass"),
        Glycopeptide.__table__.c.hypothesis_id,
    ]
    mass_field = Glycopeptide.__table__.c.calculated_mass
    identity_field = Glycopeptide.__table__.c.id

    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        super(GlycopeptideDiskBackedStructureDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count)
        self._convert_cache = CachingGlycopeptideParser()

    def _convert(self, bundle):
        inst = self._convert_cache.parse(bundle)
        inst.hypothesis_id = self.hypothesis_id
        return inst

    @property
    def hypothesis(self):
        return self.session.query(GlycopeptideHypothesis).get(self.hypothesis_id)

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(Glycopeptide.__table__.c.hypothesis_id == self.hypothesis_id)


class LazyLoadingGlycopeptideDiskBackedStructureDatabase(GlycopeptideDiskBackedStructureDatabase):
    fields = [
        Glycopeptide.__table__.c.id,
        Glycopeptide.__table__.c.calculated_mass,
        # Glycopeptide.__table__.c.glycopeptide_sequence,
        Glycopeptide.__table__.c.protein_id,
        Peptide.__table__.c.start_position,
        Peptide.__table__.c.end_position,
        Peptide.__table__.c.calculated_mass.label("peptide_mass"),
        Glycopeptide.__table__.c.hypothesis_id,
    ]

    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        super(LazyLoadingGlycopeptideDiskBackedStructureDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count)
        self._batch_manager = _GlycopeptideBatchManager(
            _GlycopeptideSequenceCache(self.session))

    def _prepare_interval(self, query_results):
        return NeutralMassDatabase(
            [GlycopeptideDatabaseRecord(
                q.id,
                q.calculated_mass,
                None,
                q.protein_id,
                q.start_position,
                q.end_position,
                q.peptide_mass,
                q.hypothesis_id,) for q in query_results],
            operator.attrgetter("calculated_mass"))

    def mark_hit(self, match):
        return self._batch_manager.mark_hit(match)

    def mark_batch(self):
        self._batch_manager.process_batch()
        self._batch_manager.clear()


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

    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        super(GlycanCompositionDiskBackedStructureDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count)
        self._convert_cache = CachingGlycanCompositionParser()

    def _convert(self, bundle):
        inst = self._convert_cache(bundle)
        inst.hypothesis_id = self.hypothesis_id
        return inst

    @property
    def glycan_composition_network(self):
        if self._glycan_composition_network is None:
            self._glycan_composition_network = CompositionGraph(tuple(self.structures))
            self._glycan_composition_network.create_edges(1, n_glycan_distance)
        return self._glycan_composition_network

    @property
    def hypothesis(self):
        return self.session.query(GlycanHypothesis).get(self.hypothesis_id)

    def glycan_compositions_of_type(self, glycan_type):
        try:
            glycan_type = glycan_type.name
        except AttributeError:
            glycan_type = str(glycan_type)
        if glycan_type in GlycanTypes:
            glycan_type = GlycanTypes[glycan_type]
        stmt = self._limit_to_hypothesis(
            select(self._get_record_properties()).select_from(
                self.selectable.join(GlycanCompositionToClass).join(GlycanClass)).where(
                GlycanClass.__table__.c.name == glycan_type)).order_by(
            self.mass_field)
        return imap(self._convert, self.session.execute(stmt))

    def glycan_composition_network_from(self, query=None):
        if query is None:
            query = self.structures
        compositions = tuple(query)
        graph = CompositionGraph(compositions)
        graph.create_edges(1)
        return graph

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
        try:
            self.start = interval.lowest_mass
            self.end = interval.highest_mass
        except IndexError:
            self.start = 0
            self.end = 0
        self.center = (self.start + self.end) / 2.
        self.size = len(self.group)

    def __repr__(self):
        return "MassIntervalNode(%0.4f, %0.4f, %r)" % (
            self.start, self.end, len(self.group) if self.group is not None else 0)

    def extend(self, new_data):
        """Add the components of `new_data` to `group` and update
        the interval's internal state

        Parameters
        ----------
        new_data : NeutralMassDatabase
            Iterable of massable objects
        """
        start = self.start
        end = self.end

        new = {x.id: x for x in (self.group)}
        new.update({x.id: x for x in (new_data)})
        self.wrap(NeutralMassDatabase(list(new.values()), self.group.mass_getter))
        self.growth += 1

        # max will deal with 0s correctly
        self.end = max(end, self.end)
        # min will prefer 0 and not behave as expected
        if start != 0 and self.start != 0:
            self.start = min(start, self.start)
        elif start != 0:
            # self.start is 0
            self.start = start
        # otherwise they're both 0 and we are out of luck

    def __iter__(self):
        return iter(self.group)

    def __getitem__(self, i):
        return self.group[i]

    def __len__(self):
        return len(self.group)

    def search_mass(self, *args, **kwargs):
        """A proxy for :meth:`NeutralMassDatabase.search_mass`
        """
        return self.group.search_mass(*args, **kwargs)

    def search_mass_ppm(self, *args, **kwargs):
        return self.group.search_mass_ppm(*args, **kwargs)
