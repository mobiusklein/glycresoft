import operator

from collections import defaultdict
try:
    from itertools import imap
except ImportError:
    imap = map

import logging


from sqlalchemy import select, join

from glycresoft.serialize import (
    GlycanComposition, Glycopeptide, Peptide,
    func, GlycopeptideHypothesis, GlycanHypothesis,
    DatabaseBoundOperation, GlycanClass, GlycanTypes,
    GlycanCompositionToClass, ProteinSite, Protein, GlycanCombination)

from .mass_collection import SearchableMassCollection, NeutralMassDatabase
from .intervals import (
    PPMQueryInterval, FixedQueryInterval, LRUIntervalSet,
    IntervalSet, MassIntervalNode)
from .index import (ProteinIndex, PeptideIndex, )
from glycresoft.structure.structure_loader import (
    CachingGlycanCompositionParser, CachingGlycopeptideParser,
    CachingPeptideParser)
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
    """A disk-backed structure database that implements the :class:`~.SearchableMassCollection`
    interface. It includes an in-memory caching system to keep recently used mass intervals in memory
    and optimistically fetches nearby masses around queried masses.

    Attributes
    ----------
    hypothesis_id : int
        The database id number of the hypothesis
    loading_interval : float
        The size of the region around a queried mass (in Daltons) to load when
        optimistically fetching records from the disk.
    cache_size : int
        The maximum number of intervals in-memory mass intervals to cache
    intervals : :class:`~.LRUIntervalSet`
        An LRU caching interval collection over masses read from the disk store
    model_type : type
        The ORM type to query against. It must provide an attribute called ``calculated_mass``
    peptides : :class:`~.PeptideIndex`
        An index for looking up :class:`~.Peptide` ORM objects.
    proteins : :class:`~.ProteinIndex`
        An index for looking up :class:`~.Protein` ORM objects.
    threshold_cache_total_count : int
        The total number of records to retain in the in-memory cache before pruning entries,
        independent of the number of loaded intervals
    """

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
        self.proteins = ProteinIndex(self.session, self.hypothesis_id)
        self.peptides = PeptideIndex(self.session, self.hypothesis_id)

    @property
    def cache_size(self):
        return self._intervals.max_size

    @cache_size.setter
    def cache_size(self, value):
        self._intervals.max_size = value

    @property
    def intervals(self):
        return self._intervals

    def reset(self, **kwargs):
        self.intervals.clear()

    def __reduce__(self):
        return self.__class__, (
            self._original_connection, self.hypothesis_id, self.cache_size,
            self.loading_interval, self.threshold_cache_total_count,
            self.model_type)

    def _make_loading_interval(self, mass, error_tolerance=1e-5):
        width = mass * error_tolerance
        if width > self.loading_interval:
            return FixedQueryInterval(mass, width * 2)
        else:
            return FixedQueryInterval(mass, self.loading_interval)

    def _upkeep_memory_intervals(self):
        """Perform routine maintainence of the interval cache, ensuring its size does not
        exceed the upper limit
        """
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

    def insert_interval(self, mass, ppm_error_tolerance=1e-5):
        logger.debug("Calling insert_interval with mass %f", mass)
        node = self.make_memory_interval(mass, ppm_error_tolerance)
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
            logger.info("Unknown Condition Overlap %r / %r", node, nearest_interval)
            return nearest_interval.group

    def clear_cache(self):
        self._intervals.clear()

    def has_interval(self, mass, ppm_error_tolerance):
        q = PPMQueryInterval(mass, ppm_error_tolerance)
        match = self._intervals.find_interval(q)
        if match is not None:
            q2 = self._make_loading_interval(mass, ppm_error_tolerance)
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
                    self.make_memory_interval_from_mass_interval(q3.start, q3.end)
                )
                return match.group
            # We might need to insert a new interval
            else:
                logger.debug("Query interval %r did not overlap with %r", q, match)
                if self.is_interval_ignored(q):
                    return match.group
                else:
                    return self.insert_interval(mass, ppm_error_tolerance)
        else:
            logger.debug("No existing interval contained %r", q)
            is_ignored = self._ignored_intervals.find_interval(q)
            if is_ignored is not None and is_ignored.contains_interval(q):
                return _empty_interval
            else:
                return self.insert_interval(mass, ppm_error_tolerance)

    def search_mass_ppm(self, mass, error_tolerance):
        self._upkeep_memory_intervals()
        return self.has_interval(mass, error_tolerance).search_mass_ppm(mass, error_tolerance)

    def search_mass(self, mass, error_tolerance):  # pylint: disable=signature-differs
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

    def make_memory_interval(self, mass, error_tolerance=None):
        interval = self._make_loading_interval(mass, error_tolerance)
        node = self.make_memory_interval_from_mass_interval(interval.start, interval.end)
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

    def search_between(self, lower, upper):
        return self.make_memory_interval_from_mass_interval(lower, upper)

    def valid_glycan_set(self):
        glycan_query = self.query(
            GlycanCombination).join(
            GlycanCombination.components).join(
            GlycanComposition.structure_classes).group_by(GlycanCombination.id)

        glycan_compositions = [
            c.convert() for c in glycan_query.all()]

        return set(glycan_compositions)


class DeclarativeDiskBackedDatabase(DiskBackedStructureDatabaseBase):
    """A base class for creating :class:`DiskBackedStructureDatabaseBase` that map against
    ad hoc SQL queries rather than ORM classes

    Attributes
    ----------
    identity_field: :class:`sqlalchemy.sql.Selectable`
        The database column from which the entity identity should be determined
    mass_field: :class:`sqlalchemy.Column`
        The column to read the mass from
    fields: list of :class:`sqlalchemy.ColumnElement`
        The columns to extract from the database to compose records

    """

    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        super(DeclarativeDiskBackedDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count, None)
        self._glycan_composition_network = None

    def __reduce__(self):
        return self.__class__, (
            self._original_connection, self.hypothesis_id, self.cache_size,
            self.loading_interval, self.threshold_cache_total_count)

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

    def __iter__(self):
        stmt = self._limit_to_hypothesis(
            select(self._get_record_properties()).select_from(
                self.selectable)).order_by(
            self.mass_field)
        cursor = self.session.execute(stmt)
        converter = self._convert
        while True:
            block_size = 2 ** 16
            block = cursor.fetchmany(block_size)
            if not block:
                break
            for row in block:
                yield converter(row)

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
        return self._convert(self.get_record(id))

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

    def make_key_maker(self):
        return GlycopeptideIdKeyMaker(self, self.hypothesis_id)


class GlycopeptideIdKeyMaker(object):
    def __init__(self, database, hypothesis_id):
        self.database = database
        self.hypothesis_id = hypothesis_id
        self.lookup_map = defaultdict(list)
        for gc in self.database.valid_glycan_set():
            self.lookup_map[str(gc)].append(gc.id)

    def make_id_controlled_structures(self, structure, references):
        glycan_ids = self.lookup_map[structure.glycan_composition]
        result = []
        for glycan_id in glycan_ids:
            for ref in references:
                ref_rec = self.database.query(Glycopeptide).get(ref.id)
                alts = self.database.query(Glycopeptide).filter(
                    Glycopeptide.peptide_id == ref_rec.peptide_id,
                    Glycopeptide.glycan_combination_id == glycan_id).all()
                for alt in alts:
                    result.append(alt.convert())
        return result

    def __call__(self, structure, references):
        return self.make_id_controlled_structures(structure, references)


class InMemoryPeptideStructureDatabase(NeutralMassDatabase):
    def __init__(self, records, source_database=None, sort=True):
        super(InMemoryPeptideStructureDatabase, self).__init__(records, sort=sort)
        self.source_database = source_database

    @property
    def hypothesis(self):
        return self.source_database.hypothesis

    @property
    def hypothesis_id(self):
        return self.source_database.hypothesis_id

    @property
    def session(self):
        return self.source_database.session

    def query(self, *args, **kwargs):
        return self.source_database.query(*args, **kwargs)

    @property
    def peptides(self):
        return self.source_database.peptides

    @property
    def proteins(self):
        return self.source_database.proteins


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
        Peptide.__table__.c.n_glycosylation_sites,
        Peptide.__table__.c.o_glycosylation_sites,
        Peptide.__table__.c.gagylation_sites,
    ]
    mass_field = Peptide.__table__.c.calculated_mass
    identity_field = Peptide.__table__.c.id

    def __init__(self, connection, hypothesis_id=1, cache_size=DEFAULT_CACHE_SIZE,
                 loading_interval=DEFAULT_LOADING_INTERVAL,
                 threshold_cache_total_count=int(DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT / 5)):
        super(PeptideDiskBackedStructureDatabase, self).__init__(
            connection, hypothesis_id, cache_size, loading_interval,
            threshold_cache_total_count)
        self._convert_cache = CachingPeptideParser()
        self.peptides = PeptideIndex(self.session, self.hypothesis_id)
        self.proteins = ProteinIndex(self.session, self.hypothesis_id)

    # This should exist, but it conflicts with other conversion mechanisms
    # def _convert(self, bundle):
    #     inst = self._convert_cache.parse(bundle)
    #     inst.hypothesis_id = self.hypothesis_id
    #     return inst

    def _limit_to_hypothesis(self, selectable):
        return selectable.where(Peptide.__table__.c.hypothesis_id == self.hypothesis_id)

    @property
    def hypothesis(self):
        return self.session.query(GlycopeptideHypothesis).get(self.hypothesis_id)

    def spanning_n_glycosylation_site(self):
        q = select(self.fields).select_from(
            self.selectable.join(Protein.__table__).join(ProteinSite.__table__)).where(
                Peptide.spans(ProteinSite.location) & (ProteinSite.name == ProteinSite.N_GLYCOSYLATION)
                & (Protein.hypothesis_id == self.hypothesis_id)).order_by(
            self.mass_field).group_by(Peptide.id)
        return q

    def having_glycosylation_site(self):
        pickle_size = func.length(
            Peptide.__table__.c.o_glycosylation_sites) + \
            func.length(
                Peptide.__table__.c.n_glycosylation_sites)
        min_pickle_size = self.session.query(func.min(pickle_size)).scalar()
        if min_pickle_size is None:
            min_pickle_size = 0
        q = select(self.fields).where(
            (pickle_size > min_pickle_size) &
            (Peptide.__table__.c.hypothesis_id == self.hypothesis_id)
        ).order_by(self.mass_field)
        return q

    def has_protein_sites(self):
        has_sites = self.query(ProteinSite).join(ProteinSite.protein).filter(
            Protein.hypothesis_id == self.hypothesis_id,
            ProteinSite.name == ProteinSite.N_GLYCOSYLATION).first()
        return has_sites is not None

