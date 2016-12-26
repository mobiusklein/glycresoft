import itertools
from uuid import uuid4
from collections import defaultdict, Counter
from multiprocessing import Process, Queue, Event
from itertools import product

from glypy import Composition
from glypy.composition import formula
from glypy.composition.glycan_composition import FrozenGlycanComposition

from glycan_profiling.serialize import DatabaseBoundOperation, func
from glycan_profiling.serialize.hypothesis import GlycopeptideHypothesis
from glycan_profiling.serialize.hypothesis.peptide import Glycopeptide, Peptide, Protein
from glycan_profiling.serialize.hypothesis.glycan import (
    GlycanCombination, GlycanClass, GlycanComposition,
    GlycanTypes, GlycanCombinationGlycanComposition)
from glycan_profiling.task import TaskBase

from glycan_profiling.database.builder.glycan import glycan_combinator
from glycan_profiling.database.builder.base import HypothesisSerializerBase

from glycopeptidepy.structure.sequence import (
    _n_glycosylation, _o_glycosylation, _gag_linker_glycosylation)


def slurp(session, model, ids, flatten=True):
    if flatten:
        ids = [j for i in ids for j in i]
    total = len(ids)
    last = 0
    step = 100
    results = []
    while last < total:
        results.extend(session.query(model).filter(
            model.id.in_(ids[last:last + step])))
        last += step
    return results


class GlycopeptideHypothesisSerializerBase(DatabaseBoundOperation, HypothesisSerializerBase):
    """Common machinery for Glycopeptide Hypothesis construction.

    Attributes
    ----------
    uuid : str
        The uuid of the hypothesis to be constructed
    """
    def __init__(self, database_connection, hypothesis_name=None, glycan_hypothesis_id=None):
        DatabaseBoundOperation.__init__(self, database_connection)
        self._hypothesis_name = hypothesis_name
        self._hypothesis_id = None
        self._hypothesis = None
        self._glycan_hypothesis_id = glycan_hypothesis_id
        self.uuid = str(uuid4().hex)

    def _construct_hypothesis(self):
        if self._hypothesis_name is None or self._hypothesis_name.strip() == "":
            self._hypothesis_name = self._make_name()

        if self.glycan_hypothesis_id is None:
            raise ValueError("glycan_hypothesis_id must not be None")
        self._hypothesis = GlycopeptideHypothesis(
            name=self._hypothesis_name, glycan_hypothesis_id=self._glycan_hypothesis_id,
            uuid=self.uuid)
        self.session.add(self._hypothesis)
        self.session.commit()

        self._hypothesis_id = self._hypothesis.id
        self._hypothesis_name = self._hypothesis.name
        self._glycan_hypothesis_id = self._hypothesis.glycan_hypothesis_id

    def _make_name(self):
        return "GlycopeptideHypothesis-" + self.uuid

    @property
    def glycan_hypothesis_id(self):
        if self._glycan_hypothesis_id is None:
            self._construct_hypothesis()
        return self._glycan_hypothesis_id

    def combinate_glycans(self, n):
        combinator = glycan_combinator.GlycanCombinationSerializer(
            self.engine, self.glycan_hypothesis_id,
            self.hypothesis_id, n)
        combinator.run()

    def _count_produced_glycopeptides(self):
        count = self.query(
            func.count(Glycopeptide.id)).filter(
            Glycopeptide.hypothesis_id == self.hypothesis_id).scalar()
        self.log("Generated %d glycopeptides" % count)

    def _sql_analyze_database(self):
        self.log("Analyzing Indices")
        self._analyze_database()
        if self.is_sqlite():
            self._sqlite_reload_analysis_plan()
        self.log("Done Analyzing Indices")


class GlycopeptideHypothesisDestroyer(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection, hypothesis_id):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.hypothesis_id = hypothesis_id

    def delete_glycopeptides(self):
        self.log("Delete Glycopeptides")
        self.session.query(Glycopeptide).filter(
            Glycopeptide.hypothesis_id == self.hypothesis_id).delete(
            synchronize_session=False)
        self.session.commit()

    def delete_peptides(self):
        self.log("Delete Peptides")
        q = self.session.query(Protein.id).filter(Protein.hypothesis_id == self.hypothesis_id)
        for protein_id, in q:
            self.session.query(Peptide).filter(
                Peptide.protein_id == protein_id).delete(
                synchronize_session=False)
            self.session.commit()

    def delete_protein(self):
        self.log("Delete Protein")
        self.session.query(Protein).filter(Protein.hypothesis_id == self.hypothesis_id).delete(
            synchronize_session=False)
        self.session.commit()

    def delete_hypothesis(self):
        self.log("Delete Hypothesis")
        self.session.query(GlycopeptideHypothesis).filter(
            GlycopeptideHypothesis.id == self.hypothesis_id).delete()
        self.session.commit()

    def run(self):
        self.delete_glycopeptides()
        self.delete_peptides()
        self.delete_protein()
        self.delete_hypothesis()
        self.session.commit()


def distinct_glycan_classes(session, hypothesis_id):
    structure_classes = session.query(GlycanClass.name.distinct()).join(
        GlycanComposition.structure_classes).join(
        GlycanCombinationGlycanComposition).join(
        GlycanCombination).filter(
        GlycanCombination.hypothesis_id == hypothesis_id).all()
    return [sc[0] for sc in structure_classes]


def composition_to_structure_class_map(session, glycan_hypothesis_id):
    mapping = defaultdict(list)
    id_to_class_iterator = session.query(GlycanComposition.id, GlycanClass.name).join(
        GlycanComposition.structure_classes).filter(
        GlycanComposition.hypothesis_id == glycan_hypothesis_id).all()
    for gc_id, sc_name in id_to_class_iterator:
        mapping[gc_id].append(sc_name)
    return mapping


def combination_structure_class_map(session, hypothesis_id, composition_class_map):
    mapping = defaultdict(list)
    iterator = session.query(
        GlycanCombinationGlycanComposition).join(GlycanCombination).filter(
        GlycanCombination.hypothesis_id == hypothesis_id).order_by(GlycanCombination.id)
    for glycan_id, combination_id, count in iterator:
        listing = mapping[combination_id]
        for i in range(count):
            listing.append(composition_class_map[glycan_id])
    return mapping


class GlycanCombinationPartitionTable(TaskBase):
    def __init__(self, session, glycan_combinations, glycan_classes, hypothesis):
        self.session = session
        self.tables = defaultdict(lambda: defaultdict(list))
        self.hypothesis_id = hypothesis.id
        self.glycan_hypothesis_id = hypothesis.glycan_hypothesis_id
        self.glycan_classes = glycan_classes
        self.build_table(glycan_combinations)

    def build_table(self, glycan_combinations):
        composition_class_map = composition_to_structure_class_map(
            self.session, self.glycan_hypothesis_id)
        combination_class_map = combination_structure_class_map(
            self.session, self.hypothesis_id, composition_class_map)

        for entry in glycan_combinations:
            size_table = self.tables[entry.count]
            component_classes = combination_class_map[entry.id]
            class_assignment_generator = product(*component_classes)
            for classes in class_assignment_generator:
                counts = Counter(c for c in classes)
                key = tuple(counts[c] for c in self.glycan_classes)
                class_table = size_table[key]
                class_table.append(entry)

    def build_key(self, mapping):
        return tuple(mapping.get(c, 0) for c in self.glycan_classes)

    def get_entries(self, size, mapping):
        key = self.build_key(mapping)
        return self.tables[size][key]

    def __getitem__(self, key):
        size, mapping = key
        return self.get_entries(size, mapping)


def limiting_combinations(iterable, n, limit=100):
    i = 0
    for result in itertools.combinations(iterable, n):
        i += 1
        yield result
        if i > limit:
            break


class GlycanCombinationRecord(object):
    __slots__ = ['id', 'calculated_mass', 'formula', 'count', 'glycan_composition_string']

    def __init__(self, combination):
        self.id = combination.id
        self.calculated_mass = combination.calculated_mass
        self.formula = combination.formula
        self.count = combination.count
        self.glycan_composition_string = combination.composition

    def convert(self):
        gc = FrozenGlycanComposition.parse(self.glycan_composition_string)
        gc.id = self.id
        gc.count = self.count
        return gc

    def __repr__(self):
        return "GlycanCombinationRecord(%d, %s)" % (
            self.id, self.glycan_composition_string)


class PeptideGlycosylator(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id
        self.hypothesis = self.session.query(GlycopeptideHypothesis).get(hypothesis_id)
        glycan_combinations = self.session.query(
            GlycanCombination).filter(
            GlycanCombination.hypothesis_id == hypothesis_id).all()
        glycan_combinations = [GlycanCombinationRecord(gc) for gc in glycan_combinations]
        self.build_size_table(glycan_combinations)

    def build_size_table(self, glycan_combinations):
        self.glycan_combination_partitions = GlycanCombinationPartitionTable(
            self.session, glycan_combinations, distinct_glycan_classes(
                self.session, self.hypothesis_id), self.hypothesis)

    def handle_peptide(self, peptide):
        water = Composition("H2O")
        peptide_composition = Composition(str(peptide.formula))
        obj = peptide.convert()

        # Handle N-linked glycosylation sites

        n_glycosylation_unoccupied_sites = set(peptide.n_glycosylation_sites)
        for site in list(n_glycosylation_unoccupied_sites):
            if obj[site][1]:
                n_glycosylation_unoccupied_sites.remove(site)
        for i in range(len(n_glycosylation_unoccupied_sites)):
            i += 1
            for gc in self.glycan_combination_partitions[i, {GlycanTypes.n_glycan: i}]:
                total_mass = peptide.calculated_mass + gc.calculated_mass - (gc.count * water.mass)
                formula_string = formula(peptide_composition + Composition(str(gc.formula)) - (water * gc.count))

                for site_set in limiting_combinations(n_glycosylation_unoccupied_sites, i):
                    sequence = peptide.convert()
                    for site in site_set:
                        sequence.add_modification(site, _n_glycosylation.name)
                    sequence.glycan = gc.convert()

                    glycopeptide_sequence = str(sequence)

                    glycopeptide = Glycopeptide(
                        calculated_mass=total_mass,
                        formula=formula_string,
                        glycopeptide_sequence=glycopeptide_sequence,
                        peptide_id=peptide.id,
                        protein_id=peptide.protein_id,
                        hypothesis_id=peptide.hypothesis_id,
                        glycan_combination_id=gc.id)
                    yield glycopeptide

        # Handle O-linked glycosylation sites
        o_glycosylation_unoccupied_sites = set(peptide.o_glycosylation_sites)
        for site in list(o_glycosylation_unoccupied_sites):
            if obj[site][1]:
                o_glycosylation_unoccupied_sites.remove(site)

        for i in range(len(o_glycosylation_unoccupied_sites)):
            i += 1
            for gc in self.glycan_combination_partitions[i, {GlycanTypes.o_glycan: i}]:
                total_mass = peptide.calculated_mass + gc.calculated_mass - (gc.count * water.mass)
                formula_string = formula(peptide_composition + Composition(str(gc.formula)) - (water * gc.count))

                for site_set in limiting_combinations(o_glycosylation_unoccupied_sites, i):
                    sequence = peptide.convert()
                    for site in site_set:
                        sequence.add_modification(site, _o_glycosylation.name)
                    sequence.glycan = gc.convert()

                    glycopeptide_sequence = str(sequence)

                    glycopeptide = Glycopeptide(
                        calculated_mass=total_mass,
                        formula=formula_string,
                        glycopeptide_sequence=glycopeptide_sequence,
                        peptide_id=peptide.id,
                        protein_id=peptide.protein_id,
                        hypothesis_id=peptide.hypothesis_id,
                        glycan_combination_id=gc.id)
                    yield glycopeptide

        # Handle GAG glycosylation sites
        gag_unoccupied_sites = set(peptide.gagylation_sites)
        for site in list(gag_unoccupied_sites):
            if obj[site][1]:
                gag_unoccupied_sites.remove(site)
        for i in range(len(gag_unoccupied_sites)):
            i += 1
            for gc in self.glycan_combination_partitions[i, {GlycanTypes.gag_linker: i}]:
                total_mass = peptide.calculated_mass + gc.calculated_mass - (gc.count * water.mass)
                formula_string = formula(peptide_composition + Composition(str(gc.formula)) - (water * gc.count))
                for site_set in limiting_combinations(gag_unoccupied_sites, i):
                    sequence = peptide.convert()
                    for site in site_set:
                        sequence.add_modification(site, _gag_linker_glycosylation.name)
                    sequence.glycan = gc.convert()

                    glycopeptide_sequence = str(sequence)

                    glycopeptide = Glycopeptide(
                        calculated_mass=total_mass,
                        formula=formula_string,
                        glycopeptide_sequence=glycopeptide_sequence,
                        peptide_id=peptide.id,
                        protein_id=peptide.protein_id,
                        hypothesis_id=peptide.hypothesis_id,
                        glycan_combination_id=gc.id)
                    yield glycopeptide


class PeptideGlycosylatingProcess(Process):
    def __init__(self, connection, hypothesis_id, input_queue, chunk_size=5000, done_event=None):
        Process.__init__(self)
        self.daemon = True
        self.connection = connection
        self.input_queue = input_queue
        self.chunk_size = chunk_size
        self.hypothesis_id = hypothesis_id
        self.done_event = done_event

    def task(self):
        database = DatabaseBoundOperation(self.connection)
        session = database.session
        has_work = True

        glycosylator = PeptideGlycosylator(database.session, self.hypothesis_id)
        result_accumulator = []

        while has_work:
            try:
                work_items = self.input_queue.get(timeout=5)
                if work_items is None:
                    has_work = False
                    continue
            except:
                if self.done_event.is_set():
                    has_work = False
                continue
            peptides = slurp(database.session, Peptide, work_items, flatten=False)
            for peptide in peptides:
                for gp in glycosylator.handle_peptide(peptide):
                    result_accumulator.append(gp)
                    if len(result_accumulator) > self.chunk_size:
                        session.bulk_save_objects(result_accumulator)
                        session.commit()
                        result_accumulator = []
            if len(result_accumulator) > 0:
                session.bulk_save_objects(result_accumulator)
                session.commit()
                result_accumulator = []

    def run(self):
        self.task()
