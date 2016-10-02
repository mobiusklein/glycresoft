import itertools
from uuid import uuid4
from collections import defaultdict
from multiprocessing import Process, Queue, Event

from glypy import Composition
from glypy.composition import formula

from glycan_profiling.serialize import DatabaseBoundOperation
from glycan_profiling.serialize.hypothesis import GlycopeptideHypothesis
from glycan_profiling.serialize.hypothesis.peptide import Glycopeptide, Peptide
from glycan_profiling.task import TaskBase

from glycan_profiling.database.builder.glycan import glycan_combinator

from glycresoft_sqlalchemy.structure.sequence import _n_glycosylation


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


class GlycopeptideHypothesisSerializerBase(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection, hypothesis_name=None, glycan_hypothesis_id=None):
        DatabaseBoundOperation.__init__(self, database_connection)
        self._hypothesis_name = hypothesis_name
        self._hypothesis_id = None
        self._hypothesis = None
        self._glycan_hypothesis_id = glycan_hypothesis_id

    def _construct_hypothesis(self):
        if self._hypothesis_name is None:
            self._hypothesis_name = self._make_name()

        if self.glycan_hypothesis_id is None:
            raise ValueError("glycan_hypothesis_id must not be None")
        self._hypothesis = GlycopeptideHypothesis(
            name=self._hypothesis_name, glycan_hypothesis_id=self._glycan_hypothesis_id)
        self.session.add(self._hypothesis)
        self.session.commit()

        self._hypothesis_id = self._hypothesis.id
        self._hypothesis_name = self._hypothesis.name
        self._glycan_hypothesis_id = self._hypothesis.glycan_hypothesis_id

    def _make_name(self):
        return "GlycopeptideHypothesis-" + str(uuid4().hex)

    @property
    def hypothesis(self):
        if self._hypothesis is None:
            self._construct_hypothesis()
        return self._hypothesis

    @property
    def hypothesis_name(self):
        if self._hypothesis_name is None:
            self._construct_hypothesis()
        return self._hypothesis_name

    @property
    def hypothesis_id(self):
        if self._hypothesis_id is None:
            self._construct_hypothesis()
        return self._hypothesis_id

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


class PeptideGlycosylator(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id
        self.glycan_combinations = self.session.query(
            glycan_combinator.GlycanCombination).filter(
            glycan_combinator.GlycanCombination.hypothesis_id == hypothesis_id).all()
        self.build_size_table()

    def build_size_table(self):
        size_map = defaultdict(list)
        for gc in self.glycan_combinations:
            size_map[gc.count].append(gc)
        self.by_size = size_map

    def handle_peptide(self, peptide):
        water = Composition("H2O")
        peptide_composition = Composition(peptide.formula)
        glycosylation_mod = _n_glycosylation.name
        unoccupied_sites = set(peptide.n_glycosylation_sites)
        obj = peptide.convert()
        for site in list(unoccupied_sites):
            if obj[site][1]:
                unoccupied_sites.remove(site)
        for i in range(len(unoccupied_sites)):
            i += 1
            for gc in self.by_size[i]:
                total_mass = peptide.calculated_mass + gc.calculated_mass - (gc.count * water.mass)
                formula_string = formula(peptide_composition + Composition(gc.formula) - (water * gc.count))

                for site_set in itertools.combinations(unoccupied_sites, i):
                    sequence = peptide.convert()
                    for site in site_set:
                        sequence.add_modification(site, glycosylation_mod)
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
                        session.add_all(result_accumulator)
                        session.commit()
                        result_accumulator = []
            session.add_all(result_accumulator)
            session.commit()
            result_accumulator = []

    def run(self):
        self.task()
