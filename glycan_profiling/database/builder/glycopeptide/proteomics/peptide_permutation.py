from collections import defaultdict
import itertools
from multiprocessing import Process, Queue, Event

from glycopeptidepy import enzyme
from .utils import slurp

from glypy.composition import formula
from glycopeptidepy.structure import sequence, modification, residue
from glycopeptidepy import PeptideSequence

from glycan_profiling.task import TaskBase
from glycan_profiling.serialize import DatabaseBoundOperation
from glycan_profiling.serialize.hypothesis.peptide import Peptide, Protein

RestrictedModificationTable = modification.RestrictedModificationTable
combinations = itertools.combinations
product = itertools.product
chain_iterable = itertools.chain.from_iterable

SequenceLocation = modification.SequenceLocation


def parent_sequence_aware_n_glycan_sequon_sites(peptide, protein):
    sites = set(sequence.find_n_glycosylation_sequons(
        peptide.modified_peptide_sequence))
    sites |= set(site - peptide.start_position for site in protein.glycosylation_sites
                 if peptide.start_position <= site < peptide.end_position)
    return list(sites)


def o_glycan_sequon_sites(peptide, protein=None):
    sites = sequence.find_o_glycosylation_sequons(
        peptide.modified_peptide_sequence)
    return sites


def gag_sequon_sites(peptide, protein=None):
    sites = sequence.find_glycosaminoglycan_sequons(
        peptide.modified_peptide_sequence)
    return sites


def split_terminal_modifications(modifications):
    n_terminal = []
    c_terminal = []
    internal = []

    for mod in modifications:
        n_term = mod.n_term_targets
        if n_term:
            n_term_rule = mod.clone(n_term)
            mod = mod - n_term_rule
            n_terminal.append(n_term_rule)
        c_term = mod.c_term_targets
        if c_term:
            c_term_rule = mod.clone(c_term)
            mod = mod - c_term_rule
            c_terminal.append(c_term_rule)
        if (mod.targets):
            internal.append(mod)

    return n_terminal, c_terminal, internal


def get_base_peptide(peptide_obj):
    if isinstance(peptide_obj, Peptide):
        return PeptideSequence(peptide_obj.base_peptide_sequence)
    else:
        return PeptideSequence(str(peptide_obj))


def modification_series(variable_sites):
    """Given a dictionary mapping between modification names and
    an iterable of valid sites, create a dictionary mapping between
    modification names and a list of valid sites plus the constant `None`

    Parameters
    ----------
    variable_sites : dict
        Description

    Returns
    -------
    dict
        Description
    """
    sites = defaultdict(list)
    for mod, varsites in variable_sites.items():
        for site in varsites:
            sites[site].append(mod)
    for site in list(sites):
        sites[site].append(None)
    return sites


def remove_empty_sites(site_mod_pairs):
    return [sm for sm in site_mod_pairs if sm[1] is not None]


def site_modification_assigner(modification_sites_dict):
    sites = modification_sites_dict.keys()
    choices = modification_sites_dict.values()
    for selected in itertools.product(*choices):
        site_mod_pairs = zip(sites, selected)
        yield remove_empty_sites(site_mod_pairs)


def peptide_permutations(sequence, constant_modifications, variable_modifications, maximum_variable_modifications=4):
    try:
        sequence = get_base_peptide(sequence)
    except residue.UnknownAminoAcidException:
        return
    (n_term_modifications,
     c_term_modifications,
     variable_modifications) = split_terminal_modifications(variable_modifications)

    n_term_modifications = [
        mod for mod in n_term_modifications if mod.find_valid_sites(sequence)]
    c_term_modifications = [
        mod for mod in c_term_modifications if mod.find_valid_sites(sequence)]

    n_term_modifications.append(None)
    c_term_modifications.append(None)

    has_fixed_n_term = False
    has_fixed_c_term = False

    for mod in constant_modifications:
        for site in mod.find_valid_sites(sequence):
            if site == SequenceLocation.n_term:
                has_fixed_n_term = True
            elif site == SequenceLocation.c_term:
                has_fixed_c_term = True
            sequence.add_modification(site, mod.name)

    if has_fixed_n_term:
        n_term_modifications = [None]
    if has_fixed_c_term:
        c_term_modifications = [None]

    variable_sites = {
        mod.name: set(
            mod.find_valid_sites(sequence)) for mod in variable_modifications}
    modification_sites = modification_series(variable_sites)

    for n_term, c_term in itertools.product(n_term_modifications, c_term_modifications):
        for assignments in site_modification_assigner(modification_sites):
            if len(assignments) > maximum_variable_modifications:
                continue
            n_variable = 0
            result = sequence.clone()
            if n_term is not None:
                result.n_term = n_term
                n_variable += 1
            if c_term is not None:
                result.c_term = c_term
                n_variable += 1
            for site, mod in assignments:
                if mod is not None:
                    result.add_modification(site, mod)
                    n_variable += 1
            yield result, n_variable


def cleave_sequence(sequence, protease, missed_cleavages=2):
    for peptide, start, end in protease.cleave(sequence, missed_cleavages=missed_cleavages):
        if len(peptide) < 5:
            continue
        missed = protease.missed_cleavages(peptide)
        if missed > missed_cleavages:
            continue

        if "X" in peptide:
            continue

        yield peptide, start, end, missed


def digest(sequence, protease, constant_modifications=None, variable_modifications=None, max_missed_cleavages=2,
           max_length=60):
    if constant_modifications is None:
        constant_modifications = []
    if variable_modifications is None:
        variable_modifications = []
    if isinstance(protease, enzyme.Protease):
        pass
    elif isinstance(protease, basestring):
        protease = enzyme.Protease(protease)
    elif isinstance(protease, (list, tuple)):
        protease = enzyme.Protease.combine(*protease)
    for peptide, start, end, n_missed_cleavages in cleave_sequence(sequence, protease, max_missed_cleavages):
        if end - start > max_length:
            continue
        for modified_peptide, n_variable_modifications in peptide_permutations(
                peptide, constant_modifications, variable_modifications):
            inst = Peptide(
                base_peptide_sequence=str(peptide),
                modified_peptide_sequence=str(modified_peptide),
                count_missed_cleavages=n_missed_cleavages,
                count_variable_modifications=n_variable_modifications,
                sequence_length=len(modified_peptide),
                start_position=start,
                end_position=end,
                calculated_mass=modified_peptide.mass,
                formula=formula(modified_peptide.total_composition()))
            yield inst


class ProteinDigestor(TaskBase):

    def __init__(self, protease, constant_modifications=None, variable_modifications=None, max_missed_cleavages=2,
                 max_length=60):
        if constant_modifications is None:
            constant_modifications = []
        if variable_modifications is None:
            variable_modifications = []
        if isinstance(protease, enzyme.Protease):
            pass
        elif isinstance(protease, basestring):
            protease = enzyme.Protease(protease)
        elif isinstance(protease, (list, tuple)):
            protease = enzyme.Protease.combine(*protease)
        self.protease = protease
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications
        self.max_missed_cleavages = max_missed_cleavages
        self.max_length = max_length

    def process_protein(self, protein_obj):
        protein_id = protein_obj.id
        hypothesis_id = protein_obj.hypothesis_id

        for peptide in digest(
                protein_obj.protein_sequence, self.protease, self.constant_modifications,
                self.variable_modifications, self.max_missed_cleavages,
                max_length=self.max_length):
            peptide.protein_id = protein_id
            peptide.hypothesis_id = hypothesis_id
            peptide.peptide_score = 0
            peptide.peptide_score_type = 'null_score'
            n_glycosites = parent_sequence_aware_n_glycan_sequon_sites(
                peptide, protein_obj)
            o_glycosites = o_glycan_sequon_sites(peptide, protein_obj)
            gag_glycosites = gag_sequon_sites(peptide, protein_obj)
            peptide.count_glycosylation_sites = len(n_glycosites)
            peptide.n_glycosylation_sites = sorted(n_glycosites)
            peptide.o_glycosylation_sites = sorted(o_glycosites)
            peptide.gagylation_sites = sorted(gag_glycosites)
            yield peptide


class ProteinDigestingProcess(Process):

    def __init__(self, connection, hypothesis_id, input_queue, digestor, done_event=None,
                 chunk_size=5000, message_handler=None):
        Process.__init__(self)
        self.connection = connection
        self.input_queue = input_queue
        self.hypothesis_id = hypothesis_id
        self.done_event = done_event
        self.digestor = digestor
        self.chunk_size = chunk_size
        self.message_handler = message_handler

    def task(self):
        database = DatabaseBoundOperation(self.connection)
        session = database.session
        has_work = True

        digestor = self.digestor
        acc = []
        if self.message_handler is None:
            self.message_handler = lambda x: None
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
            proteins = slurp(session, Protein, work_items, flatten=False)
            acc = []

            threshold_size = 3000

            for protein in proteins:
                size = len(protein.protein_sequence)
                if size > threshold_size:
                    self.message_handler("Started digesting %s (%d)" % (protein.name, size))
                i = 0
                for peptide in digestor.process_protein(protein):
                    acc.append(peptide)
                    i += 1
                    if len(acc) > self.chunk_size:
                        session.bulk_save_objects(acc)
                        session.commit()
                        acc = []
                    if i % 10000 == 0:
                        self.message_handler(
                            "Digested %d peptides from %r (%d)" % (
                                i, protein.name, size))
                if size > threshold_size:
                    self.message_handler("Finished digesting %s (%d)" % (protein.name, size))
            session.bulk_save_objects(acc)
            session.commit()
            acc = []
        if acc:
            session.bulk_save_objects(acc)
            session.commit()
            acc = []

    def run(self):
        self.task()


class MultipleProcessProteinDigestor(TaskBase):
    def __init__(self, connection, hypothesis_id, protein_ids, digestor, n_processes=4):
        self.connection = connection
        self.hypothesis_id = hypothesis_id
        self.protein_ids = protein_ids
        self.digestor = digestor
        self.n_processes = n_processes

    def run(self):
        logger = self.ipc_logger()
        input_queue = Queue(20 * self.n_processes)
        done_event = Event()
        processes = [
            ProteinDigestingProcess(
                self.connection, self.hypothesis_id, input_queue,
                self.digestor, done_event=done_event,
                message_handler=logger.sender()) for i in range(
                self.n_processes)
        ]
        protein_ids = self.protein_ids
        i = 0
        n = len(protein_ids)
        chunk_size = 2
        interval = 30
        for process in processes:
            input_queue.put(protein_ids[i:(i + chunk_size)])
            i += chunk_size
            process.start()

        last = i
        while i < n:
            input_queue.put(protein_ids[i:(i + chunk_size)])
            i += chunk_size
            if i - last > interval:
                self.log("... Dealt Proteins %d-%d %0.2f%%" % (i - chunk_size, min(i, n), (min(i, n) / float(n)) * 100))
                last = i

        done_event.set()
        for process in processes:
            process.join()
        logger.stop()
