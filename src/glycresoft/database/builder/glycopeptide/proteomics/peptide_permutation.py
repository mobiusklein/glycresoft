import bisect
import itertools

from multiprocessing import Process, Queue, Event, cpu_count

from typing import List, Optional, Set, Tuple

from lxml.etree import XMLSyntaxError

from ms_deisotope.peak_dependency_network.intervals import Interval, IntervalTreeNode

from glycopeptidepy import enzyme
from .utils import slurp
from .uniprot import (uniprot, get_uniprot_accession, UniprotProteinDownloader, UniprotProteinXML, Empty)

from glypy.composition import formula
from glycopeptidepy.io import annotation
from glycopeptidepy.structure import sequence, modification, residue
from glycopeptidepy.structure.modification import ModificationRule
from glycopeptidepy.algorithm import PeptidoformGenerator
from glycopeptidepy import PeptideSequence

from glycresoft.task import TaskBase
from glycresoft.serialize import DatabaseBoundOperation
from glycresoft.serialize.hypothesis.peptide import Peptide, Protein

from six import string_types as basestring


RestrictedModificationTable = modification.RestrictedModificationTable
combinations = itertools.combinations
product = itertools.product
chain_iterable = itertools.chain.from_iterable

SequenceLocation = modification.SequenceLocation


def span_test(site_list, start, end):
    i = bisect.bisect_left(site_list, start)
    after_start = site_list[i:]
    out = []
    for j in after_start:
        if j < end:
            out.append(j)
        else:
            break
    return out


def n_glycan_sequon_sites(peptide: Peptide, protein: Protein, use_local_sequence: bool=False,
                          include_cysteine: bool = False, allow_modified=frozenset()):
    sites = set()
    sites |= set(site - peptide.start_position for site in span_test(
        protein.n_glycan_sequon_sites, peptide.start_position, peptide.end_position))
    if use_local_sequence:
        sites |= set(sequence.find_n_glycosylation_sequons(
            peptide.modified_peptide_sequence, allow_modified=allow_modified, include_cysteine=include_cysteine))
    return sorted(sites)


def o_glycan_sequon_sites(peptide: Peptide, protein: Protein=None, use_local_sequence: bool=False):
    sites = set()
    sites |= set(site - peptide.start_position for site in span_test(
        protein.o_glycan_sequon_sites, peptide.start_position, peptide.end_position))
    if use_local_sequence:
        sites |= set(sequence.find_o_glycosylation_sequons(
            peptide.modified_peptide_sequence))
    return sorted(sites)


def gag_sequon_sites(peptide: Peptide, protein: Protein = None, use_local_sequence: bool = False):
    sites = set()
    sites |= set(site - peptide.start_position for site in span_test(
        protein.glycosaminoglycan_sequon_sites, peptide.start_position, peptide.end_position))
    if use_local_sequence:
        sites = sequence.find_glycosaminoglycan_sequons(
            peptide.modified_peptide_sequence)
    return sorted(sites)


def get_base_peptide(peptide_obj):
    if isinstance(peptide_obj, Peptide):
        return PeptideSequence(peptide_obj.base_peptide_sequence)
    else:
        return PeptideSequence(str(peptide_obj))


class PeptidePermuter(PeptidoformGenerator):
    constant_modifications: List[ModificationRule]
    variable_modifications: List[ModificationRule]
    max_variable_modifications: int

    n_term_modifications: List[ModificationRule]
    c_term_modifications: List[ModificationRule]

    def prepare_peptide(self, sequence):
        return get_base_peptide(sequence)

    @classmethod
    def peptide_permutations(cls,
                             sequence,
                             constant_modifications: List[ModificationRule],
                             variable_modifications: List[ModificationRule],
                             max_variable_modifications: int=4):
        inst = cls(constant_modifications, variable_modifications,
                   max_variable_modifications)
        return inst(sequence)


peptide_permutations = PeptidePermuter.peptide_permutations


def cleave_sequence(sequence,
                    protease: enzyme.Protease,
                    missed_cleavages: int=2,
                    min_length: int=6,
                    max_length: int=60,
                    semispecific: bool=False):
    peptide: str
    start: int
    end: int
    missed: int
    for peptide, start, end, missed in protease.cleave(sequence, missed_cleavages=missed_cleavages,
                                                       min_length=min_length, max_length=max_length,
                                                       semispecific=semispecific):
        if "X" in peptide:
            continue
        yield peptide, start, end, missed


class ProteinDigestor(TaskBase):

    protease: enzyme.Protease
    constant_modifications: List[ModificationRule]
    variable_modifications: List[ModificationRule]
    peptide_permuter: PeptidePermuter
    min_length: int
    max_length: int
    max_missed_cleavages: int
    variable_signal_peptide: int
    max_variable_modifications: int
    semispecific: bool
    include_cysteine_n_glycosylation: bool
    require_glycosylation_sites: bool

    def __init__(self, protease, constant_modifications=None, variable_modifications=None,
                 max_missed_cleavages: int=2, min_length: int=6, max_length: int=60, semispecific: bool=False,
                 max_variable_modifications=None, require_glycosylation_sites: bool=False,
                 include_cysteine_n_glycosylation: bool=False):
        if constant_modifications is None:
            constant_modifications = []
        if variable_modifications is None:
            variable_modifications = []
        self.protease = self._prepare_protease(protease)
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications
        self.peptide_permuter = PeptidePermuter(
            self.constant_modifications,
            self.variable_modifications,
            max_variable_modifications=max_variable_modifications)
        self.max_missed_cleavages = max_missed_cleavages
        self.min_length = min_length
        self.max_length = max_length
        self.semispecific = semispecific
        self.include_cysteine_n_glycosylation = include_cysteine_n_glycosylation
        self.max_variable_modifications = max_variable_modifications
        self.require_glycosylation_sites = require_glycosylation_sites

    def _prepare_protease(self, protease):
        if isinstance(protease, enzyme.Protease):
            pass
        elif isinstance(protease, basestring):
            protease = enzyme.Protease(protease)
        elif isinstance(protease, (list, tuple)):
            protease = enzyme.Protease.combine(*protease)
        return protease

    def cleave(self, sequence: Protein):
        return cleave_sequence(sequence, self.protease, self.max_missed_cleavages,
                               min_length=self.min_length, max_length=self.max_length,
                               semispecific=self.semispecific)

    def digest(self, protein: Protein):
        sequence = protein.protein_sequence
        try:
            size = len(protein)
        except residue.UnknownAminoAcidException:
            size = len(sequence)
        for peptide, start, end, n_missed_cleavages in self.cleave(sequence):
            if end - start > self.max_length:
                continue
            protein_n_term = start == 0
            protein_c_term = end == size
            for inst in self.modify_string(
                    peptide, protein_n_term=protein_n_term, protein_c_term=protein_c_term):
                inst.count_missed_cleavages = n_missed_cleavages
                inst.start_position = start
                inst.end_position = end
                yield inst

    def modify_string(self, peptide: Peptide, protein_n_term: bool=False, protein_c_term: bool=False):
        base_peptide = str(peptide)
        for modified_peptide, n_variable_modifications in self.peptide_permuter(
                peptide, protein_n_term=protein_n_term, protein_c_term=protein_c_term):
            formula_string = formula(modified_peptide.total_composition())
            # formula_string = ""
            inst = Peptide(
                base_peptide_sequence=base_peptide,
                modified_peptide_sequence=str(modified_peptide),
                count_missed_cleavages=-1,
                count_variable_modifications=n_variable_modifications,
                sequence_length=len(modified_peptide),
                start_position=-1,
                end_position=-1,
                calculated_mass=modified_peptide.mass,
                formula=formula_string)
            yield inst

    def process_protein(self, protein_obj: Protein):
        protein_id = protein_obj.id
        hypothesis_id = protein_obj.hypothesis_id

        for peptide in self.digest(protein_obj):
            peptide.protein_id = protein_id
            peptide.hypothesis_id = hypothesis_id
            peptide.peptide_score = 0
            peptide.peptide_score_type = 'null_score'
            n_glycosites = n_glycan_sequon_sites(
                peptide, protein_obj, include_cysteine=self.include_cysteine_n_glycosylation)
            o_glycosites = o_glycan_sequon_sites(peptide, protein_obj)
            gag_glycosites = gag_sequon_sites(peptide, protein_obj)
            if self.require_glycosylation_sites:
                if (len(n_glycosites) + len(o_glycosites) + len(gag_glycosites)) == 0:
                    continue
            peptide.count_glycosylation_sites = len(n_glycosites)
            peptide.n_glycosylation_sites = sorted(n_glycosites)
            peptide.o_glycosylation_sites = sorted(o_glycosites)
            peptide.gagylation_sites = sorted(gag_glycosites)
            yield peptide

    def __call__(self, protein_obj):
        return self.process_protein(protein_obj)

    @classmethod
    def digest_protein(cls, sequence, protease: enzyme.Protease, constant_modifications: List[ModificationRule] = None,
                       variable_modifications: List[ModificationRule] = None, max_missed_cleavages: int=2,
                       min_length: int=6, max_length: int=60, semispecific: bool=False):
        inst = cls(
            protease, constant_modifications, variable_modifications,
            max_missed_cleavages, min_length, max_length, semispecific)
        if isinstance(sequence, basestring):
            sequence = Protein(protein_sequence=sequence)
        return inst(sequence)


digest = ProteinDigestor.digest_protein


class ProteinDigestingProcess(Process):
    process_name = "protein-digest-worker"

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
            except Exception:
                if self.done_event.is_set():
                    has_work = False
                continue
            proteins = slurp(session, Protein, work_items, flatten=False)
            acc = []

            threshold_size = 3000

            for protein in proteins:
                size = len(protein.protein_sequence)
                if size > threshold_size:
                    self.message_handler("...... Started digesting %s (%d)" % (protein.name, size))
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
                            "...... Digested %d peptides from %r (%d)" % (
                                i, protein.name, size))
                if size > threshold_size:
                    self.message_handler("...... Finished digesting %s (%d)" % (protein.name, size))
            session.bulk_save_objects(acc)
            session.commit()
            acc = []
        if acc:
            session.bulk_save_objects(acc)
            session.commit()
            acc = []

    def run(self):
        new_name = getattr(self, 'process_name', None)
        if new_name is not None:
            TaskBase().try_set_process_name(new_name)
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
        input_queue = Queue(2 * self.n_processes)
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
        if n <= self.n_processes:
            chunk_size = 1
        elif n < 100:
            chunk_size = 2
        else:
            chunk_size = int(n / 100.0)
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
                self.log("... Dealt Proteins %d-%d %0.2f%%" % (
                    i - chunk_size, min(i, n), (min(i, n) / float(n)) * 100))
                last = i

                error_occurred = False
                for process in processes:
                    if process.exitcode is not None and process.exitcode != 0:
                        error_occurred = True

                if error_occurred:
                    self.error("An error occurred while digesting proteins.")
                    done_event.set()
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    logger.stop()
                    raise ValueError(
                        "One or more worker processes exited with an error.")

        done_event.set()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise ValueError("One or more worker processes exited with an error.")
        logger.stop()


class PeptideInterval(Interval):
    def __init__(self, peptide):
        Interval.__init__(self, peptide.start_position, peptide.end_position, [peptide])
        self.data['peptide'] = str(peptide)


class ProteinSplitter(TaskBase):

    constant_modifications: List[ModificationRule]
    variable_modifications: List[ModificationRule]
    min_length: int
    variable_signal_peptide: int
    include_cysteine_n_glycosylation: bool

    def __init__(self, constant_modifications=None, variable_modifications=None,
                 min_length=6, variable_signal_peptide=10, include_cysteine_n_glycosylation: bool=False):
        if constant_modifications is None:
            constant_modifications = []
        if variable_modifications is None:
            variable_modifications = []

        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications
        self.min_length = min_length
        self.variable_signal_peptide = variable_signal_peptide
        self.peptide_permuter = PeptidePermuter(
            self.constant_modifications, self.variable_modifications)
        self.include_cysteine_n_glycosylation = include_cysteine_n_glycosylation

    def __reduce__(self):
        return self.__class__, (self.constant_modifications,
                                self.variable_modifications,
                                self.min_length,
                                self.variable_signal_peptide,
                                self.include_cysteine_n_glycosylation)

    def handle_protein(self, protein_obj: Protein, sites: Optional[Set[int]]=None):
        annot_sites = self.get_split_sites_from_annotations(protein_obj)
        if sites is None:
            try:
                accession = get_uniprot_accession(protein_obj.name)
                if accession:
                    try:
                        extra_sites, more_annots = self.get_split_sites_from_uniprot(accession)
                        # Trigger SQLAlchemy mutable update via assignment instead of in-place updates
                        protein_obj.annotations = list(protein_obj.annotations) + list(more_annots)
                        return self.split_protein(protein_obj, sorted(annot_sites | extra_sites))
                    except IOError:
                        return self.split_protein(protein_obj, annot_sites)
                else:
                    return self.split_protein(protein_obj, annot_sites)
            except XMLSyntaxError:
                return self.split_protein(protein_obj, annot_sites)
            except Exception as e:
                self.error(
                    ("An unhandled error occurred while retrieving"
                     " non-proteolytic cleavage sites"), e)
                return self.split_protein(protein_obj, annot_sites)
        else:
            if not isinstance(sites, set):
                sites = set(sites)
            try:
                return self.split_protein(protein_obj, sorted(sites | annot_sites))
            except IOError:
                return []

    def get_split_sites_from_annotations(self, protein: Protein) -> Set[int]:
        annots = protein.annotations
        split_sites = self._split_sites_from_annotations(annots)
        return split_sites

    def _split_sites_from_annotations(self, annots: annotation.AnnotationCollection) -> Set[int]:
        cleavable_annots = annots.cleavable()
        split_sites = set()
        for annot in cleavable_annots:
            split_sites.add(annot.start)
            split_sites.add(annot.end)
            if self.variable_signal_peptide and annot.feature_type == annotation.SignalPeptide.feature_type:
                for i in range(1, self.variable_signal_peptide + 1):
                    split_sites.add(annot.end + i)
        if 0 in split_sites:
            split_sites.remove(0)
        return split_sites

    def get_split_sites_from_features(self, record) -> Set[int]:
        if isinstance(record, uniprot.UniProtProtein):
            annots = annotation.from_uniprot(record)
        elif isinstance(record, annotation.AnnotationCollection):
            annots = record
        return self._split_sites_from_annotations(annots)

    def get_split_sites_from_uniprot(self, accession: str) -> Tuple[Set[int], annotation.AnnotationCollection]:
        try:
            record = uniprot.get(accession)
        except Exception as err:
            self.error(f"Failed to obtain Uniprot Record for {accession!r}: {err.__class__}:{str(err)}")
            return set()
        if record is None:
            self.error(f"Failed to obtain Uniprot Record for {accession!r}")
            return set()
        sites = self.get_split_sites_from_features(record)
        return sites, annotation.from_uniprot(record)

    def _make_split_expression(self, sites: List[int]) -> List:
        return [
            (Peptide.start_position < s) & (Peptide.end_position > s) for s in sites]

    def _permuted_peptides(self, sequence):
        return self.peptide_permuter(sequence)

    def split_protein(self, protein_obj: Protein, sites: Optional[List[int]]=None):
        if isinstance(sites, set):
            sites = sorted(sites)
        if sites is None:
            sites = []
        if not sites:
            return
        seen = set()
        sites_seen = set()
        peptides = protein_obj.peptides.all()
        peptide_intervals = IntervalTreeNode.build(map(PeptideInterval, peptides))
        for site in sites:
            overlap_region = peptide_intervals.contains_point(site - 1)
            spanned_intervals: IntervalTreeNode = IntervalTreeNode.build(overlap_region)
            # No spanned peptides. May be caused by regions of protein which digest to peptides
            # of unacceptable size.
            if spanned_intervals is None:
                continue
            lo = spanned_intervals.start
            hi = spanned_intervals.end
            # Get the set of all sites spanned by any peptide which spans the current query site
            spanned_sites = [s for s in sites if lo <= s <= hi]
            for i in range(1, len(spanned_sites) + 1):
                for split_sites in itertools.combinations(spanned_sites, i):
                    site_key = frozenset(split_sites)
                    if site_key in sites_seen:
                        continue
                    sites_seen.add(site_key)
                    spanning_peptides_query = spanned_intervals.contains_point(split_sites[0])
                    for site_j in split_sites[1:]:
                        spanning_peptides_query = [
                            sp for sp in spanning_peptides_query if site_j in sp
                        ]
                    spanning_peptides = []
                    for sp in spanning_peptides_query:
                        spanning_peptides.extend(sp)
                    for peptide in spanning_peptides:
                        peptide_start_position = peptide.start_position
                        adjusted_sites = [0] + [s - peptide_start_position for s in split_sites] + [
                            peptide.sequence_length]
                        for j in range(len(adjusted_sites) - 1):
                            # TODO: Cleavage sites may be off by one in the start. Revisit the index math here.
                            begin = adjusted_sites[j]
                            end = adjusted_sites[j + 1]
                            if end - begin < self.min_length:
                                continue
                            start_position = begin + peptide_start_position
                            end_position = end + peptide_start_position
                            if (start_position, end_position) in seen:
                                continue
                            else:
                                seen.add((start_position, end_position))
                            for modified_peptide, n_variable_modifications in self._permuted_peptides(
                                    peptide.base_peptide_sequence[begin:end]):

                                inst = Peptide(
                                    base_peptide_sequence=str(peptide.base_peptide_sequence[begin:end]),
                                    modified_peptide_sequence=str(modified_peptide),
                                    count_missed_cleavages=peptide.count_missed_cleavages,
                                    count_variable_modifications=n_variable_modifications,
                                    sequence_length=len(modified_peptide),
                                    start_position=start_position,
                                    end_position=end_position,
                                    calculated_mass=modified_peptide.mass,
                                    formula=formula(modified_peptide.total_composition()),
                                    protein_id=protein_obj.id)
                                inst.hypothesis_id = protein_obj.hypothesis_id
                                inst.peptide_score = 0
                                inst.peptide_score_type = 'null_score'
                                n_glycosites = n_glycan_sequon_sites(
                                    inst, protein_obj, include_cysteine=self.include_cysteine_n_glycosylation)
                                o_glycosites = o_glycan_sequon_sites(inst, protein_obj)
                                gag_glycosites = gag_sequon_sites(inst, protein_obj)
                                inst.count_glycosylation_sites = len(n_glycosites)
                                inst.n_glycosylation_sites = sorted(n_glycosites)
                                inst.o_glycosylation_sites = sorted(o_glycosites)
                                inst.gagylation_sites = sorted(gag_glycosites)
                                yield inst


class UniprotProteinAnnotator(TaskBase):
    hypothesis_builder: DatabaseBoundOperation
    protein_ids: List[int]
    constant_modifications: List[ModificationRule]
    variable_modifications: List[ModificationRule]
    variable_signal_peptide: int
    include_cysteine_n_glycosylation: bool

    def __init__(self, hypothesis_builder, protein_ids, constant_modifications,
                 variable_modifications, variable_signal_peptide=10,
                 include_cysteine_n_glycosylation: bool=False,
                 uniprot_source_file: Optional[str]=None):
        self.hypothesis_builder = hypothesis_builder
        self.protein_ids = protein_ids
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications
        self.variable_signal_peptide = variable_signal_peptide
        self.session = hypothesis_builder.session
        self.include_cysteine_n_glycosylation = include_cysteine_n_glycosylation
        self.uniprot_source_file = uniprot_source_file

    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    def commit(self):
        return self.session.commit()

    def run(self):
        self.log("Begin Applying Protein Annotations")
        splitter = ProteinSplitter(self.constant_modifications, self.variable_modifications,
                                   variable_signal_peptide=self.variable_signal_peptide,
                                   include_cysteine_n_glycosylation=self.include_cysteine_n_glycosylation)
        i = 0
        j = 0
        protein_ids = self.protein_ids
        protein_names = [self.query(Protein.name).filter(
            Protein.id == protein_id).first()[0] for protein_id in protein_ids]
        name_to_id = {n: i for n, i in zip(protein_names, protein_ids)}
        if self.uniprot_source_file is not None:
            self.log(f"... Loading from local XML file {self.uniprot_source_file!r}")
            uniprot_queue = UniprotProteinXML(self.uniprot_source_file, protein_names)
        else:
            self.log("... Loading from UniProt web service")
            uniprot_queue = UniprotProteinDownloader(
                protein_names,
                n_threads=getattr(self.hypothesis_builder, "n_processes", cpu_count() // 2) * 4)
            uniprot_queue.start()
        n = len(protein_ids)
        interval = int(min((n * 0.1) + 1, 1000))
        acc = []
        seen = set()
        # TODO: This seems to periodically exit before finishing processing of all proteins?
        while True:
            try:
                protein_name, record = uniprot_queue.get()
                protein_id = name_to_id[protein_name]
                seen.add(protein_id)
                protein = self.query(Protein).get(protein_id)
                i += 1
                # Record not found in UniProt
                if isinstance(record, (Exception, str)) and not protein.annotations:
                    message = str(record)
                    # Many, many records that used to exist no longer do. We don't care if they are absent.
                    # if "Document is empty" not in message:
                    self.log(f"... Skipping {protein_name}: {message}")
                    continue
                if record is None and not protein.annotations:
                    self.log(f"... Skipping {protein_name}: no record found")
                    continue
                if i % interval == 0:
                    self.log(
                        f"... {i * 100. / n:0.3f}% Complete ({i}/{n}). {j} Peptides Produced.")

                if not isinstance(record, (uniprot.UniProtProtein, annotation.AnnotationCollection)):
                    sites = None
                else:
                    if isinstance(record, uniprot.UniProtProtein):
                        record = annotation.from_uniprot(record)
                    # Trigger SQLAlchemy mutable update via assignment instead of in-place updates
                    protein.annotations = list(protein.annotations) + list(record)
                    self.session.add(protein)
                    sites = splitter.get_split_sites_from_features(record)
                for peptide in splitter.handle_protein(protein, sites):
                    acc.append(peptide)
                    j += 1
                    if len(acc) > 100000:
                        self.log(
                            f"... {i * 100. / n:0.3f}% Complete ({i}/{n}). {j} Peptides Produced.")
                        self.session.bulk_save_objects(acc)
                        self.session.commit()
                        acc = []
            except Empty:
                uniprot_queue.join()
                if uniprot_queue.done_event.is_set():
                    break

        leftover_ids = set(protein_ids) - seen
        if leftover_ids:
            self.log(f"{len(leftover_ids)} IDs not covered by queue, sequentially querying UniProt")

        for protein_id in leftover_ids:
            i += 1
            protein = self.query(Protein).get(protein_id)
            if i % interval == 0:
                self.log(
                    f"... {i * 100. / n:0.3f}% Complete ({i}/{n}). {j} Peptides Produced.")
            for peptide in splitter.handle_protein(protein):
                acc.append(peptide)
                j += 1
                if len(acc) > 100000:
                    self.log(
                        f"... {i * 100. / n:0.3f}% Complete ({i}/{n}). {j} Peptides Produced.")
                    self.session.bulk_save_objects(acc)
                    self.session.commit()
                    acc = []
            self.session.add(protein)
        self.log(
            f"... {i * 100. / n:0.3f}% Complete ({i}/{n}). {j} Peptides Produced.")
        self.session.bulk_save_objects(acc)
        self.session.commit()
        acc = []
