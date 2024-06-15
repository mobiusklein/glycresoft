import re
import threading
import multiprocessing

from queue import Empty
from typing import IO, List, Deque, Union

import urllib3
from ms_deisotope.data_source._compression import get_opener

from glycopeptidepy.io import uniprot, fasta, annotation
from glycopeptidepy.io.utils import UniprotToPeffConverter

from glycresoft.task import TaskBase
from glycresoft.serialize import Protein


uniprot_accession_pattern = re.compile(
    r"""(([OPQ]\d[A-Z0-9][A-Z0-9][A-Z0-9]\d)|
        ([A-Z0-9]\d[A-Z][A-Z0-9][A-Z0-9]\d)|
        ([A-NR-Z]\d[A-Z][A-Z0-9][A-Z0-9]\d[A-Z][A-Z0-9][A-Z0-9]\d))""",
        re.X)


def is_uniprot_accession(accession: str) -> bool:
    return bool(uniprot_accession_pattern.match(accession))


def get_uniprot_accession(name):
    try:
        peff_header = fasta.peff_parser(name)
        # Don't bother UniProt over PEFF specifications, they won't parse correctly
        # with the partial UniProt parser
        if peff_header:
            return None
    except fasta.UnparsableDeflineError:
        pass
    try:
        return fasta.partial_uniprot_parser(name).accession
    except (AttributeError, fasta.UnparsableDeflineError):
        # Some programs will just put the accession number into the mzIdentML
        # identifier, so we'll check if the first word looks like an accession
        # number and try it if so.
        accession = str(name).split(" ")[0]
        if is_uniprot_accession(accession):
            return accession
        return None


def retry(task, n=5):
    result = None
    errs = []
    for i in range(n):
        try:
            result = task()
            return result, errs
        except Exception as err:
            errs.append(err)
    raise errs[-1]


HAS_BATCH_HANDLER = hasattr(uniprot, 'get_features_for_many')


class UniprotSource(TaskBase):
    def task_handler(self, accession_number):
        accession = get_uniprot_accession(accession_number)
        if accession is None:
            accession = accession_number
        protein_data, errs = retry(lambda: uniprot.get(accession), 5)
        self.output_queue.put((accession_number, protein_data))
        return protein_data

    def batch_handler(self, accession_list):
        idents_map = {}
        for acc in accession_list:
            acc_ = get_uniprot_accession(acc)
            if acc_ is None:
                acc_ = acc
            idents_map[acc_] = acc
        idents = list(idents_map)
        result, errs = retry(lambda: uniprot.get_features_for_many(idents), 5)
        if errs:
            self.log(f"... Handled batch of size {len(accession_list)} and encountered {len(errs)} errors")

        for (acc, protein_data) in result:
            name = idents_map[acc]
            self.output_queue.put((name, protein_data))

    def fetch(self, accession_number):
        if isinstance(accession_number, list) and HAS_BATCH_HANDLER:
            try:
                self.batch_handler(accession_number)
            except Exception as e:
                self.error_handler(accession_number[0], e)
        else:
            try:
                self.task_handler(accession_number)
            except Exception as e:
                self.error_handler(accession_number, e)

    def error_handler(self, accession_number, error):
        self.output_queue.put((accession_number, Exception(str(error))))


class UniprotRequestingProcess(multiprocessing.Process, UniprotSource):
    process_name = "glycresoft-annotation-worker"

    def __init__(self, input_queue, output_queue, feeder_done_event, done_event):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.feeder_done_event = feeder_done_event
        self.done_event = done_event

    def run(self):
        new_name = getattr(self, 'process_name', None)
        if new_name is not None:
            TaskBase().try_set_process_name(new_name)
        urllib3.disable_warnings()
        while True:
            try:
                accession = self.input_queue.get(True, 3)
            except Empty:
                if self.feeder_done_event.is_set():
                    break
            self.input_queue.task_done()
            self.fetch(accession)

        self.done_event.set()


def chunked(seq, size=128):
    n = len(seq)
    for i in range(0, n + size, size):
        z = seq[i:i + size]
        if z:
            yield z


class UniprotProteinDownloader(UniprotSource):
    accession_list: List[str]
    input_queue: multiprocessing.JoinableQueue
    output_queue: multiprocessing.Queue
    no_more_work: multiprocessing.Event
    done_event: multiprocessing.Event
    workers: List[UniprotRequestingProcess]

    def __init__(self, accession_list, n_threads=10):
        self.accession_list = accession_list
        self.n_threads = n_threads
        self.input_queue = multiprocessing.JoinableQueue(2000)
        self.output_queue = multiprocessing.Queue()
        self.no_more_work = multiprocessing.Event()
        self.done_event = multiprocessing.Event()
        self.workers = []

    def task_handler(self, accession_number):
        accession = get_uniprot_accession(accession_number)
        if accession is None:
            accession = accession_number
        protein_data = uniprot.get(accession)
        self.output_queue.put((accession_number, protein_data))
        return protein_data

    def fetch(self, accession_number):
        try:
            self.task_handler(accession_number)
        except Exception as e:
            self.error_handler(accession_number, e)

    def error_handler(self, accession_number, error):
        self.output_queue.put((accession_number, error))

    def feeder_task(self):
        n = len(self.accession_list)
        k = n // max(100, self.n_threads)
        k = max(min(k, 100), 1)
        self.log(f"... Submitting UniProt queries in batches of size {k}")

        for i, item in enumerate(chunked(self.accession_list, k)):
            if i % 1000 == 0 and i:
                self.input_queue.join()
            self.input_queue.put(item)
        self.no_more_work.set()

    def run(self):
        feeder = threading.Thread(target=self.feeder_task)
        feeder.start()
        self.workers = workers = []
        n = min(self.n_threads, len(self.accession_list))
        for i in range(n):
            t = UniprotRequestingProcess(
                self.input_queue, self.output_queue, self.no_more_work,
                multiprocessing.Event())
            t.daemon = True
            t.start()
            workers.append(t)

    def join(self):
        for worker in self.workers:
            if not worker.done_event.is_set():
                break
            else:
                worker.join()
        else:
            self.done_event.set()
            return True
        return False

    def start(self):
        self.run()

    def get(self, blocking=True, timeout=3):
        return self.output_queue.get(blocking, timeout)


class UniprotProteinXML(UniprotSource):
    path: Union[str, IO]
    store: annotation.AnnotationDatabase
    queue: Deque[str]
    done_event: threading.Event

    def __init__(self, path: str, ids: List[str]):
        self.path = path
        self.store = {}
        self.work = Deque(ids)
        self.load()
        self.done_event = threading.Event()

    def join(self):
        return

    def load(self):
        if self.path == '-':
            self.store = annotation.AnnotationDatabase({})
        else:
            stream = get_opener(self.path)
            if uniprot.is_uniprot_xml(stream):
                self.store = annotation.AnnotationDatabase.from_uniprot_xml(stream)
            else:
                self.store = annotation.AnnotationDatabase.load(stream)

    def fetch(self, accession_number: str):
        accession = get_uniprot_accession(accession_number)
        try:
            features = self.store[accession]
            return accession_number, features
        except KeyError:
            return accession_number, Exception(f"{accession_number!r} not found")

    def get(self, *args, **kwargs):
        if self.work:
            acc = self.work.popleft()
            if not self.work:
                self.done_event.set()
            return self.fetch(acc)
        else:
            raise Empty()


class UniprotProteinSource(TaskBase):
    def __init__(self, accession_list, hypothesis_id, n_threads=4):
        self._accession_list = accession_list
        self.n_threads = n_threads
        self.downloader = None
        self.hypothesis_id = hypothesis_id
        self.proteins = []

        self._clean_accession_list()

    def _clean_accession(self, accession_string):
        if accession_string.startswith(">"):
            accession_string = accession_string[1:]
        accession_string = accession_string.strip()
        result = None
        if "|" in accession_string:
            try:
                fields = fasta.default_parser(accession_string)
                result = fields.accession
            except fasta.UnparsableDeflineError:
                result = accession_string
        else:
            result = accession_string
        return result

    def _clean_accession_list(self):
        cleaned = []
        for entry in self._accession_list:
            cleaned.append(self._clean_accession(entry))
        self._accession_list = cleaned

    def _make_downloader(self):
        self.downloader = UniprotProteinDownloader(
            self._accession_list, self.n_threads)
        return self.downloader

    def make_protein(self, accession, uniprot_protein):
        sequence = uniprot_protein.sequence
        name = "sp|{accession}|{gene_name} {description}".format(
            accession=accession, gene_name=uniprot_protein.gene_name,
            description=uniprot_protein.recommended_name)
        protein = Protein(
            name=name, protein_sequence=sequence,
            hypothesis_id=self.hypothesis_id)
        return protein

    def run(self):
        downloader = self._make_downloader()
        downloader.start()

        has_work = True

        while has_work:
            try:
                accession, value = downloader.get(True, 3)
                if isinstance(value, Exception):
                    self.log("Could not retrieve %s - %r" % (accession, value))
                else:
                    protein = self.make_protein(accession, value)
                    self.proteins.append(protein)
            except Empty:
                # If we haven't completed the download process, block
                # now and wait for the threads to join, then continue
                # trying to fetch results
                if not downloader.done_event.is_set():
                    downloader.join()
                    continue
                # Otherwise we've already waited for all the results to
                # arrive and we can stop iterating
                else:
                    has_work = False


class UniprotToPEFFTranslator(TaskBase):
    input_fasta_path: str
    output_path: str
    n_processes: int

    def __init__(self, input_fasta_path, output_path, n_processes: int=12) -> None:
        super().__init__()
        self.input_fasta_path = input_fasta_path
        self.output_path = output_path
        self.n_processes = n_processes

    def run(self):
        reader = fasta.ProteinFastaFileReader(self.input_fasta_path, index=True)

        accessions = []
        for key in reader.index.keys():
            acc = key.get("accession")
            if acc:
                accessions.append(acc)

        queue = UniprotProteinDownloader(accessions, self.n_processes)

        queue.start()

        header_block = fasta.PEFFHeaderBlock()
        header_block['Prefix'] = 'sp'
        header_block['SequenceType'] = 'AA'

        converter = UniprotToPeffConverter()

        with open(self.output_path, 'wb') as fh:
            writer = fasta.PEFFWriter(fh)
            writer.write_header([header_block])
            while True:
                try:
                    name, rec = queue.get()
                    try:
                        prot = converter(rec)
                        writer.write(prot)
                    except fasta.UnknownAminoAcidException as err:
                        self.log(f"Skipping {name}, failed to convert to PEFF: {err}")

                except Empty:
                    queue.join()
                    if queue.done_event.is_set():
                        break
