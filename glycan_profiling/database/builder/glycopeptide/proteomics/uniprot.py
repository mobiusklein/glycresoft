import threading

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

from glycopeptidepy.io import uniprot, fasta
# from glycopeptidepy.structure.residue import UnknownAminoAcidException

from glycan_profiling.task import TaskBase
from glycan_profiling.serialize import Protein


def get_uniprot_accession(name):
    try:
        return fasta.partial_uniprot_parser(name).accession
    except (AttributeError, fasta.UnparsableDeflineError):
        return None

# TODO: Transition ot use worker processes with multiple threads instead of just worker threads.
class UniprotProteinDownloader(TaskBase):
    def __init__(self, accession_list, n_threads=10):
        self.accession_list = accession_list
        self.n_threads = n_threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.no_more_work = threading.Event()
        self.done_event = threading.Event()
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

    def fetcher_task(self):
        has_work = True
        while has_work:
            try:
                accession_number = self.input_queue.get(True, 5)
                self.input_queue.task_done()
                self.fetch(accession_number)
            except Empty:
                if self.no_more_work.is_set():
                    has_work = False
                else:
                    continue

    def feeder_task(self):
        for i, item in enumerate(self.accession_list):
            if i % 100 == 0 and i:
                self.input_queue.join()
            self.input_queue.put(item)
        # self.input_queue.join()
        self.no_more_work.set()

    def run(self):
        feeder = threading.Thread(target=self.feeder_task)
        feeder.start()
        self.workers = workers = []
        for i in range(self.n_threads):
            t = threading.Thread(target=self.fetcher_task)
            t.daemon = True
            t.start()
            workers.append(t)

    def join(self):
        for worker in self.workers:
            if worker.is_alive():
                break
        else:
            self.done_event.set()
            return True
        return False

    def start(self):
        self.run()

    def get(self, blocking=True, timeout=3):
        return self.output_queue.get(blocking, timeout)


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
