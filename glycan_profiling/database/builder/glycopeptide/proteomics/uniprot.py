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


class UniprotProteinDownloader(object):
    def __init__(self, accession_list, n_threads=4):
        self.accession_list = accession_list
        self.n_threads = n_threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.no_more_work = threading.Event()
        self.done_event = threading.Event()
        self.workers = []

    def fetch(self, accession_number):
        try:
            protein_data = uniprot.get(accession_number)
            self.output_queue.put((accession_number, protein_data))
        except Exception as e:
            self.output_queue.put((accession_number, e))

    def fetcher_task(self):
        has_work = True
        while has_work:
            try:
                accession_number = self.input_queue.get(True, 5)
                self.fetch(accession_number)
            except Empty:
                if self.no_more_work.is_set():
                    has_work = False
                else:
                    continue

    def feeder_task(self):
        for item in self.accession_list:
            self.input_queue.put(item)
        self.no_more_work.set()

    def run(self):
        feeder = threading.Thread(target=self.feeder_task)
        feeder.start()
        workers = []
        for i in range(self.n_threads):
            t = threading.Thread(target=self.fetcher_task)
            t.daemon = True
            t.start()
            workers.append(t)
        self.workers = workers

    def join(self):
        for worker in self.workers:
            worker.join()
        self.done_event.set()

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
