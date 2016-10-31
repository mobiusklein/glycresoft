import requests
import gzip

from lxml import etree

from glypy.utils import StringIO
from glypy.io import glycoct
from glypy.composition import glycan_composition
from glypy.algorithms import subtree_search
from glypy import motifs

from glycopeptidepy import HashableGlycanComposition

from glycan_profiling.task import TaskBase


o_motifs = ([motifs[key] for key in motifs.keys() if "O-Glycan core" in key and "fuzzy" not in key.lower()])
n_motif = motifs["N-Glycan core basic 1"]


def is_n_glycan(structure):
    return subtree_search.subtree_of(
        n_motif, structure) == 1


def is_o_glycan(structure):
    for motif in o_motifs:
        if subtree_search.subtree_of(motif, structure):
            return True
    return False


class GlycomeDBLoader(TaskBase):
    def __init__(self, database_dump_handle=None):
        self.record_counter = 0
        self.taxa = set()
        self.misses = []
        self.database_dump_handle = database_dump_handle

    def get_database_dump(self):
        response = requests.get(u'http://www.glycome-db.org/http-services/getStructureDump.action?user=eurocarbdb')
        response.raise_for_status()
        handle = gzip.GzipFile(fileobj=StringIO(response.content))
        return handle

    def parse_all_structures(self, handle=None):
        if handle is None:
            handle = self.get_database_dump()
        xml = etree.parse(handle)

        longest_string = 0

        for structure in xml.iterfind(".//structure"):
            self.record_counter += 1
            try:
                glycomedb_id = int(structure.attrib['id'])
                glycoct_str = structure.find("sequence").text
                size = len(glycoct_str)
                if longest_string < size:
                    longest_string = size
                taxa = [int(t.attrib['ncbi']) for t in structure.iterfind(".//taxon")]
                glycan = glycoct.loads(glycoct_str)
                if (glycoct.loads(str(glycan)).mass() - glycan.mass()) > 0.00001:
                    raise Exception("Mass did not match on reparse")
                yield glycan, taxa, glycomedb_id
            except Exception as e:
                self.misses.append((glycomedb_id, e))

    def run(self):
        compositions = set()
        cases = []
        taxa = {}
        for glycan, taxa, glycomedb_id in self.parse_all_structures(self.database_dump_handle):
            inst = glycan_composition.GlycanComposition.from_glycan(glycan)
            inst.drop_stems()
            n_glycan = is_n_glycan(glycan)
            o_glycan = is_o_glycan(glycan)

            if self.record_counter % 100 == 0:
                self.log("Processed %d records (%r)\n%s, %r" % (
                    self.record_counter, taxa, inst,
                    [glycomedb_id, n_glycan, o_glycan]))

            string = str(inst)
            if string in compositions:
                continue
            compositions.add(string)
            cases.append(inst)
        return cases
