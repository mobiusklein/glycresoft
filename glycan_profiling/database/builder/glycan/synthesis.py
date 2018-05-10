# import glypy
import json
from collections import defaultdict, deque


from glypy.io import iupac, glycoct
from glypy.structure.glycan_composition import HashableGlycanComposition
from glypy.enzyme import (
    make_n_glycan_pathway, MultiprocessingGlycome,
    Glycosylase, Glycosyltransferase, _enzyme_graph_inner)

from glycan_profiling.task import TaskBase, log_handle


class MultiprocessingGlycomeTask(MultiprocessingGlycome, TaskBase):
    def log(self, i, chunks, current_generation):
        log_handle.log(".... Task %d/%d finished (%d items generated)" % (
            i, len(chunks), len(current_generation)))


class GlycanSynthesis(TaskBase):
    def __init__(self, glycosylases=None, glycosyltransferases=None, seeds=None, limits=None,
                 convert_to_composition=True, n_processes=5):
        self.glycosylases = glycosylases or {}
        self.glycosyltransferases = glycosyltransferases or {}
        self.seeds = seeds or []
        self.limits = limits or []
        self.convert_to_composition = convert_to_composition
        self.n_processes = n_processes

    def remove_enzyme(self, enzyme_name):
        if enzyme_name in self.glycosylases:
            return self.glycosylases.pop(enzyme_name)
        elif enzyme_name in self.glycosyltransferases:
            return self.glycosyltransferases.pop(enzyme_name)
        else:
            raise KeyError(enzyme_name)

    def add_enzyme(self, enzyme_name, enzyme):
        if isinstance(enzyme, Glycosylase):
            self.glycosylases[enzyme_name] = enzyme
        elif isinstance(enzyme, Glycosyltransferase):
            self.glycosyltransferases[enzyme_name] = enzyme
        else:
            raise TypeError("Don't know where to put object of type %r" % type(enzyme))

    def add_limit(self, limit):
        self.limits.append(limit)

    def add_seed(self, structure):
        if structure in self.seeds:
            return
        self.seeds.append(structure)

    def build_glycome(self):
        glycome = MultiprocessingGlycomeTask(
            self.glycosylases, self.glycosyltransferases,
            self.seeds, track_generations=False,
            limits=self.limits, processes=self.n_processes)
        return glycome

    def convert_enzyme_graph_composition(self, glycome):
        self.log("Converting Enzyme Graph into Glycan Set")
        glycans = set()
        glycans.update(glycome.enzyme_graph)
        for i, v in enumerate(glycome.enzyme_graph.values()):
            if i and i % 100000 == 0:
                self.log(".... %d Glycans In Set" % (len(glycans)))
            glycans.update(v)
        self.log(".... %d Glycans In Set" % (len(glycans)))

        composition_graph = defaultdict(_enzyme_graph_inner)
        compositions = set()

        cache = StructureConverter()

        i = 0
        for s in glycans:
            i += 1
            gc = cache[s]
            for child, enz in glycome.enzyme_graph[s].items():
                composition_graph[gc][cache[child]].update(enz)
            if i % 1000 == 0:
                self.log(".... Converted %d Compositions (%d/%d Structures, %0.2f%%)" % (
                    len(compositions), i, len(glycans), float(i) / len(glycans) * 100.0))
            compositions.add(gc)
        return compositions, composition_graph

    def extract_structures(self, glycome):
        self.log("Converting Enzyme Graph into Glycan Set")
        solutions = list()
        for i, structure in enumerate(glycome.seen):
            if i and i % 10000 == 0:
                self.log(".... %d Glycans Extracted" % (i,))
            solutions.append(structure)
        return solutions, glycome.enzyme_graph

    def run(self):
        glycome = self.build_glycome()
        for i, gen in enumerate(glycome.run()):
            self.log(".... Generation %d: %d Structures" % (i, len(gen)))
        if self.convert_to_composition:
            compositions, composition_enzyme_graph = self.convert_enzyme_graph_composition(glycome)
            return compositions, composition_enzyme_graph
        else:
            structures, enzyme_graph = self.extract_structures(glycome)
            return structures, enzyme_graph


class Limiter(object):
    def __init__(self, max_nodes=26, max_mass=5500.0):
        self.max_mass = max_mass
        self.max_nodes = max_nodes

    def __call__(self, x):
        return len(x) < self.max_nodes and x.mass() < self.max_mass


class NGlycanSynthesis(GlycanSynthesis):
    def _get_initial_components(self):
        glycosidases, glycosyltransferases, seeds = make_n_glycan_pathway()
        glycosyltransferases.pop('siat2_3')

        child = iupac.loads("a-D-Neup5Gc")
        parent = iupac.loads("b-D-Galp-(1-4)-b-D-Glcp2NAc")
        siagct2_6 = Glycosyltransferase(6, 2, parent, child, parent_node_id=3)

        glycosyltransferases['siagct2_6'] = siagct2_6
        return glycosidases, glycosyltransferases, seeds

    def __init__(self, glycosylases=None, glycosyltransferases=None, seeds=None, limits=None,
                 convert_to_composition=True, n_processes=5):
        sylases, transferases, more_seeds = self._get_initial_components()
        sylases.update(glycosylases or {})
        transferases.update(glycosyltransferases or {})
        more_seeds.extend(seeds or [])
        super(NGlycanSynthesis, self).__init__(sylases, transferases, more_seeds,
                                               limits, convert_to_composition, n_processes)


class HumanNGlycanSynthesis(NGlycanSynthesis):
    def _get_initial_components(self):
        glycosidases, glycosyltransferases, seeds = super(
            HumanNGlycanSynthesis, self)._get_initial_components()
        glycosyltransferases.pop('siagct2_6')
        glycosyltransferases.pop('agal13galt')
        glycosyltransferases.pop('gntE')
        return glycosidases, glycosyltransferases, seeds


class StructureConverter(object):
    def __init__(self):
        self.cache = dict()

    def convert(self, structure_text):
        if structure_text in self.cache:
            return self.cache[structure_text]
        structure = glycoct.loads(structure_text)
        gc = HashableGlycanComposition.from_glycan(structure).thaw()
        gc.drop_stems()
        gc.drop_configurations()
        gc.drop_positions()
        gc = HashableGlycanComposition(gc)
        self.cache[structure_text] = gc
        return gc

    def __getitem__(self, structure_text):
        return self.convert(structure_text)

    def __repr__(self):
        return "%s(%d)" % (self.__class__.__name__, len(self.cache))


class EnzymeGraph(object):
    def __init__(self, graph, seeds=None):
        self.graph = graph
        self.seeds = set()
        if seeds is None:
            seeds = self.parentless()
        self.seeds.update(seeds)

    def clone(self):
        graph = defaultdict(_enzyme_graph_inner)
        for outer_key, outer_value in self.graph.items():
            for inner_key, inner_value in outer_value.items():
                graph[outer_key][inner_key] = inner_value.copy()
        return self.__class__(graph, self.seeds.copy())

    def node_count(self):
        acc = set()
        acc.update(self.graph)
        for i, v in enumerate(self.graph.values()):
            acc.update(v)
        return len(acc)

    def edge_count(self):
        edges = 0
        for outer_key, outer_value in self.graph.items():
            for inner_key, inner_value in outer_value.items():
                edges += len(inner_value)
        return edges

    def __repr__(self):
        return "{}({:d})".format(self.__class__.__name__, self.node_count())

    def enzymes(self):
        enzyme_set = set()
        for outer_key, outer_value in self.graph.items():
            for inner_key, inner_value in outer_value.items():
                enzyme_set.update(inner_value)
        return enzyme_set

    def remove_enzyme(self, enzyme):
        edges_removed = list()
        for outer_key, outer_value in list(self.graph.items()):
            for inner_key, inner_value in list(outer_value.items()):
                if enzyme in inner_value:
                    inner_value.remove(enzyme)
                    edges_removed.append((outer_key, inner_key))
                if not inner_value:
                    outer_value.pop(inner_key)
                if not outer_value:
                    self.graph.pop(outer_key)
        # nodes_to_remove = self.parentless() - self.seeds
        # for node in nodes_to_remove:
        #     self.remove(node)
        return edges_removed

    def parents(self, target):
        parents = []
        for outer_key, outer_value in self.graph.items():
            for inner_key, inner_value in outer_value.items():
                if inner_key == target:
                    parents.append(outer_key)
        return parents

    def parentless(self):
        is_parent = set(self.graph)
        is_parented = set()
        for i, v in enumerate(self.graph.values()):
            is_parented.update(v)
        return is_parent - is_parented

    def children(self, target):
        children = []
        children.extend(self.graph[target])
        return children

    def remove(self, prune):
        items = deque([prune])
        i = 0
        while items:
            node = items.popleft()
            print("Removing %s" % (node,))
            if node in self.graph:
                i += 1
                children = self.graph.pop(node)
                items.extend(children)
        return i

    def _dump(self):
        data_structure = {
            "seeds": [str(sd) for sd in self.seeds],
            "enzymes": list(self.enzymes()),
            "graph": {}
        }
        outgraph = {}
        for outer_key, outer_value in self.graph.items():
            outgraph_inner = dict()
            for inner_key, inner_value in outer_value.items():
                outgraph_inner[str(inner_key)] = list(inner_value)
            outgraph[str(outer_key)] = outgraph_inner
        data_structure['graph'] = outgraph
        return data_structure

    def dump(self, fh):
        d = self._dump()
        json.dump(d, fh, sort_keys=True, indent=2)

    def dumps(self):
        d = self._dump()
        return json.dumps(d, sort_keys=True, indent=2)

    @classmethod
    def _load_entity(self, entity):
        return entity

    @classmethod
    def _load(cls, data_structure):
        seeds = {cls._load_entity(sd) for sd in data_structure["seeds"]}
        graph = dict()
        for outer_key, outer_value in data_structure["graph"].items():
            outgraph_inner = dict()
            for inner_key, inner_value in outer_value.items():
                outgraph_inner[cls._load_entity(inner_key)] = set(inner_value)
            graph[cls._load_entity(outer_key)] = outgraph_inner
        inst = cls(graph, seeds)
        return inst

    @classmethod
    def loads(cls, text):
        data = json.loads(text)
        return cls._load(data)

    @classmethod
    def load(cls, fd):
        data = json.load(fd)
        return cls._load(data)

    def __eq__(self, other):
        return self.graph == other.graph

    def __ne__(self, other):
        return self.graph != other.graph


# This may be too memory intensive to use on large graphs because
# a single :class:`Glycan` instance uses many times the memory that
# a :class:`GlycanComposition` does.
class GlycanStructureEnzymeGraph(EnzymeGraph):

    @classmethod
    def _load_entity(self, entity):
        return glycoct.loads(entity)


class GlycanCompositionEnzymeGraph(EnzymeGraph):

    @classmethod
    def _load_entity(self, entity):
        return HashableGlycanComposition.parse(entity)


class AdaptExistingGlycanGraph(TaskBase):
    def __init__(self, graph, enzymes_to_use):
        self.graph = graph
        self.enzymes_to_use = set(enzymes_to_use)
        self.enzymes_available = set(self.graph.enzymes())
        if (self.enzymes_to_use - self.enzymes_available):
            raise ValueError("Required enzymes %r not found" % (
                self.enzymes_to_use - self.enzymes_available,))

    def remove_enzymes(self):
        for enz in self.enzymes_to_use:
            self.log(".... Removing Enzyme %s" % (enz,))
            self.graph.remove_enzyme(enz)
            for entity in (self.graph.parentless() - self.graph.seeds):
                self.graph.remove(entity)

    def run(self):
        self.log("Adapting Enzyme Graph with %d nodes and %d edges" % (
            self.graph.node_count(), self.graph.edge_count()))
        self.remove_enzymes()
        self.log("After Adaption, Graph has %d nodes and %d edges" % (
            self.graph.node_count(), self.graph.edge_count()))
