from collections import defaultdict

from glypy.io import iupac, glycoct
from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenGlycanComposition
from glypy.enzyme import (
    make_n_glycan_pathway, make_mucin_type_o_glycan_pathway,
    MultiprocessingGlycome, Glycosylase, Glycosyltransferase,
    EnzymeGraph, GlycanCompositionEnzymeGraph, _enzyme_graph_inner)

from glycresoft.task import TaskBase, log_handle

from glycresoft.database.builder.glycan.glycan_source import (
    GlycanHypothesisSerializerBase, GlycanTransformer,
    DBGlycanComposition, formula, GlycanCompositionToClass,
    GlycanTypes)

from glycresoft.structure import KeyTransformingDecoratorDict


def key_transform(name):
    return str(name).lower().replace(" ", '-')


synthesis_register = KeyTransformingDecoratorDict(key_transform)


class MultiprocessingGlycomeTask(MultiprocessingGlycome, TaskBase):
    def _log(self, message):
        log_handle.log(message)

    def log_generation_chunk(self, i, chunks, current_generation):
        self._log(".... Task %d/%d finished (%d items generated)" % (
            i, len(chunks), len(current_generation)))


class GlycanSynthesis(TaskBase):
    glycan_classification = None

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
        logger = self.ipc_logger()
        glycome = self.build_glycome()
        old_logger = glycome._log
        glycome._log = logger.handler
        for i, gen in enumerate(glycome.run()):
            self.log(".... Generation %d: %d Structures" % (i, len(gen)))
        self.glycome = glycome
        logger.stop()
        glycome._log = old_logger
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


GlycanSynthesis.size_limiter_type = Limiter


@synthesis_register("n-glycan")
@synthesis_register("mammalian-n-glycan")
class NGlycanSynthesis(GlycanSynthesis):

    glycan_classification = GlycanTypes.n_glycan

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


@synthesis_register("human-n-glycan")
class HumanNGlycanSynthesis(NGlycanSynthesis):
    def _get_initial_components(self):
        glycosidases, glycosyltransferases, seeds = super(
            HumanNGlycanSynthesis, self)._get_initial_components()
        glycosyltransferases.pop('siagct2_6')
        glycosyltransferases.pop('agal13galt')
        glycosyltransferases.pop('gntE')
        return glycosidases, glycosyltransferases, seeds


@synthesis_register("mucin-o-glycan")
@synthesis_register("mammalian-mucin-o-glycan")
class MucinOGlycanSynthesis(GlycanSynthesis):

    glycan_classification = GlycanTypes.o_glycan

    def _get_initial_components(self):
        glycosidases, glycosyltransferases, seeds = make_mucin_type_o_glycan_pathway()
        parent = iupac.loads("a-D-Galp2NAc")
        child = iupac.loads("a-D-Neup5Gc")
        sgt6gal1 = Glycosyltransferase(6, 2, parent, child, terminal=False)
        glycosyltransferases['sgt6gal1'] = sgt6gal1

        parent = iupac.loads("b-D-Galp-(1-3)-a-D-Galp2NAc")
        child = iupac.loads("a-D-Neup5Gc")
        sgt3gal2 = Glycosyltransferase(3, 2, parent, child, parent_node_id=3)
        glycosyltransferases['sgt3gal2'] = sgt3gal2

        parent = iupac.loads("b-D-Galp-(1-3)-a-D-Galp2NAc")
        child = iupac.loads("a-D-Neup5Gc")
        sgt6gal2 = Glycosyltransferase(6, 2, parent, child, parent_node_id=3)
        glycosyltransferases['sgt6gal2'] = sgt6gal2
        return glycosidases, glycosyltransferases, seeds

    def __init__(self, glycosylases=None, glycosyltransferases=None, seeds=None, limits=None,
                 convert_to_composition=True, n_processes=5):
        sylases, transferases, more_seeds = self._get_initial_components()
        sylases.update(glycosylases or {})
        transferases.update(glycosyltransferases or {})
        more_seeds.extend(seeds or [])
        super(MucinOGlycanSynthesis, self).__init__(sylases, transferases, more_seeds,
                                                    limits, convert_to_composition, n_processes)


@synthesis_register("human-mucin-o-glycan")
class HumanMucinOGlycanSynthesis(MucinOGlycanSynthesis):
    def _get_initial_components(self):
        glycosidases, glycosyltransferases, seeds = super(HumanMucinOGlycanSynthesis)._get_initial_components()
        glycosyltransferases.pop("sgt6gal1")
        glycosyltransferases.pop("sgt3gal2")
        glycosyltransferases.pop("sgt6gal2")
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


class AdaptExistingGlycanGraph(TaskBase):
    def __init__(self, graph, enzymes_to_remove):
        self.graph = graph
        self.enzymes_to_remove = set(enzymes_to_remove)
        self.enzymes_available = set(self.graph.enzymes())
        if (self.enzymes_to_remove - self.enzymes_available):
            raise ValueError("Required enzymes %r not found" % (
                self.enzymes_to_remove - self.enzymes_available,))

    def remove_enzymes(self):
        enz_to_remove = self.enzymes_to_remove
        for enz in enz_to_remove:
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


class ExistingGraphGlycanHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, enzyme_graph, database_connection, enzymes_to_remove=None, reduction=None,
                 derivatization=None, hypothesis_name=None, glycan_classification=None):
        if enzymes_to_remove is None:
            enzymes_to_remove = set()

        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)

        self.enzyme_graph = enzyme_graph
        self.enzymes_to_remove = set(enzymes_to_remove)
        self.glycan_classification = glycan_classification

        self.reduction = reduction
        self.derivatization = derivatization

        self.loader = None
        self.transformer = None

    def build_glycan_compositions(self):
        adapter = AdaptExistingGlycanGraph(self.enzyme_graph, self.enzymes_to_remove)
        adapter.start()
        components = adapter.graph.nodes()
        for component in components:
            if isinstance(component, FrozenGlycanComposition):
                component = component.thaw()
            yield component, [self.glycan_classification]

    def make_pipeline(self):
        self.loader = self.build_glycan_compositions()
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def run(self):
        self.make_pipeline()
        structure_class_lookup = self.structure_class_loader

        acc = []
        counter = 0
        for composition, structure_classes in self.transformer:
            mass = composition.mass()
            composition_string = composition.serialize()
            formula_string = formula(composition.total_composition())
            inst = DBGlycanComposition(
                calculated_mass=mass, formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id)
            self.session.add(inst)
            self.session.flush()
            counter += 1
            for structure_class in structure_classes:
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if len(acc) % 100 == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
        if acc:
            self.session.execute(GlycanCompositionToClass.insert(), acc)
            acc = []
        self.session.commit()
        self.log("Generated %d glycan compositions" % counter)


class SynthesisGlycanHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, glycome, database_connection, reduction=None,
                 derivatization=None, hypothesis_name=None, glycan_classification=None):
        if glycan_classification is None:
            glycan_classification = glycome.glycan_classification

        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)

        self.glycome = glycome
        self.glycan_classification = glycan_classification

        self.reduction = reduction
        self.derivatization = derivatization

        self.loader = None
        self.transformer = None

    def build_glycan_compositions(self):
        components, enzyme_graph = self.glycome.start()
        for component in components:
            yield component, [self.glycan_classification]

    def make_pipeline(self):
        self.loader = self.build_glycan_compositions()
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def run(self):
        self.make_pipeline()
        structure_class_lookup = self.structure_class_loader

        acc = []
        counter = 0
        for composition, structure_classes in self.transformer:
            mass = composition.mass()
            composition_string = composition.serialize()
            formula_string = formula(composition.total_composition())
            inst = DBGlycanComposition(
                calculated_mass=mass, formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id)
            if (counter + 1) % 100 == 0:
                self.log("Stored %d glycan compositions" % counter)
            self.session.add(inst)
            self.session.flush()
            counter += 1
            for structure_class in structure_classes:
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if len(acc) % 100 == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
        if acc:
            self.session.execute(GlycanCompositionToClass.insert(), acc)
            acc = []
        self.session.commit()
        self.log("Stored %d glycan compositions" % counter)
