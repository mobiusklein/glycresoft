import unittest

import glypy
import pickle

from textwrap import dedent
from io import BytesIO

from glycan_profiling.database import composition_network
from glycan_profiling.database.builder.glycan import constrained_combinatorics

compositions = [
    glypy.glycan_composition.FrozenGlycanComposition(HexNAc=2, Hex=i) for i in range(3, 10)
] + [
    glypy.glycan_composition.FrozenGlycanComposition(HexNAc=4, Hex=5, NeuAc=2),
    glypy.glycan_composition.FrozenGlycanComposition(HexNAc=3, Hex=5),
]


class NeighborhoodWalkerTest(unittest.TestCase):
    def make_human_definition_buffer(self):
        text = b'''Hex 3 10
HexNAc 2 9
Fuc 0 5
NeuAc 0 5

Fuc < HexNAc
HexNAc > NeuAc + 1'''
        buff = BytesIO()
        buff.write(text)
        buff.seek(0)
        return buff

    def make_mammalian_definition_buffer(self):
        text = b'''Hex 3 10
HexNAc 2 9
Fuc 0 5
NeuAc 0 5
NeuGc 0 5

Fuc < HexNAc
HexNAc > (NeuAc + NeuGc) + 1'''
        buff = BytesIO()
        buff.write(text)
        buff.seek(0)
        return buff        

    def generate_compositions(self, rule_buffer):
        rules, constraints = constrained_combinatorics.parse_rules_from_file(rule_buffer)
        compositions = list(constrained_combinatorics.CombinatoricCompositionGenerator(
            rules_table=rules, constraints=constraints))
        # strip out glycan classes
        compositions = [c[0] for c in compositions]
        return compositions

    def test_human_network(self):
        rules_buffer = self.make_human_definition_buffer()
        compositions = self.generate_compositions(rules_buffer)
        g = composition_network.CompositionGraph(compositions)
        self.assertEqual(len(g), 1424)
        walker = composition_network.NeighborhoodWalker(g)
        
        neighborhood_sizes = {
        "tri-antennary": 164,
        "bi-antennary": 84,
        "asialo-bi-antennary": 96,
        "tetra-antennary": 252,
        "hybrid": 64,
        "over-extended": 170,
        "asialo-tri-antennary": 120,
        "penta-antennary": 336,
        "asialo-penta-antennary": 144,
        "high-mannose": 16,
        "asialo-tetra-antennary": 136}
        for k, v in neighborhood_sizes.items():
            self.assertEqual(v, len(walker.neighborhood_maps[k]), "%s had %d members, not %d" % (
                k, len(walker.neighborhood_maps[k]), v))

    def test_mammalian_network(self):
        rules_buffer = self.make_mammalian_definition_buffer()
        compositions = self.generate_compositions(rules_buffer)
        g = composition_network.CompositionGraph(compositions)
        self.assertEqual(len(g), 5096)

        walker = composition_network.NeighborhoodWalker(g)

        neighborhood_sizes = {
        'tri-antennary': 312,
        'bi-antennary': 120,
        'asialo-bi-antennary': 144,
        'tetra-antennary': 600,
        'hybrid': 92,
        'over-extended': 363,
        'asialo-tri-antennary': 180,
        'penta-antennary': 960,
        'asialo-penta-antennary': 216,
        'high-mannose': 16,
        'asialo-tetra-antennary': 204,
        }
        for k, v in neighborhood_sizes.items():
            self.assertEqual(v, len(walker.neighborhood_maps[k]), "%s had %d members, not %d" % (
                k, len(walker.neighborhood_maps[k]), v))




class CompositionGraphTest(unittest.TestCase):

    def test_construction(self):
        g = composition_network.CompositionGraph(compositions)
        self.assertTrue(len(g) == 9)
        self.assertTrue(len(g.edges) == 0)
        g.create_edges(1)
        self.assertTrue(len(g.edges) == 7)

    def test_index(self):
        g = composition_network.CompositionGraph(compositions)
        node = g["{Hex:5; HexNAc:2}"]
        self.assertTrue(node.glycan_composition["Hex"] == 5)
        self.assertTrue(node.glycan_composition["HexNAc"] == 2)
        i = node.index
        self.assertEqual(g[i], node)

    def test_bridge(self):
        g = composition_network.CompositionGraph(compositions)
        g.create_edges(1)
        for edge in g.edges:
            self.assertTrue(edge.order < 2)
        node_to_remove = g["{Hex:5; HexNAc:2}"]
        removed_edges = g.remove_node(node_to_remove)
        neighbors = [e[node_to_remove] for e in removed_edges]
        n_order_2_edges = 0
        for edge in g.edges:
            if edge.order == 2:
                self.assertTrue(edge.node1 in neighbors and edge.node2 in neighbors)
                n_order_2_edges += 1
        self.assertTrue(n_order_2_edges > 0)
        node_to_remove = g["{Hex:6; HexNAc:2}"]
        removed_edges = g.remove_node(node_to_remove)
        neighbors = [e[node_to_remove] for e in removed_edges]
        for edge in g.edges:
            if edge.order == 3:
                self.assertTrue(edge.node1 in neighbors and edge.node2 in neighbors)

    def test_pickle(self):
        g = composition_network.CompositionGraph(compositions)
        g.create_edges(1)
        self.assertEqual(g, pickle.loads(pickle.dumps(g)))

    def test_clone(self):
        g = composition_network.CompositionGraph(compositions)
        g.create_edges(1)
        self.assertEqual(g, g.clone())




if __name__ == '__main__':
    unittest.main()
