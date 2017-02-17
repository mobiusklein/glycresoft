import unittest

import glypy
import pickle
from glycan_profiling.database import composition_network

compositions = [
    glypy.glycan_composition.FrozenGlycanComposition(HexNAc=2, Hex=i) for i in range(3, 10)
] + [
    glypy.glycan_composition.FrozenGlycanComposition(HexNAc=4, Hex=5, NeuAc=2),
    glypy.glycan_composition.FrozenGlycanComposition(HexNAc=3, Hex=5),
]


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


if __name__ == '__main__':
    unittest.main()
