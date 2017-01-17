from collections import defaultdict

from glycopeptidepy.structure import sequence

from glycan_profiling.chromatogram_tree.relation_graph import (
    ChromatogramGraph, ChromatogramGraphEdge, ChromatogramGraphNode)


class GlycopeptideChromatogramGraphNode(ChromatogramGraphNode):
    def __init__(self, chromatogram, index, edges=None):
        super(GlycopeptideChromatogramGraphNode, self).__init__(chromatogram, index, edges)
        self.backbone = None
        if chromatogram.composition:
            if hasattr(chromatogram, 'entity'):
                entity = chromatogram.entity
                self.backbone = entity.get_sequence(include_glycan=False)


class GlycopeptideChromatogramGraph(ChromatogramGraph):
    def __init__(self, chromatograms):
        self.sequence_map = defaultdict(list)
        super(GlycopeptideChromatogramGraph, self).__init__(chromatograms)

    def _construct_graph_nodes(self, chromatograms):
        nodes = []
        for i, chroma in enumerate(chromatograms):
            node = (GlycopeptideChromatogramGraphNode(chroma, i))
            nodes.append(node)
            if node.chromatogram.composition:
                self.enqueue_seed(node)
                self.assignment_map[node.chromatogram.composition] = node
                if node.backbone is not None:
                    self.sequence_map[node.backbone].append(node)
        return nodes

    def find_edges(self, node, query_width=2., transitions=None, **kwargs):
        super(GlycopeptideChromatogramGraph, self).find_edges(node, query_width, transitions)
        if node.backbone is not None:
            for other_node in self.sequence_map[node.backbone]:
                ppm_error = 0
                rt_error = node.center - other_node.center
                ChromatogramGraphEdge(node, other_node, 'backbone', ppm_error, rt_error)
