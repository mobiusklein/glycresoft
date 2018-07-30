from collections import namedtuple

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition

from glycan_profiling import serialize
from glycan_profiling.task import log_handle, TaskBase
from glycan_profiling.database import GlycanCompositionDiskBackedStructureDatabase
from glycan_profiling.database.composition_network import NeighborhoodWalker, make_n_glycan_neighborhoods
from glycan_profiling.tandem.glycopeptide.identified_structure import IdentifiedGlycoprotein
from glycan_profiling.composition_distribution_model import (
    smooth_network, display_table, VariableObservationAggregation,
    GlycanCompositionSolutionRecord,
    AbundanceWeightedObservationAggregation)
from glycan_profiling.models import GeneralScorer, get_feature


GlycanPriorRecord = namedtuple("GlycanPriorRecord", ("score", "matched"))
_default_chromatogram_scorer = GeneralScorer.clone()
_default_chromatogram_scorer.add_feature(get_feature("null_charge"))


class GlycosylationSiteModel(object):
    def __init__(self, protein_name, position, site_distribution, lmbda, glycan_map):
        self.protein_name = protein_name
        self.position = position
        self.site_distribution = site_distribution
        self.lmbda = lmbda
        self.glycan_map = glycan_map

    def __getitem__(self, key):
        return self.glycan_map[key][0]

    def to_dict(self):
        d = {}
        d['protein_name'] = self.protein_name
        d['position'] = self.position
        d['lmbda'] = self.lmbda
        d['site_distribution'] = dict(**self.site_distribution)
        d['glycan_map'] = {
            str(k): (v.score, v.matched) for k, v in self.glycan_map.items()
        }
        return d

    @classmethod
    def from_dict(cls, d):
        name = d['protein_name']
        position = d['position']
        lmbda = d['lmbda']
        site_distribution = d['site_distribution']
        glycan_map = d['glycan_map']
        glycan_map = {
            HashableGlycanComposition.parse(k): GlycanPriorRecord(v[0], v[1])
            for k, v in glycan_map.items()
        }
        inst = cls(name, position, site_distribution, lmbda, glycan_map)
        return inst


class GlycosylationSiteModelBuilder(TaskBase):
    def __init__(self, glycan_graph, chromatogram_scorer=None, belongingness_matrix=None):
        if chromatogram_scorer is None:
            chromatogram_scorer = _default_chromatogram_scorer
        self.network = glycan_graph
        self.chromatogram_scorer = chromatogram_scorer
        self.belongingness_matrix = belongingness_matrix
        if self.belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()

    def build_belongingness_matrix(self):
        network = self.network
        neighborhood_walker = NeighborhoodWalker(network, network.neighborhoods)

        neighborhood_count = len(neighborhood_walker.neighborhoods)
        belongingness_matrix = np.zeros(
            (len(network), neighborhood_count))

        for node in network:
            was_in = neighborhood_walker.neighborhood_assignments[node]
            for i, neighborhood in enumerate(neighborhood_walker.neighborhoods):
                if neighborhood.name in was_in:
                    belongingness_matrix[node.index, i] = neighborhood_walker.compute_belongingness(
                        node, neighborhood.name)
        return belongingness_matrix
