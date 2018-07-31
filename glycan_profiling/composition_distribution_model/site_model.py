from collections import namedtuple

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition

from glycan_profiling import serialize
from glycan_profiling.task import TaskBase
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

    def __init__(self, glycan_graph, chromatogram_scorer=None, belongingness_matrix=None,
                 require_multiple_observations=True):
        if chromatogram_scorer is None:
            chromatogram_scorer = _default_chromatogram_scorer
        self.network = glycan_graph
        self.chromatogram_scorer = chromatogram_scorer
        self.belongingness_matrix = belongingness_matrix
        self.require_multiple_observations = require_multiple_observations
        if self.belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()
        self.target_site_models = []
        self.decoy_site_models = []

    def build_belongingness_matrix(self):
        network = self.network
        neighborhood_walker = NeighborhoodWalker(
            network, network.neighborhoods)

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

    def handle_glycoprotein(self, glycoprotein):
        self.log("Building Model for \"%s\"" % (glycoprotein.name, ))
        for i, site in enumerate(glycoprotein.site_map['N-Linked'].sites):
            gps_for_site = glycoprotein.site_map[
                'N-Linked'][glycoprotein.site_map['N-Linked'].sites[i]]
            gps_for_site = [
                gp for gp in gps_for_site if gp.chromatogram is not None]

            self.log('... %d Identified Glycopeptides At Site %d' %
                     (len(gps_for_site), site))

            glycopeptides = [
                gp for gp in gps_for_site if gp.chromatogram is not None]
            records = []
            for gp in glycopeptides:
                ms1_score = gp.ms1_score
                records.append(GlycanCompositionSolutionRecord(
                    gp.glycan_composition, ms1_score, gp.total_signal))

            learnable_cases = [rec for rec in records if rec.score > 0]

            if self.require_multiple_observations:
                agg = VariableObservationAggregation(self.network)
                agg.collect(learnable_cases)
                recs, var = agg.build_records()
                stable_cases = set([gc[0].glycan_composition for gc in filter(
                    lambda x: x[1] != 1.0, zip(recs, np.diag(var)))])
                self.log("... %d Stable Glycan Compositions" %
                         (len(stable_cases)))
                if len(stable_cases) == 0:
                    stable_cases = set([gc.glycan_composition for gc in recs])
                    self.log("... No Stable Cases Found. Using %d Glycan Compositions" % (
                        len(stable_cases), ))
                if len(stable_cases) == 0:
                    continue
            else:
                stable_cases = {
                    case.glycan_composition for case in learnable_cases}
            fitted_network, search_result, params = smooth_network(
                self.network, [
                    gp for gp in learnable_cases
                    if gp.score > 0 and gp.glycan_composition in stable_cases],
                belongingness_matrix=self.belongingness_matrix,
                observation_aggregator=AbundanceWeightedObservationAggregation)
            self.log("Lambda: %f" % (params.lmbda,))
            display_table([x.name for x in self.network.neighborhoods],
                          np.array(params.tau).reshape((-1, 1)))
            updated_params = params.clone()
            updated_params.lmbda = min(0.2, params.lmbda)
            fitted_network = search_result.annotate_network(updated_params)
            for node in fitted_network:
                if node.marked:
                    node.score *= 0.25
            self.target_site_models.append(
                GlycosylationSiteModel(glycoprotein.name, len(glycoprotein) - site - 1,
                                       dict(zip([x.name for x in self.network.neighborhoods],
                                                updated_params.tau.tolist())), updated_params.lmbda, {
                    str(node.glycan_composition): (node.score, not node.marked)
                    for node in fitted_network}))
            updated_params_decoy = params.clone()
            updated_params_decoy.tau[:] = updated_params_decoy.tau.mean()
            updated_params_decoy.lmbda = min(0.2, params.lmbda)
            fitted_network_decoy = search_result.annotate_network(
                updated_params_decoy)
            for node in fitted_network_decoy:
                # no decoy glycans are truly identified a priori, though the identified glycan compositions
                # will still carry a higher score from the estimation procedure
                node.score *= 0.25
            self.decoy_site_models.append(
                GlycosylationSiteModel(glycoprotein.name, len(glycoprotein) - site - 1,
                                       dict(zip([x.name for x in self.network.neighborhoods],
                                                updated_params_decoy.tau.tolist())), updated_params_decoy.lmbda, {
                    str(node.glycan_composition): (node.score, not node.marked)
                    for node in fitted_network_decoy}))
