from collections import namedtuple, defaultdict
try:
    from collections.abc import Mapping, Sequence
except ImportError:
    from collections import Mapping, Sequence

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition
from glycopeptidepy.structure.parser import strip_modifications

from glycan_profiling import serialize
from glycan_profiling.task import TaskBase
from glycan_profiling.structure import PeptideProteinRelation

from glycan_profiling.database import GlycanCompositionDiskBackedStructureDatabase

from glycan_profiling.database.builder.glycopeptide.proteomics.fasta import DeflineSuffix
from glycan_profiling.database.builder.glycopeptide.proteomics.sequence_tree import SuffixTree

from glycan_profiling.tandem.glycopeptide.identified_structure import IdentifiedGlycoprotein

from glycan_profiling.database.composition_network import NeighborhoodWalker, make_n_glycan_neighborhoods
from glycan_profiling.composition_distribution_model import (
    smooth_network, display_table, VariableObservationAggregation,
    GlycanCompositionSolutionRecord,
    AbundanceWeightedObservationAggregation)
from glycan_profiling.models import GeneralScorer, get_feature


GlycanPriorRecord = namedtuple("GlycanPriorRecord", ("score", "matched"))
_default_chromatogram_scorer = GeneralScorer.clone()
_default_chromatogram_scorer.add_feature(get_feature("null_charge"))


MINIMUM = 1e-4


class GlycosylationSiteModel(object):

    def __init__(self, protein_name, position, site_distribution, lmbda, glycan_map):
        self.protein_name = protein_name
        self.position = position
        self.site_distribution = site_distribution
        self.lmbda = lmbda
        self.glycan_map = glycan_map

    def __getitem__(self, key):
        return self.glycan_map[key][0]

    def get_record(self, key):
        try:
            return self.glycan_map[key]
        except KeyError:
            return GlycanPriorRecord(MINIMUM, False)

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
        try:
            site_distribution = d['site_distribution']
        except KeyError:
            site_distribution = d['tau']
        glycan_map = d['glycan_map']
        glycan_map = {
            HashableGlycanComposition.parse(k): GlycanPriorRecord(v[0], v[1])
            for k, v in glycan_map.items()
        }
        inst = cls(name, position, site_distribution, lmbda, glycan_map)
        return inst

    def _pack(self):
        new_map = {}
        for key, value in self.glycan_map.items():
            if value.score > MINIMUM:
                new_map[key] = value
        self.glycan_map = new_map

    def __repr__(self):
        template = ('{self.__class__.__name__}({self.protein_name!r}, {self.position}, '
                    '{site_distribution}, {self.lmbda}, <{glycan_map_size} Glycans>)')
        glycan_map_size = len(self.glycan_map)
        site_distribution = {k: v for k, v in self.site_distribution.items() if v > 0.0}
        return template.format(self=self, glycan_map_size=glycan_map_size, site_distribution=site_distribution)


class GlycoproteinGlycosylationModel(object):
    def __init__(self, protein, glycosylation_sites=None):
        self.protein = protein
        self.glycosylation_sites = sorted(glycosylation_sites or [], key=lambda x: x.position)

    def __getitem__(self, i):
        return self.glycosylation_sites[i]

    def __len__(self):
        return len(self.glycosylation_sites)

    @property
    def id(self):
        return self.protein.id

    @property
    def name(self):
        return self.protein.name

    def find_sites_in(self, start, end):
        spans = []
        for site in self.glycosylation_sites:
            if start <= site.position <= end:
                spans.append(site)
            elif end < site.position:
                break
        return spans

    def _guess_sites_from_sequence(self, sequence):
        prot_seq = str(self.protein)
        query_seq = strip_modifications(sequence)
        try:
            start = prot_seq.index(query_seq)
            end = start + len(query_seq)
            return PeptideProteinRelation(start, end, self.protein.id, self.protein.hypothesis_id)
        except ValueError:
            return None

    def score(self, glycopeptide):
        pr = glycopeptide.protein_relation
        sites = self.find_sites_in(pr.start_position, pr.end_position)
        if len(sites) > 1:
            raise NotImplementedError("Not compatible with multiple spanning glycosites (yet)")
        try:
            site = sites[0]
            try:
                rec = site.glycan_map[glycopeptide.glycan_composition]
            except KeyError:
                return MINIMUM
            return rec.score
        except IndexError:
            return MINIMUM

    @classmethod
    def bind_to_hypothesis(cls, session, site_models, hypothesis_id=1, fuzzy=True):
        by_protein_name = defaultdict(list)
        for site in site_models:
            by_protein_name[site.protein_name].append(site)
        protein_models = {}
        proteins = session.query(serialize.Protein).filter(
            serialize.Protein.hypothesis_id == hypothesis_id).all()
        protein_name_map = {prot.name: prot for prot in proteins}
        if fuzzy:
            tree = SuffixTree()
            for prot in proteins:
                tree.add_ngram(DeflineSuffix(prot.name, prot.name))

        for protein_name, sites in by_protein_name.items():
            if fuzzy:
                labels = list(tree.subsequences_of(protein_name))
                protein = protein_name_map[labels[0].original]
            else:
                protein = protein_name_map[protein_name]

            model = cls(protein, sites)
            protein_models[model.id] = model
        return protein_models

    def __repr__(self):
        template = "{self.__class__.__name__}({self.name}, {self.glycosylation_sites})"
        return template.format(self=self)


class GlycoproteinSiteSpecificGlycomeModel(object):
    def __init__(self, glycoprotein_models):
        if isinstance(glycoprotein_models, Mapping):
            self.glycoprotein_models = glycoprotein_models
        else:
            self.glycoprotein_models = {
                ggm.id: ggm for ggm in glycoprotein_models
            }

    def find_model(self, glycopeptide):
        protein_id = glycopeptide.protein_relation.protein_id
        glycoprotein_model = self.glycoprotein_models[protein_id]
        return glycoprotein_model

    def score(self, spectrum_match):
        glycopeptide = spectrum_match.target
        glycoprotein_model = self.find_model(glycopeptide)
        score = glycoprotein_model.score(glycopeptide)
        return min(spectrum_match.score, score)

    @classmethod
    def bind_to_hypothesis(cls, session, site_models, hypothesis_id=1, fuzzy=True):
        inst = cls(
            GlycoproteinGlycosylationModel.bind_to_hypothesis(
                session, site_models, hypothesis_id, fuzzy))
        return inst


class GlycosylationSiteModelBuilder(TaskBase):

    def __init__(self, glycan_graph, chromatogram_scorer=None, belongingness_matrix=None,
                 unobserved_penalty_scale=0.25, lambda_limit=0.2, require_multiple_observations=True):
        if chromatogram_scorer is None:
            chromatogram_scorer = _default_chromatogram_scorer
        self.network = glycan_graph
        if not self.network.neighborhoods:
            self.network.neighborhoods = make_n_glycan_neighborhoods()
        self.chromatogram_scorer = chromatogram_scorer
        self.belongingness_matrix = belongingness_matrix
        self.require_multiple_observations = require_multiple_observations
        self.unobserved_penalty_scale = unobserved_penalty_scale
        self.lambda_limit = lambda_limit
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

    def add_glycoprotein(self, glycoprotein, evaluate_chromatograms=False):
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
                if evaluate_chromatograms:
                    ms1_score = self.chromatogram_scorer.logitscore(gp.chromatogram)
                else:
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
            updated_params.lmbda = min(self.lambda_limit, params.lmbda)
            fitted_network = search_result.annotate_network(updated_params)
            for node in fitted_network:
                if node.marked:
                    node.score *= self.unobserved_penalty_scale
            self.target_site_models.append(
                GlycosylationSiteModel(glycoprotein.name, site,
                                       dict(zip([x.name for x in self.network.neighborhoods],
                                                updated_params.tau.tolist())), updated_params.lmbda, {
                                                    str(node.glycan_composition): (node.score, not node.marked)
                                                    for node in fitted_network}))
            updated_params_decoy = params.clone()
            updated_params_decoy.tau[:] = updated_params_decoy.tau.mean()
            updated_params_decoy.lmbda = min(self.lambda_limit, params.lmbda)
            fitted_network_decoy = search_result.annotate_network(
                updated_params_decoy)
            for node in fitted_network_decoy:
                # no decoy glycans are truly identified a priori, though the identified glycan compositions
                # will still carry a higher score from the estimation procedure
                node.score *= self.unobserved_penalty_scale
            self.decoy_site_models.append(
                GlycosylationSiteModel(glycoprotein.name, len(glycoprotein) - site - 1,
                                       dict(zip([x.name for x in self.network.neighborhoods],
                                                updated_params_decoy.tau.tolist())), updated_params_decoy.lmbda, {
                    str(node.glycan_composition): (node.score, not node.marked)
                    for node in fitted_network_decoy}))
