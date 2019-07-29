import re
import json
import time
import warnings

from collections import namedtuple, defaultdict, deque
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from multiprocessing.pool import ThreadPool
from threading import RLock, Condition

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition
from glycopeptidepy.structure.parser import strip_modifications

from glycan_profiling import serialize
from glycan_profiling.task import TaskBase
from glycan_profiling.structure import PeptideProteinRelation, LRUMapping

from glycan_profiling.database import (
    GlycanCompositionDiskBackedStructureDatabase, GlycopeptideDiskBackedStructureDatabase)

from glycan_profiling.database.builder.glycopeptide.proteomics.fasta import DeflineSuffix
from glycan_profiling.database.builder.glycopeptide.proteomics.sequence_tree import SuffixTree

from glycan_profiling.database.composition_network import NeighborhoodWalker, make_n_glycan_neighborhoods
from glycan_profiling.composition_distribution_model import (
    smooth_network, display_table, VariableObservationAggregation,
    GlycanCompositionSolutionRecord, GlycomeModel)
from glycan_profiling.models import GeneralScorer, get_feature


GlycanPriorRecord = namedtuple("GlycanPriorRecord", ("score", "matched"))
_default_chromatogram_scorer = GeneralScorer.clone()
_default_chromatogram_scorer.add_feature(get_feature("null_charge"))

MINIMUM = 1e-4


glycan_composition_cache = dict()


def parse_glycan_composition(string):
    try:
        return glycan_composition_cache[string]
    except KeyError:
        gc = HashableGlycanComposition.parse(string)
        glycan_composition_cache[string] = gc
        return gc


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
            parse_glycan_composition(k): GlycanPriorRecord(v[0], v[1])
            for k, v in glycan_map.items()
        }
        inst = cls(name, position, site_distribution, lmbda, glycan_map)
        return inst

    def pack(self, inplace=True):
        if inplace:
            self._pack()
            return self
        return self.copy()._pack()

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

    def copy(self, deep=False):
        dup = self.__class__(
            self.protein_name, self.position, self.site_distribution, self.lmbda, self.glycan_map)
        if deep:
            dup.site_distribution = dup.site_distribution.copy()
            dup.glycan_map = dup.glycan_map.copy()
        return dup

    def clone(self, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def observed_glycans(self, threshold=0):
        return {k: v.score for k, v in self.glycan_map.items() if v.matched and v.score > threshold}


class GlycoproteinSiteSpecificGlycomeModel(object):
    def __init__(self, protein, glycosylation_sites=None):
        self.protein = protein
        self._glycosylation_sites = []
        self.glycosylation_sites = glycosylation_sites

    @property
    def glycosylation_sites(self):
        return self._glycosylation_sites

    @glycosylation_sites.setter
    def glycosylation_sites(self, glycosylation_sites):
        self._glycosylation_sites = sorted(glycosylation_sites or [], key=lambda x: x.position)

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
            score_acc = 0.0
            warnings.warn("Multiple glycosylation sites are not (yet) supported")
            for site in sites:
                try:
                    rec = site.glycan_map[glycopeptide.glycan_composition]
                    score_acc += (rec.score)
                except KeyError:
                    score_acc += (MINIMUM)
            return score_acc / len(sites)
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


class ReversedProteinSiteReflectionGlycoproteinSiteSpecificGlycomeModel(GlycoproteinSiteSpecificGlycomeModel):
    @property
    def glycosylation_sites(self):
        return self._glycosylation_sites

    @glycosylation_sites.setter
    def glycosylation_sites(self, glycosylation_sites):
        temp = []
        n = len(str(self.protein))
        for site in glycosylation_sites:
            site = site.copy()
            site.position = n - site.position - 1
            temp.append(site)
        self._glycosylation_sites = sorted(temp or [], key=lambda x: x.position)


class GlycoproteomeModel(object):
    def __init__(self, glycoprotein_models):
        if isinstance(glycoprotein_models, Mapping):
            self.glycoprotein_models = dict(glycoprotein_models)
        else:
            self.glycoprotein_models = {
                ggm.id: ggm for ggm in glycoprotein_models
            }

    def find_model(self, glycopeptide):
        if glycopeptide.protein_relation is None:
            return None
        protein_id = glycopeptide.protein_relation.protein_id
        glycoprotein_model = self.glycoprotein_models[protein_id]
        return glycoprotein_model

    def score(self, glycopeptide):
        glycoprotein_model = self.find_model(glycopeptide)
        if glycoprotein_model is None:
            score = MINIMUM
        else:
            score = glycoprotein_model.score(glycopeptide)
        return score

    @classmethod
    def bind_to_hypothesis(cls, session, site_models, hypothesis_id=1, fuzzy=True, site_model_cls=GlycoproteinSiteSpecificGlycomeModel):
        inst = cls(
            site_model_cls.bind_to_hypothesis(
                session, site_models, hypothesis_id, fuzzy))
        return inst


class SubstringGlycoproteomeModel(object):
    def __init__(self, models, cache_size=2**15):
        self.models = models
        self.sequence_to_model = {
            str(model.protein): model for model in models.values()
        }
        self.peptide_to_protein = LRUMapping(cache_size)

    def get_models(self, glycopeptide):
        out = []
        seq = strip_modifications(glycopeptide)
        if seq in self.peptide_to_protein:
            return list(self.peptide_to_protein[seq])
        pattern = re.compile(seq)
        for case in self.sequence_to_model:
            if seq in case:
                bounds = pattern.finditer(case)
                for match in bounds:
                    protein_model = self.sequence_to_model[case]
                    site_models = protein_model.find_sites_in(match.start(), match.end())
                    out.append(site_models)
        self.peptide_to_protein[seq] = tuple(out)
        return out

    def find_proteins(self, glycopeptide):
        out = []
        seq = strip_modifications(glycopeptide)
        for case in self.sequence_to_model:
            if seq in case:
                out.append(self.sequence_to_model[case])
        return out

    def score(self, glycopeptide):
        models = self.get_models(glycopeptide)
        if len(models) == 0:
            return MINIMUM
        sites = models[0]
        if len(sites) == 0:
            return MINIMUM
        try:
            acc = []
            for site in sites:
                try:
                    rec = site.glycan_map[glycopeptide.glycan_composition]
                    acc.append(rec.score)
                except KeyError:
                    pass
            return max(sum(acc) / len(acc), MINIMUM) if acc else MINIMUM
        except IndexError:
            return MINIMUM

    def __call__(self, glycopeptide):
        return self.get_models(glycopeptide)


def _truncate_name(name, limit=30):
    if len(name) > limit:
        return name[:limit - 3] + '...'
    return name


class GlycosylationSiteModelBuilder(TaskBase):
    _timeout_per_unit = 300

    def __init__(self, glycan_graph, chromatogram_scorer=None, belongingness_matrix=None,
                 unobserved_penalty_scale=None, lambda_limit=0.2,
                 require_multiple_observations=True,
                 observation_aggregator=None, n_threads=4):
        if observation_aggregator is None:
            observation_aggregator = VariableObservationAggregation
        if chromatogram_scorer is None:
            chromatogram_scorer = _default_chromatogram_scorer
        if unobserved_penalty_scale is None:
            unobserved_penalty_scale = 1.0

        self.network = glycan_graph
        if not self.network.neighborhoods:
            self.network.neighborhoods = make_n_glycan_neighborhoods()

        self.chromatogram_scorer = chromatogram_scorer
        self.belongingness_matrix = belongingness_matrix
        self.observation_aggregator = observation_aggregator
        self.require_multiple_observations = require_multiple_observations
        self.unobserved_penalty_scale = unobserved_penalty_scale
        self.lambda_limit = lambda_limit

        if self.belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()

        self.site_models = []
        self.n_threads = n_threads
        self.thread_pool = ThreadPool(n_threads)
        self._lock = RLock()
        self._concurrent_jobs = 0

    def build_belongingness_matrix(self):
        network = self.network
        neighborhood_walker = NeighborhoodWalker(
            network, network.neighborhoods)
        belongingness_matrix = neighborhood_walker.build_belongingness_matrix()
        return belongingness_matrix

    def _transform_glycopeptide(self, glycopeptide, evaluate_chromatograms=False):
        gp = glycopeptide
        if evaluate_chromatograms:
            ms1_score = self.chromatogram_scorer.logitscore(gp.chromatogram)
        else:
            ms1_score = gp.ms1_score
        return GlycanCompositionSolutionRecord(gp.glycan_composition, ms1_score, gp.total_signal)

    def add_glycoprotein(self, glycoprotein, evaluate_chromatograms=False):
        async_results = []
        sites_to_log = []
        for i, site in enumerate(glycoprotein.site_map['N-Linked'].sites):
            gps_for_site = glycoprotein.site_map[
                'N-Linked'][glycoprotein.site_map['N-Linked'].sites[i]]
            gps_for_site = [
                gp for gp in gps_for_site if gp.chromatogram is not None]

            self.log('... %d Identified Glycopeptides At Site %d for %s' %
                     (len(gps_for_site), site, _truncate_name(glycoprotein.name, )))

            glycopeptides = [
                gp for gp in gps_for_site if gp.chromatogram is not None]
            records = []
            for gp in glycopeptides:
                records.append(self._transform_glycopeptide(gp, evaluate_chromatograms))

            if self.n_threads == 1:
                self.fit_site_model(records, site, glycoprotein)
            else:
                sites_to_log.append(site)
                with self._lock:
                    self._concurrent_jobs += 1
                    async_results.append(
                        self.thread_pool.apply_async(self.fit_site_model, (records, site, glycoprotein, )))
        if async_results:
            time.sleep(20)
            for i, result in enumerate(async_results):
                if not result.ready():
                    self.log("... Waiting for Result from Site %d of %s" % (
                        sites_to_log[i], _truncate_name(glycoprotein.name, )))
                result.get(self._timeout_per_unit * self.n_threads + 20)
                with self._lock:
                    self._concurrent_jobs -= 1
            with self._lock:
                self.log("... Finished Fitting %s, %d Tasks Pending" % (
                    _truncate_name(glycoprotein.name), self._concurrent_jobs))

    def _get_learnable_cases(self, observations):
        learnable_cases = [rec for rec in observations if rec.score > 1]
        if not learnable_cases:
            return []
        if self.require_multiple_observations:
            agg = VariableObservationAggregation(self.network)
            agg.collect(learnable_cases)
            recs, var = agg.build_records()
            # TODO: Rewrite to avoid using VariableObservationAggregation because calculation
            #       of the variance matrix is expensive.
            #
            # Use VariableObservationAggregation algorithm to collect the glycan
            # composition observations according to the network definition of multiple
            # observations, and then extract the observed indices along the diagonal
            # of the variance matrix.
            #
            # Those indices which are equal to 1.0 are those
            # where the glycan composition was only observed once, according to the
            # transformation VariableObservationAggregation carries out when estimating
            # the variance of each glycan composition observed.
            rec_variance = np.diag(var.variance_matrix)[var.observed_indices]
            stable_cases = set([gc.glycan_composition for gc, v in zip(recs, rec_variance) if v != 1.0])
            self.log("... %d Stable Glycan Compositions" % (
                len(stable_cases)))
            if len(stable_cases) == 0:
                stable_cases = set([gc.glycan_composition for gc in recs])
                self.log("... No Stable Cases Found. Using %d Glycan Compositions" % (
                    len(stable_cases), ))
            if len(stable_cases) == 0:
                return []
        else:
            stable_cases = {
                case.glycan_composition for case in learnable_cases}
        learnable_cases = [
            rec for rec in learnable_cases
            if rec.score > 1 and rec.glycan_composition in stable_cases
        ]
        return learnable_cases

    def fit_site_model(self, observations, site, glycoprotein):
        learnable_cases = self._get_learnable_cases(observations)

        if not learnable_cases:
            return None

        acc = defaultdict(list)
        for case in learnable_cases:
            acc[case.glycan_composition].append(case)
        with self._lock:
            self.log("... %d Glycan Compositions for Site %d of %s" % (
                len(acc), site, _truncate_name(glycoprotein.name, )))
            for key, value in sorted(acc.items(), key=lambda x: x[0].mass()):
                self.log("... %s: [%s]" % (
                    key,
                    ', '.join(["%0.2f" % f for f in sorted([r.score for r in value])])
                ))

        fitted_network, search_result, params = smooth_network(
            self.network, learnable_cases,
            belongingness_matrix=self.belongingness_matrix,
            observation_aggregator=VariableObservationAggregation,
            annotate_network=False)
        if params is None:
            self.log("Skipping Site %d of %s" % (site, _truncate_name(glycoprotein.name) ))
            return
        with self._lock:
            self.log("... Site %d of %s Lambda: %f" % (site, _truncate_name(glycoprotein.name), params.lmbda,))
            display_table([x.name for x in self.network.neighborhoods],
                          np.array(params.tau).reshape((-1, 1)))
        updated_params = params.clone()
        updated_params.lmbda = min(self.lambda_limit, params.lmbda)
        self.log("... Projecting Solution Onto Network for Site %d of %s" %
                 (site, _truncate_name(glycoprotein.name)))
        fitted_network = search_result.annotate_network(updated_params)
        for node in fitted_network:
            if node.marked:
                node.score *= self.unobserved_penalty_scale

        site_distribution = dict(zip([x.name for x in self.network.neighborhoods], updated_params.tau.tolist()))
        glycan_map = {
            str(node.glycan_composition): GlycanPriorRecord(node.score, not node.marked)
            for node in fitted_network
        }
        site_model = GlycosylationSiteModel(
            glycoprotein.name,
            site,
            site_distribution,
            updated_params.lmbda,
            glycan_map)
        with self._lock:
            self.site_models.append(site_model.pack())
        return site_model

    def save_models(self, path):
        with open(path, 'wt') as fh:
            prepared = []
            for site in sorted(self.site_models, key=lambda x: (x.protein_name, x.position)):
                prepared.append(site.to_dict())
            json.dump(prepared, fh)


class GlycoproteinSiteModelBuildingWorkflow(TaskBase):
    _timeout_per_unit_site = 300

    def __init__(self, analyses, glycopeptide_database, glycan_database,
                 unobserved_penalty_scale=None, lambda_limit=0.2,
                 require_multiple_observations=True, observation_aggregator=None,
                 output_path=None, n_threads=4):
        if observation_aggregator is None:
            observation_aggregator = VariableObservationAggregation
        if unobserved_penalty_scale is None:
            unobserved_penalty_scale = 1.0

        self.analyses = analyses
        self.glycopeptide_database = glycopeptide_database
        self.glycan_database = glycan_database

        self.unobserved_penalty_scale = unobserved_penalty_scale
        self.lambda_limit = lambda_limit

        self.require_multiple_observations = require_multiple_observations
        self.observation_aggregator = observation_aggregator

        self.output_path = output_path

        self.n_threads = n_threads
        self.thread_pool = ThreadPool(self.n_threads)
        self._lock = RLock()
        self._count_barrier = Condition()
        self._concurrent_jobs = 0

    @classmethod
    def from_paths(cls, analysis_paths_and_ids, glycopeptide_hypothesis_path, glycopeptide_hypothesis_id,
                   glycan_hypothesis_path, glycan_hypothesis_id, unobserved_penalty_scale=None,
                   lambda_limit=0.2, require_multiple_observations=True, observation_aggregator=None,
                   output_path=None, n_threads=4):
        gp_db = GlycopeptideDiskBackedStructureDatabase(
            glycopeptide_hypothesis_path, glycopeptide_hypothesis_id)
        gc_db = GlycanCompositionDiskBackedStructureDatabase(
            glycan_hypothesis_path, glycan_hypothesis_id)

        analyses = [serialize.AnalysisDeserializer(conn, analysis_id=an_id)
                    for conn, an_id in analysis_paths_and_ids]
        inst = cls(
            analyses, gp_db, gc_db, unobserved_penalty_scale=unobserved_penalty_scale,
            lambda_limit=lambda_limit, require_multiple_observations=require_multiple_observations,
            observation_aggregator=observation_aggregator, output_path=output_path,
            n_threads=n_threads,
        )
        return inst

    def make_glycan_network(self):
        network = self.glycan_database.glycan_composition_network
        network = network.augment_with_decoys()
        network.create_edges()

        model = GlycomeModel([], network)
        belongingness_matrix = model.belongingness_matrix
        network.neighborhoods = model.neighborhood_walker.neighborhoods
        return network, belongingness_matrix

    def load_identified_glycoproteins_from_analysis(self, analysis):
        if not isinstance(analysis, serialize.Analysis):
            analysis = analysis.analysis
        idgps = analysis.aggregate_identified_glycoproteins()
        return idgps

    def build_reference_protein_map(self):
        proteins = list(self.glycopeptide_database.proteins)
        index = {p.name: p for p in proteins}
        return index

    def aggregate_identified_glycoproteins(self):
        acc = defaultdict(list)
        n = len(self.analyses)
        for i, analysis in enumerate(self.analyses, 1):
            self.log("... Loading Glycopeptides for %s (%d/%d)" % (analysis.name, i, n))
            for idgp in self.load_identified_glycoproteins_from_analysis(analysis):
                acc[idgp.name].append(idgp)
        result = []
        self.log("Merging Glycoproteins Across Replicates")
        n = float(len(acc))
        i = 1.0
        last = 0.1
        should_log = False
        for name, duplicates in acc.items():
            if i / n > last:
                should_log = True
                last += 0.1
            if i % 100 == 0:
                should_log = True
            if should_log:
                self.log("... Merging %s (%d/%d, %0.2f%%)" % (name, i, n, i * 100.0 / n))
                should_log = False
            agg = duplicates[0]
            result.append(agg.merge(*duplicates[1:]))
            i += 1
        return result

    def thread_pool_saturated(self, ratio=1.0):
        with self._lock:
            jobs = self._concurrent_jobs
        return (jobs / float(self.n_threads)) >= ratio

    def _add_glycoprotein(self, glycoprotein, builder, k_sites):
        # Acquire the condition lock, then wait until the thread pool is empty
        # enough to do some work, then release the condition lock and do the work
        self._count_barrier.acquire()
        while self.thread_pool_saturated():
            self._count_barrier.wait()
        self._count_barrier.release()

        with self._lock:
            self._concurrent_jobs += k_sites

        # This should block until all site model fitting finishes.
        builder.add_glycoprotein(glycoprotein)

        with self._lock:
            self._concurrent_jobs -= k_sites

        # Acquire the condition lock, wake up the next thread waiting, and release
        # the condition lock
        self._count_barrier.acquire()
        self._count_barrier.notify()
        self._count_barrier.release()

    def _fit_glycoprotein_site_models(self, glycoproteins, builder):
        n = len(glycoproteins)
        n_sites = sum(len(gp.site_map['N-Linked']) for gp in glycoproteins)
        k_sites_acc = 0
        self.log(
            "Analyzing %d glycoproteins with %d occupied N-glycosites" % (n, n_sites))
        result_collector = deque()
        for i, gp in enumerate(glycoproteins, 1):
            k_sites = len(gp.site_map["N-Linked"])
            k_sites_acc += k_sites
            self.log("Building Model for \"%s\" with %d occupied N-glycosites %d/%d (%0.2f%%, %0.2f%% sites)" % (
                _truncate_name(gp.name), k_sites, i, n, i * 100.0 / n, k_sites_acc * 100.0 / n_sites))
            if self.n_threads == 1:
                builder.add_glycoprotein(gp)
            else:
                with self._lock:
                    result = self.thread_pool.apply_async(self._add_glycoprotein, (gp, builder, k_sites))
                    result_collector.append((result, gp, k_sites))
                    time.sleep(1) # sleep briefly to allow newly queued task to start
                # If the thread pool is full, we'll stop enqueuing new jobs and wait for it to clear out
                if self.thread_pool_saturated():
                    while self.thread_pool_saturated():
                        running_result, running_gp, running_gp_k_sites = result_collector.popleft()
                        while running_result.ready() and result_collector:
                            running_result.get()  # get will re-raise errors if they occurred.
                            running_result, running_gp, running_gp_k_sites = result_collector.popleft()
                        if not running_result.ready():
                            self.log("... Awaiting %s with %d Sites" % (
                                _truncate_name(running_gp.name), running_gp_k_sites))
                            running_result.get(
                                self._timeout_per_unit_site * running_gp_k_sites * self.n_threads)

        # Now drain any pending tasks.
        while result_collector:
            running_result, running_gp, running_gp_k_sites = result_collector.popleft()
            while running_result.ready() and result_collector:
                # get will re-raise errors if they occurred.
                running_result.get()
                running_result, running_gp, running_gp_k_sites = result_collector.popleft()
            if not running_result.ready():
                self.log("... Awaiting %s with %d Sites" % (
                    _truncate_name(running_gp.name), running_gp_k_sites))
                running_result.get(
                    self._timeout_per_unit_site * running_gp_k_sites * self.n_threads)

    def run(self):
        self.log("Building Belongingness Matrix")
        network, belongingness_matrix = self.make_glycan_network()

        builder = GlycosylationSiteModelBuilder(
            network, belongingness_matrix=belongingness_matrix,
            unobserved_penalty_scale=self.unobserved_penalty_scale,
            lambda_limit=self.lambda_limit,
            require_multiple_observations=self.require_multiple_observations,
            observation_aggregator=self.observation_aggregator,
            n_threads=self.n_threads)

        self.log("Aggregating Glycoproteins")
        glycoproteins = self.aggregate_identified_glycoproteins()
        glycoproteins = sorted(
            glycoproteins,
            key=lambda x: len(x.identified_glycopeptides),
            reverse=True)

        self._fit_glycoprotein_site_models(glycoproteins, builder)

        self.log("Saving Models")
        if self.output_path is not None:
            builder.save_models(self.output_path)


# '''
# import sys
# import glob
# import json

# import glycan_profiling
# from glycan_profiling import database, composition_distribution_model
# from glycan_profiling.composition_distribution_model import site_model
# from glycan_profiling.database import composition_network

# import glypy
# import glycopeptidepy

# import pandas as pd
# import numpy as np
# from scipy import linalg

# rho = composition_distribution_model.DEFAULT_RHO

# db = database.GlycopeptideDiskBackedStructureDatabase("../database/mouse_glycoproteome/synthesis-n-glycopeptide.db")
# db2 = database.GlycanCompositionDiskBackedStructureDatabase("../database/mouse_glycoproteome/synthesis-n-glycopeptide.db")

# network = db2.glycan_composition_network

# print("Initializing Augmented Network")
# augmented_network = network.augment_with_decoys()
# augmented_network.create_edges()

# print("Building Belongingness Matrix")
# model = composition_distribution_model.glycome_network_smoothing.GlycomeModel([], augmented_network)

# belongingness_matrix = model.belongingness_matrix
# augmented_network.neighborhoods = model.neighborhood_walker.neighborhoods

# print("Loading Records")
# table = pd.concat(map(pd.read_table, glob.glob("./MouseBrain-Z-T-*/*-chromatograms.txt")))
# table['glycan'] = table.glycopeptide.apply(lambda x: str(glycopeptidepy.PeptideSequence(x).glycan_composition))


# protein_names = table.protein_name.unique()
# site_models = []
# n = len(protein_names)
# for prot_i, protein_name in enumerate(protein_names):
#     protein = db.proteins[protein_name]
#     rows_for_protein = table[table.protein_name == protein_name]
#     print('\t'.join(map(str, [
#         protein_name,
#         prot_i,
#         prot_i * 100.0 / n
#     ])))

#     for site in protein.n_glycan_sequon_sites:

#         selector_mask = ((rows_for_protein.start_position <= site) & (rows_for_protein.end_position >= site))
#         record_rows = rows_for_protein[selector_mask]
#         records = []
#         for i, row in record_rows.iterrows():
#             record = composition_distribution_model.GlycanCompositionSolutionRecord(row.glycan, row.ms1_score)
#             if record.score < 1:
#                 continue

#         print site
#         print len(records)
#         print [str(r.glycan_composition) for r in records]
#         model = composition_distribution_model.GlycomeModel(records, augmented_network, belongingness_matrix)
#         reduction = model.find_threshold_and_lambda(rho)
#         grid_search = composition_distribution_model.ThresholdSelectionGridSearch(model, reduction)
#         params = grid_search.average_solution()
#         print params
#         composition_distribution_model.display_table(model.neighborhood_names, params.tau.reshape((-1, 1)))
#         dup = params.clone()
#         dup.lmbda = min(dup.lmbda, 0.2)
#         network = grid_search.annotate_network(dup)
#         glycan_map = {
#             str(node.glycan_composition): site_model.GlycanPriorRecord(node.score, not node.marked)
#             for node in network
#             if node.score > 1e-4
#         }
#         site_mod = site_model.GlycosylationSiteModel(
#             protein_name, site, dict(zip(model.neighborhood_names, params.tau)), dup.lmbda, glycan_map)
#         site_models.append(site_mod)

# print("Serializing Sites To Disk")
# with open("./glycosite_models.json", 'w') as fh:
#     sites = [s.to_dict() for s in site_models]
#     json.dump(sites, fh)

# '''
