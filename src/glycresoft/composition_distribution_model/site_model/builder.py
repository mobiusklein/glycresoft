import time
import json
import logging

from collections import defaultdict, deque, namedtuple

import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Event, JoinableQueue
from multiprocessing.managers import RemoteError

from queue import Empty as QueueEmptyException
from threading import RLock, Condition, Thread

import numpy as np

from glycresoft import serialize
from glycresoft.task import TaskBase, IPCLoggingManager

from glycresoft.database import (
    GlycanCompositionDiskBackedStructureDatabase, GlycopeptideDiskBackedStructureDatabase)


from glycresoft.database.composition_network import NeighborhoodWalker, make_n_glycan_neighborhoods
from glycresoft.composition_distribution_model import (
    smooth_network, display_table, VariableObservationAggregation,
    GlycanCompositionSolutionRecord, GlycomeModel)
from glycresoft.models import GeneralScorer, get_feature


from .glycosite_model import GlycanPriorRecord, GlycosylationSiteModel
from .glycoprotein_model import ProteinStub

_default_chromatogram_scorer = GeneralScorer.clone()
_default_chromatogram_scorer.add_feature(get_feature("null_charge"))

logger = logging.getLogger("glycresoft.glycosite_model")
logger.addHandler(logging.NullHandler())


def _truncate_name(name, limit=30):
    if len(name) > limit:
        return name[:limit - 3] + '...'
    return name



class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


EmptySite = namedtuple("EmptySite", ("position", "protein_name"))


class GlycosylationSiteModelBuilder(TaskBase):
    _timeout_per_unit = 300

    def __init__(self, glycan_graph, chromatogram_scorer=None, belongingness_matrix=None,
                 unobserved_penalty_scale=None, lambda_limit=0.2,
                 require_multiple_observations=True,
                 observation_aggregator=None, n_threads=1):
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
        self._lock = RLock() if self.n_threads > 1 else FakeLock()
        self._concurrent_jobs = 0
        if self.n_threads > 1:
            self.thread_pool = ThreadPool(n_threads)
        else:
            self.thread_pool = None

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

    def prepare_glycoprotein(self, glycoprotein, evaluate_chromatograms=False):
        prepared = []
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
                records.append(self._transform_glycopeptide(
                    gp, evaluate_chromatograms))
            prepared.append((records, site, ProteinStub(glycoprotein.name)))
        return prepared

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
                records.append(self._transform_glycopeptide(
                    gp, evaluate_chromatograms))

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
            stable_cases = set(
                [gc.glycan_composition for gc, v in zip(recs, rec_variance) if v != 1.0])
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
        log_buffer = []
        log_buffer.append("... %d Glycan Compositions for Site %d of %s" % (
            len(acc), site, _truncate_name(glycoprotein.name, )))
        # for key, value in sorted(acc.items(), key=lambda x: x[0].mass()):
        #     log_buffer.append("... %s: [%s]" % (
        #         key,
        #         ', '.join(["%0.2f" % f for f in sorted(
        #             [r.score for r in value])])
        #     ))
        # self.log('\n'.join(log_buffer))

        fitted_network, search_result, params = smooth_network(
            self.network, learnable_cases,
            belongingness_matrix=self.belongingness_matrix,
            observation_aggregator=VariableObservationAggregation,
            annotate_network=False)
        if params is None:
            self.log("Skipping Site %d of %s" %
                     (site, _truncate_name(glycoprotein.name)))
            return None
        self.log("... Site %d of %s Lambda: %f" %
                 (site, _truncate_name(glycoprotein.name), params.lmbda,))
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

        site_distribution = dict(
            zip([x.name for x in self.network.neighborhoods], updated_params.tau.tolist()))
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
        self.site_models.append(site_model.pack())
        return site_model

    def save_models(self, path):
        with open(path, 'wt') as fh:
            prepared = []
            for site in sorted(self.site_models, key=lambda x: (x.protein_name, x.position)):
                prepared.append(site.to_dict())
            json.dump(prepared, fh)


class GlycositeModelBuildingProcess(Process):
    process_name = "glycosylation-site-modeler"

    def __init__(self, builder, input_queue, output_queue, producer_done_event, output_done_event, log_handler):
        Process.__init__(self)
        self.builder = builder
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.producer_done_event = producer_done_event
        self.output_done_event = output_done_event
        self.log_handler = log_handler

    def all_work_done(self):
        """A helper method to encapsulate :attr:`output_done_event`'s ``is_set``
        method.

        Returns
        -------
        bool
        """
        return self.output_done_event.is_set()

    def log(self, message):
        """Send a normal logging message via :attr:`log_handler`

        Parameters
        ----------
        message : str
            The message to log
        """
        logger.info(message)

    def debug(self, message):
        """Send a debugging message via :attr:`log_handler`

        Parameters
        ----------
        message : str
            The message to log
        """
        logger.debug(message)

    def handle_item(self, observations, site, glycoprotein):
        model = self.builder.fit_site_model(observations, site, glycoprotein)
        if model is None:
            model = EmptySite(site, glycoprotein.name)
        self.output_queue.put(model)
        # Do not accumulate models within the process
        self.builder.site_models = []

    def task(self):
        """The worker process's main loop where it will poll for new work items,
        process incoming work items and send them back to the master process.
        """
        has_work = True
        self.items_handled = 0
        strikes = 0
        self.log_handler.add_handler()
        while has_work:
            try:
                observations, site, glycoprotein = self.input_queue.get(True, 5)
                self.input_queue.task_done()
                strikes = 0
            except QueueEmptyException:
                if self.producer_done_event.is_set():
                    has_work = False
                    break
                else:
                    strikes += 1
                    if strikes % 1000 == 0:
                        self.log("... %d iterations without work for %r" %
                                 (strikes, self))
                    continue
            self.items_handled += 1
            try:
                self.handle_item(observations, site, glycoprotein)
            except Exception:
                import traceback
                message = "An error occurred while processing %r of %r on %r:\n%s" % (
                    site, glycoprotein.name, self, traceback.format_exc())
                self.log(message)
                break
        self.cleanup()

    def cleanup(self):
        self.output_done_event.set()

    def run(self):
        new_name = getattr(self, 'process_name', None)
        if new_name is not None:
            TaskBase().try_set_process_name(new_name)
        # The task might actually use the same logger from different threads
        # which causes a deadlock. This "fixes" that. by writing directly to STDOUT
        # at the cost of not being able to write to file instead.
        # TaskBase.log_to_stdout()
        self.output_done_event.clear()
        try:
            self.task()
        except Exception:
            import traceback
            self.log("An exception occurred while executing %r.\n%s" % (
                self, traceback.format_exc()))
            self.cleanup()


class GlycoproteinSiteModelBuildingWorkflowBase(TaskBase):
    def __init__(self, analyses, glycopeptide_database, glycan_database,
                 unobserved_penalty_scale=None, lambda_limit=0.2,
                 require_multiple_observations=True, observation_aggregator=None,
                 output_path=None, n_threads=None, q_value_threshold=0.05,
                 network=None, include_decoy_glycans=True):
        if observation_aggregator is None:
            observation_aggregator = VariableObservationAggregation
        if unobserved_penalty_scale is None:
            unobserved_penalty_scale = 1.0

        self.q_value_threshold = q_value_threshold
        self.analyses = analyses
        self.glycopeptide_database = glycopeptide_database
        self.glycan_database = glycan_database

        self.unobserved_penalty_scale = unobserved_penalty_scale
        self.lambda_limit = lambda_limit

        self.require_multiple_observations = require_multiple_observations
        self.observation_aggregator = observation_aggregator

        self.output_path = output_path
        self.network = network
        self.include_decoy_glycans = include_decoy_glycans

    @classmethod
    def from_paths(cls, analysis_paths_and_ids, glycopeptide_hypothesis_path, glycopeptide_hypothesis_id,
                   glycan_hypothesis_path, glycan_hypothesis_id, unobserved_penalty_scale=None,
                   lambda_limit=0.2, require_multiple_observations=True, observation_aggregator=None,
                   output_path=None, n_threads=4, q_value_threshold=0.05, network=None, include_decoy_glycans=True):
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
            n_threads=n_threads, q_value_threshold=q_value_threshold, network=network,
            include_decoy_glycans=include_decoy_glycans
        )
        return inst

    def count_glycosites(self, glycoproteins):
        n_sites = sum(len(gp.site_map['N-Linked']) for gp in glycoproteins)
        return n_sites

    def make_glycan_network(self):
        if self.network is None:
            network = self.glycan_database.glycan_composition_network
        else:
            network = self.network
        if self.include_decoy_glycans:
            network = network.augment_with_decoys()
        network.create_edges()

        model = GlycomeModel([], network)
        belongingness_matrix = model.belongingness_matrix
        network.neighborhoods = model.neighborhood_walker.neighborhoods
        return network, belongingness_matrix

    def load_identified_glycoproteins_from_analysis(self, analysis):
        if not isinstance(analysis, serialize.Analysis):
            analysis = analysis.analysis
        idgps = analysis.aggregate_identified_glycoproteins(
            analysis.identified_glycopeptides.filter(
                serialize.IdentifiedGlycopeptide.q_value <= self.q_value_threshold))
        return idgps

    def build_reference_protein_map(self):
        proteins = list(self.glycopeptide_database.proteins)
        index = {p.name: p for p in proteins}
        return index

    def aggregate_identified_glycoproteins(self):
        acc = defaultdict(list)
        n = len(self.analyses)
        for i, analysis in enumerate(self.analyses, 1):
            self.log("... Loading Glycopeptides for %s (%d/%d)" %
                     (analysis.name, i, n))
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
                self.log("... Merging %s (%d/%d, %0.2f%%)" %
                         (name, i, n, i * 100.0 / n))
                should_log = False
            agg = duplicates[0]
            result.append(agg.merge(*duplicates[1:]))
            i += 1
        return result

    def _fit_glycoprotein_site_models(self, glycoproteins, builder):
        raise NotImplementedError()

    def _init_builder(self, network, belongingness_matrix):
        builder = GlycosylationSiteModelBuilder(
            network, belongingness_matrix=belongingness_matrix,
            unobserved_penalty_scale=self.unobserved_penalty_scale,
            lambda_limit=self.lambda_limit,
            require_multiple_observations=self.require_multiple_observations,
            observation_aggregator=self.observation_aggregator,
            n_threads=self.n_threads)
        return builder

    def run(self):
        self.log("Building Belongingness Matrix")
        network, belongingness_matrix = self.make_glycan_network()

        builder = self._init_builder(network, belongingness_matrix)

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


class ThreadedGlycoproteinSiteModelBuildingWorkflow(GlycoproteinSiteModelBuildingWorkflowBase):
    _timeout_per_unit_site = 300

    def __init__(self, analyses, glycopeptide_database, glycan_database,
                 unobserved_penalty_scale=None, lambda_limit=0.2,
                 require_multiple_observations=True, observation_aggregator=None,
                 output_path=None, n_threads=4, q_value_threshold=0.05, network=None, include_decoy_glycans=True):
        super(ThreadedGlycoproteinSiteModelBuildingWorkflow, self).__init__(
            analyses, glycopeptide_database, glycan_database,
            unobserved_penalty_scale, lambda_limit,
            require_multiple_observations, observation_aggregator,
            output_path,  q_value_threshold=q_value_threshold, network=network,
            include_decoy_glycans=include_decoy_glycans
        )

        self.n_threads = n_threads
        self.thread_pool = ThreadPool(self.n_threads)
        self._lock = RLock()
        self._count_barrier = Condition()
        self._concurrent_jobs = 0

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
        n_sites = self.count_glycosites(glycoproteins)
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
                    result = self.thread_pool.apply_async(
                        self._add_glycoprotein, (gp, builder, k_sites))
                    result_collector.append((result, gp, k_sites))
                    # sleep briefly to allow newly queued task to start
                    time.sleep(1)
                # If the thread pool is full, we'll stop enqueuing new jobs and wait for it to clear out
                if self.thread_pool_saturated():
                    while self.thread_pool_saturated():
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


class MultiprocessingGlycoproteinSiteModelBuildingWorkflow(GlycoproteinSiteModelBuildingWorkflowBase):

    def __init__(self, analyses, glycopeptide_database, glycan_database,
                 unobserved_penalty_scale=None, lambda_limit=0.2,
                 require_multiple_observations=True, observation_aggregator=None,
                 output_path=None, n_threads=4, q_value_threshold=0.05, network=None,
                 include_decoy_glycans=True):
        super(MultiprocessingGlycoproteinSiteModelBuildingWorkflow, self).__init__(
            analyses, glycopeptide_database, glycan_database,
            unobserved_penalty_scale, lambda_limit,
            require_multiple_observations, observation_aggregator,
            output_path, q_value_threshold=q_value_threshold, network=network,
            include_decoy_glycans=include_decoy_glycans
        )

        self.builder = None

        self.input_queue = JoinableQueue(1000)
        self.output_queue = JoinableQueue(1000)
        self.input_done_event = Event()
        self.n_threads = 1
        self.n_workers = n_threads
        self.workers = []
        self._has_remote_error = False
        self.ipc_manager = IPCLoggingManager()

    def prepare_glycoprotein_for_dispatch(self, glycoprotein, builder):
        prepared = builder.prepare_glycoprotein(glycoprotein)
        return prepared

    def feed_queue(self, glycoproteins, builder):
        n = len(glycoproteins)
        n_sites = self.count_glycosites(glycoproteins)
        self.log(
            "Analyzing %d glycoproteins with %d occupied N-glycosites" % (n, n_sites))
        i_site = 0
        for glycoprotein in glycoproteins:
            prepared = self.prepare_glycoprotein_for_dispatch(glycoprotein, builder)
            for work_item in prepared:
                i_site += 1
                self.input_queue.put(work_item)
                if i_site % 50 == 0 and i_site != 0:
                    self.input_queue.join()
        self.input_done_event.set()

    def _handle_local(self, glycoproteins, builder, seen):
        for glycoprotein in glycoproteins:
            prepared = self.prepare_glycoprotein_for_dispatch(
                glycoprotein, builder)
            for records, site, protein_stub in prepared:
                key = (protein_stub.name, site)
                if key in seen:
                    continue
                else:
                    seen[key] = -1
                    model = builder.fit_site_model(records, site, protein_stub)
                    if model is not None:
                        self.builder.site_models.append(model)

    def make_workers(self):
        for _i in range(self.n_workers):
            worker = GlycositeModelBuildingProcess(
                self.builder, self.input_queue, self.output_queue,
                producer_done_event=self.input_done_event,
                output_done_event=Event(),
                log_handler=self.ipc_manager.sender())
            self.workers.append(worker)
            worker.start()

    def clear_pool(self):
        for _i, worker in enumerate(self.workers):
            exitcode = worker.exitcode
            if exitcode != 0 and exitcode is not None:
                self.log("... Worker Process %r had exitcode %r" % (worker, exitcode))
            try:
                worker.join(1)
            except AttributeError:
                pass
            if worker.is_alive():
                self.debug("... Worker Process %r is still alive and incomplete" % (worker, ))
                worker.terminate()

    def all_workers_finished(self):
        """Check if all worker processes have finished.
        """
        worker_still_busy = False
        assert self.workers
        for worker in self.workers:
            try:
                is_done = worker.all_work_done()
                if not is_done:
                    worker_still_busy = True
                    break
            except (RemoteError, KeyError):
                worker_still_busy = True
                self._has_remote_error = True
                break
        return not worker_still_busy

    def _fit_glycoprotein_site_models(self, glycoproteins, builder):
        self.builder = builder
        feeder_thread = Thread(target=self.feed_queue, args=(glycoproteins, builder))
        feeder_thread.daemon = True
        feeder_thread.start()
        self.make_workers()
        n_sites = self.count_glycosites(glycoproteins)
        seen = dict()
        strikes = 0
        start_time = time.time()
        i = 0
        has_work = True
        while has_work:
            try:
                site_model = self.output_queue.get(True, 3)
                self.output_queue.task_done()
                key = (site_model.protein_name, site_model.position)
                seen[(key)] = i
                if key in seen:
                    self.debug(
                        "...... Duplicate Results For %s. First seen at %r, now again at %r" % (
                            key, seen[key], i, ))
                else:
                    seen[key] = i
                i += 1
                strikes = 0
                if i % 1 == 0:
                    self.log(
                        "...... Processed %d sites (%0.2f%%)" % (i, i * 100. / n_sites))
                if not isinstance(site_model, EmptySite):
                    self.builder.site_models.append(site_model)
            except QueueEmptyException:
                if len(seen) == n_sites:
                    has_work = False
                # do worker life cycle management here
                elif self.all_workers_finished():
                    if len(seen) == n_sites:
                        has_work = False
                    else:
                        strikes += 1
                        if strikes % 25 == 0:
                            self.log(
                                "...... %d cycles without output (%d/%d, %0.2f%% Done)" % (
                                    strikes, len(seen), n_sites, len(seen) * 100. / n_sites))
                            self.debug("...... Processes")
                            for worker in self.workers:
                                self.debug("......... %r" % (worker,))
                            self.debug("...... IPC Manager: %r" % (self.ipc_manager,))
                        if strikes > 150:
                            self.log("Too much time has elapsed waiting for final results, finishing locally.")
                            self._handle_local(glycoproteins, builder, seen)
                else:
                    strikes += 1
                    if strikes % 50 == 0:
                        self.log(
                            "...... %d cycles without output (%d/%d, %0.2f%% Done, %d children still alive)" % (
                                strikes, len(seen), n_sites, len(
                                    seen) * 100. / n_sites,
                                len(multiprocessing.active_children()) - 1))
                        try:
                            input_queue_size = self.input_queue.qsize()
                        except Exception:
                            input_queue_size = -1
                        is_feeder_done = self.input_done_event.is_set()
                        self.log("...... Input Queue Status: %r. Is Feeder Done? %r" % (
                            input_queue_size, is_feeder_done))
                    if strikes > 150:
                        self.log("Too much time has elapsed waiting for workers, finishing locally.")
                        self._handle_local(glycoproteins, builder, seen)
                continue
        self.clear_pool()
        self.ipc_manager.stop()
        feeder_thread.join()
        dispatcher_end = time.time()
        self.log("... Dispatcher Finished (%0.3g sec.)" %
                 (dispatcher_end - start_time))


GlycoproteinSiteModelBuildingWorkflow = MultiprocessingGlycoproteinSiteModelBuildingWorkflow
