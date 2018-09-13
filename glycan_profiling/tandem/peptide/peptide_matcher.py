from collections import OrderedDict
from multiprocessing import Manager as IPCManager

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.task import TaskBase

from ..glycopeptide.scoring import GroupwiseTargetDecoyAnalyzer
from ..glycopeptide.glycopeptide_matcher import (
    GlycopeptideResolver,
    GlycopeptideMatcher,
    TargetDecoyInterleavingGlycopeptideMatcher,
    ComparisonGlycopeptideMatcher)
from ..spectrum_evaluation import TandemClusterEvaluatorBase, DEFAULT_BATCH_SIZE, ScanQuery
from ..process_dispatcher import SpectrumIdentificationWorkerBase
from ..temp_store import TempFileManager, SpectrumMatchStore
from ..chromatogram_mapping import ChromatogramMSMSMapper

from ..workflow import ExclusiveDatabaseSearchComparerBase

from glycan_profiling.structure import (
    CachingPeptideParser)


class PeptideIdentificationWorker(SpectrumIdentificationWorkerBase):
    def __init__(self, input_queue, output_queue, producer_done_event, consumer_done_event,
                 scorer_type, evaluation_args, spectrum_map, mass_shift_map, log_handler,
                 parser_type):
        SpectrumIdentificationWorkerBase.__init__(
            self, input_queue, output_queue, producer_done_event, consumer_done_event,
            scorer_type, evaluation_args, spectrum_map, mass_shift_map,
            log_handler=log_handler)
        self.parser = parser_type()

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher


class ExclusivePeptideDatabaseSearchComparer(ExclusiveDatabaseSearchComparerBase):

    def _make_evaluator(self, batch):
        raise NotImplementedError()
