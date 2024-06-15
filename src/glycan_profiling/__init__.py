import sys
import glycresoft

from glycresoft import serialize, chromatogram_tree, config, task, structure

sys.modules["glycan_profiling"] = glycresoft
sys.modules['glycan_profiling.serialize'] = serialize
sys.modules['glycan_profiling.chromatogram_tree'] = chromatogram_tree
sys.modules["glycan_profiling.config"] = config
sys.modules["glycan_profiling.task"] = task
sys.modules["glycan_profiling.structure"] = structure