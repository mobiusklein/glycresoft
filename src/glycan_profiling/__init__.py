import sys
import glycresoft

from glycresoft import serialize, chromatogram_tree, config, task

sys.modules["glycresoft"] = glycresoft
sys.modules['glycresoft.serialize'] = serialize
sys.modules['glycresoft.chromatogram_tree'] = chromatogram_tree
sys.modules["glycresoft.config"] = config
sys.modules["glycresoft.task"] = task