import importlib
import sys
from types import ModuleType

import glycresoft

class LazyModule(ModuleType):
    def __init__(self, name, mod_name):
        super().__init__(name)
        self.__mod_name = name

    def __getattr__(self, attr):
        if "_lazy_module" not in self.__dict__:
            self._lazy_module = importlib.import_module(self.__mod_name, package="glycresoft")
        return getattr(self._lazy_module, attr)

sys.modules['glycan_profiling'] = LazyModule("glycresoft", "glycresoft")
sys.modules["glycan_profiling.serialize"] = LazyModule("glycresoft.serialize", "glycresoft")
sys.modules["glycan_profiling.chromatogram_tree"] = LazyModule("glycresoft.chromatogram_tree", "glycresoft")
sys.modules["glycan_profiling.config"] = LazyModule("glycresoft.config", "glycresoft")
sys.modules["glycan_profiling.task"] = LazyModule("glycresoft.task", "glycresoft")
sys.modules["glycan_profiling.structure"] = LazyModule("glycresoft.structure", "glycresoft")

