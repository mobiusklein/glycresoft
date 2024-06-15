# Copyright (c) 2018, Joshua Klein
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dill as _dill
import warnings
from sqlalchemy import exc as sa_exc

try:
    warnings.simplefilter("ignore", category=sa_exc.SAWarning)
    warnings.filterwarnings(action="ignore", module="SPARQLWrapper")
    warnings.filterwarnings(
        action="ignore",
        category=DeprecationWarning,
        module="pysqlite2.dbapi2")
finally:
    pass

from glycresoft import serialize

from glycresoft.piped_deconvolve import (
    ScanGenerator
)

from glycresoft.chromatogram_tree import (
    MassShift, CompoundMassShift, Chromatogram, ChromatogramTreeList,
    ChromatogramTreeNode, ChromatogramInterface, ChromatogramFilter,
    mass_shift)


from glycresoft.trace import (
    ChromatogramExtractor, ChromatogramProcessor)

from glycresoft import database
from glycresoft.database import (
    NeutralMassDatabase, GlycopeptideDiskBackedStructureDatabase,
    GlycanCompositionDiskBackedStructureDatabase)

from glycresoft import profiler

from glycresoft.config.config_file import get_configuration

get_configuration()

__all__ = [
    "ScanGenerator",
    "MassShift", "CompoundMassShift", "Chromatogram",
    "ChromatogramTreeNode", "ChromatogramTreeList",
    "ChromatogramInterface", "ChromatogramFilter",
    "mass_shift", "ChromatogramExtractor", "ChromatogramProcessor",
    "NeutralMassDatabase", "GlycopeptideDiskBackedStructureDatabase",
    "GlycanCompositionDiskBackedStructureDatabase", "serialize",
    "profiler", "plotting"
]
