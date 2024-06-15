'''Implements a multiprocessing deconvolution algorithm
'''

from ms_deisotope.tools.deisotoper.scan_generator import ScanGenerator as _ScanGenerator

from glycresoft.task import (
    TaskBase,
)

from glycresoft.config import get_configuration


user_config = get_configuration()
huge_tree = user_config.get("xml_huge_tree", False)


class ScanGenerator(TaskBase, _ScanGenerator):
    pass
