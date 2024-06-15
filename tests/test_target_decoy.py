import unittest


import numpy as np

from glycresoft.tandem import target_decoy
from . import fixtures


class TestNearestValueLookup(unittest.TestCase):

    def _get_instance(self):
        pairs = np.loadtxt(fixtures.get_test_data('numpairs.txt'))
        return target_decoy.NearestValueLookUp(map(list, pairs))

    def test_query(self):
        nvl = self._get_instance()
        indices = np.linspace(0, 20)
        values = [0.34342784, 0.26780415, 0.22610257, 0.19385776, 0.16619172,
                  0.14381995, 0.12177308, 0.10591941, 0.09414169, 0.08243626,
                  0.07534748, 0.06410644, 0.05670103, 0.05028203, 0.04507182,
                  0.04016268, 0.03451865, 0.03071733, 0.02744457, 0.02566964,
                  0.02328244, 0.02104853, 0.01972112, 0.0186802, 0.01601498,
                  0.01471843, 0.01368375, 0.01263298, 0.01172756, 0.0101476,
                  0.00941398, 0.0079023, 0.00655977, 0.00564417, 0.00503525,
                  0.00482724, 0.00415692, 0.00415692, 0.00398406, 0.00398406,
                  0.00330487, 0.00282247, 0.00254669, 0.00232626, 0.00178678,
                  0.00178678, 0.00152486, 0.00124185, 0.00124185, 0.00096339]
        for ind, val in zip(indices, values):
            q = nvl[ind]
            self.assertAlmostEqual(val, q)
