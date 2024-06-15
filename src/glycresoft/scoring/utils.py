from operator import mul
try:
    reduce
except NameError:
    from functools import reduce

import numpy as np


def logit(x):
    return np.log(x) - np.log(1 - x)


def logitsum(xs):
    total = 0
    for x in xs:
        total += logit(x)
    return total


def prod(*x):
    return reduce(mul, x, 1)
