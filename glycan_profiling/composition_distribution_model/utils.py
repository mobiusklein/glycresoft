from __future__ import print_function

import numpy as np


def display_table(names, values, sigfig=3, filter_empty=1, print_fn=None):
    if print_fn is None:
        print_fn = print
    values = np.array(values)
    maxlen = len(max(names, key=len)) + 2
    fstring = ("%%0.%df" % sigfig)
    for i in range(len(values)):
        if values[i, :].sum() or not filter_empty:
            print_fn(names[i].ljust(maxlen) + ('|'.join([fstring % f for f in values[i, :]])))
