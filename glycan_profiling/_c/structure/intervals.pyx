cimport cython

from cpython.list cimport PyList_Size, PyList_GET_ITEM
from ms_deisotope._c.peak_dependency_network.intervals cimport SpanningMixin


cdef class QueryIntervalBase(SpanningMixin):

    cdef:
        public double center
        public size_t size

    def __hash__(self):
        return hash((self.start, self.center, self.end))

    def __eq__(self, other):
        return (self.start, self.center, self.end) == (other.start, other.center, other.end)

    def __repr__(self):
        return "QueryInterval(%0.4f, %0.4f)" % (self.start, self.end)

    def extend(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.center = (self.start + self.end) / 2.


cdef class PPMQueryInterval(QueryIntervalBase):

    def __init__(self, mass, error_tolerance=2e-5):
        self.center = mass
        self.start = mass - (mass * error_tolerance)
        self.end = mass + (mass * error_tolerance)


cdef class FixedQueryInterval(QueryIntervalBase):

    def __init__(self, mass, width=3):
        self.center = mass
        self.start = mass - width
        self.end = mass + width


cdef class IntervalFilter(object):
    cdef:
        public list intervals

    def __init__(self, intervals):
        self.intervals = list(intervals)

    cpdef bint test(self, double mass):
        cdef:
            size_t i, n
            SpanningMixin v
            list intervals
        intervals = self.intervals
        n = PyList_Size(intervals)
        for i in range(n):
            v = <SpanningMixin>PyList_GET_ITEM(intervals, i)
            if v._contains(mass):
                return True
        return False

    def __iter__(self):
        return iter(self.intervals)

    def __getitem__(self, i):
        return self.intervals[i]

    def __len__(self):
        return len(self.intervals)

    def __call__(self, mass):
        return self.test(mass)


