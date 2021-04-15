cimport cython

from cpython.list cimport PyList_Size, PyList_GET_ITEM
from ms_deisotope._c.peak_dependency_network.intervals cimport SpanningMixin


cdef class QueryIntervalBase(SpanningMixin):

    cdef:
        public double center

    def __init__(self, center, start, end):
        self.center = center
        self.start = start
        self.end = end

    def __hash__(self):
        return hash((self.start, self.center, self.end))

    def __eq__(self, other):
        return (self.start, self.center, self.end) == (other.start, other.center, other.end)

    def __repr__(self):
        return "QueryInterval(%0.4f, %0.4f)" % (self.start, self.end)

    def __reduce__(self):
        return QueryIntervalBase, (self.center, self.start, self.end)

    cpdef QueryIntervalBase extend(self, QueryIntervalBase other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.center = (self.start + self.end) / 2.
        return self

    @staticmethod
    cdef QueryIntervalBase _create(double center, double start, double end):
        cdef QueryIntervalBase new = QueryIntervalBase.__new__(QueryIntervalBase)
        new.center = center
        new.start = start
        new.end = end
        return new

    cpdef QueryIntervalBase scale(self, double x):
        new = QueryIntervalBase._create(
            self.center,
            self.center - ((self.center - self.start) * x),
            self.center + ((self.end - self.center) * x))
        return new

    cpdef QueryIntervalBase copy(self):
        return QueryIntervalBase._create(self.center, self.start, self.end)

    cpdef QueryIntervalBase difference(self, SpanningMixin other):
        if self.contains_interval(other):
            return self.copy()
        elif self.is_contained_in_interval(other):
            return QueryIntervalBase._create(self.center, self.center, self.center)
        if self.start < other.start:
            if self.end < other.start:
                return self.copy()
            else:
                return QueryIntervalBase._create(self.center, self.start, other.start)
        elif self.start > other.end:
            return self.copy()
        elif self.start <= other.end:
            return QueryIntervalBase._create(self.center, other.end, self.end)
        else:
            return self.copy()


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

    cpdef IntervalFilter compress(self):
        cdef:
            size_t i, n
            list out
            QueryIntervalBase last, interval
        n = PyList_Size(self.intervals)
        out = []
        if n == 0:
            return self
        last = <QueryIntervalBase>PyList_GET_ITEM(self.intervals, 0)
        for i in range(1, n):
            interval = <QueryIntervalBase>PyList_GET_ITEM(self.intervals, i)
            if interval.overlaps(last):
                last.extend(interval)
            else:
                out.append(last)
                last = interval
        out.append(last)
        return self

