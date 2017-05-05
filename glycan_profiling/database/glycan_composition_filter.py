'''
Usage:
    list_of_db_glycans = get_db_glycans()
    gcf = GlycanCompositionFilter(list_of_db_glycans)
    index = InclusionFilter(gcf.query("Fuc", 1, 3).add("HexNAc", 4, 6).add("Neu5Ac", 0, 2))
    id_of_interest = 5
    print(5 in index)
'''
from collections import defaultdict, Counter


class FilterTreeNode(object):
    def __init__(self, members=None):
        if members is None:
            members = []
        self.members = members
        self.children = defaultdict(FilterTreeNode)
        self.splitting_key = None
        self.split_value = None

    def add(self, member):
        self.members.append(member)

    def split_on(self, key):
        if self.splitting_key is not None:
            raise ValueError("Already split node")
        for member in self.members:
            self.children[member[key]].add(member)

        self.splitting_key = key
        for key, child in self.children.items():
            child.split_value = key

    def split_sequence(self, key_sequence):
        if len(key_sequence) == 0:
            return
        key = key_sequence[0]
        self.split_on(key)
        for child in self.children.values():
            child.split_sequence(key_sequence[1:])

    def __repr__(self):
        return "FilterTreeNode(%d, %r, ^%r, %d)" % (
            len(self.members), self.splitting_key,
            self.split_value, len(self.children))

    def query(self, key, lo=0, hi=100):
        if str(key) == str(self.splitting_key):
            result_set = []
            hi = min(self.children.keys(), hi)
            for i in range(lo, hi + 1):
                result_set.append(self.children[i])
            return QuerySet(result_set)
        else:
            out = []
            for child in self.children.values():
                out.append(child.query(key, lo, hi))
            return QuerySet.union(out)

    def __iter__(self):
        if len(self.children) == 0:
            for member in self.members:
                yield member
        else:
            for child in self.children.values():
                for item in child:
                    yield item


class QuerySet(object):
    def __init__(self, node_list):
        self.node_list = node_list

    def __iter__(self):
        for node in self.node_list:
            for item in node:
                yield item

    def query(self, key, lo=0, hi=100):
        out = []
        for node in self.node_list:
            out.append(node.query(key, lo, hi))
        return QuerySet.union(out)

    @classmethod
    def union(cls, queries):
        node_list = []
        for q in queries:
            node_list.extend(q.node_list)
        return cls(node_list)

    @classmethod
    def intersect(cls, queries):
        item_map = {}
        member_map = Counter()
        for q in queries:
            for item in q:
                item_map[item.id] = item
                member_map[item.id] += 1
        n = len(queries)
        out = []
        for key, value in member_map.items():
            if value == n:
                out.append(item_map[key])
        return out


class QueryInterval(object):
    def __init__(self, key, low=0, high=100):
        self.key = str(key)
        self.low = low
        self.high = high

    def __repr__(self):
        return "QueryInterval(%r, %d, %d)" % (self.key, self.low, self.high)

    def __iter__(self):
        yield self.key
        yield self.low
        yield self.high

    def __eq__(self, other):
        return (self.key == other.key and self.low == other.low and self.high == other.high)

    def __ne__(self, other):
        return not (self == other)


class QueryComposer(object):
    def __init__(self, parent):
        self.parent = parent
        self.filters = []

    def __call__(self, filter_spec_or_key, lo=0, hi=100):
        if isinstance(filter_spec_or_key, QueryInterval):
            self.filters.append(filter_spec_or_key)
        else:
            self.filters.append(QueryInterval(filter_spec_or_key, lo, hi))
        return self

    def add(self, *args, **kwargs):
        self(*args, **kwargs)
        return self

    def collate(self):
        monosaccharides = list(map(str, self.parent.monosaccharides))
        filters = sorted(self.filters, key=lambda x: str(x.key))
        kept_filters = []
        for filt in filters:
            if filt.key in monosaccharides:
                kept_filters.append(filt)
        return kept_filters

    def __eq__(self, other):
        if isinstance(other, QueryComposer):
            return self.collate() == other.collate()
        else:
            return self.collate() == other

    def __ne__(self, other):
        return not (self == other)

    def all(self):
        return list(self._compose())

    def _compose(self):
        query = None
        for filt in self.collate():
            if query is None:
                query = self.parent.root.query(*filt)
            else:
                query = query.query(*filt)
        if query is None:
            return self.parent
        return query

    def __iter__(self):
        return iter(self._compose())


class GlycanCompositionFilter(object):
    def __init__(self, members):
        self.members = members
        self.monosaccharides = list()
        self._extract_monosaccharides(members)
        self.root = FilterTreeNode(members)
        self._build_partitions()

    def _extract_monosaccharides(self, iterable):
        monosaccharides = set()
        for case in iterable:
            monosaccharides.update(case.keys())
        self.monosaccharides = sorted(monosaccharides, key=str)

    def _build_partitions(self):
        self.root.split_sequence(self.monosaccharides)

    def query(self, *args, **kwargs):
        compose = QueryComposer(self)
        return compose(*args, **kwargs)

    def filter_index(self, *args, **kwargs):
        query_set = self.query(*args, **kwargs)
        return InclusionFilter(query_set)

    def __iter__(self):
        return iter(self.members)


class InclusionFilter(object):
    def __init__(self, query_set):
        self.query_set = query_set
        self.hash_index = set()
        self._build_index()

    def _build_index(self):
        for case in self.query_set:
            self.hash_index.add(case.id)

    def __repr__(self):
        return "InclusionFilter()"

    def contains_all(self, keys):
        for key in keys:
            if key not in self:
                return False
        return True

    def __contains__(self, key):
        return key in self.hash_index
