cimport cython
from cpython.dict cimport PyDict_GetItem
from cpython.object cimport PyObject

@cython.freelist(10000)
cdef class LRUNode(object):
    cdef:
        public object data
        public LRUNode forward
        public LRUNode backward

    def __init__(self, data, forward, backward):
        self.data = data
        self.forward = forward
        self.backward = backward

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __repr__(self):
        return "LRUNode(%s)" % self.data

    @staticmethod
    cdef LRUNode _create(object data, LRUNode forward, LRUNode backward):
        cdef LRUNode inst = LRUNode.__new__(LRUNode)
        inst.data = data
        inst.forward = forward
        inst.backward = backward
        return inst


cdef class LRUCache(object):
    cdef:
        public LRUNode head
        public dict _mapping

    def __init__(self):
        self.head = LRUNode._create(None, None, None)
        self.head.forward = self.head
        self.head.backward = self.head
        self._mapping = dict()

    cpdef move_node_up(self, LRUNode node):
        was_head_lru = self.head.backward is self.head
        if was_head_lru:
            raise ValueError("Head cannot be LRU")
        mover_forward = node.forward
        mover_backward = node.backward

        pushed = self.head.forward
        # If the most-recently used node is being moved up,
        # short-circuit
        if pushed == node:
            return
        # If the node being moved up happens to form a cycle with
        # the head node, (which should technically be handled by the
        # above short-circuit) don't make any changes as this will
        # lead to a detached head node.
        if mover_backward is self.head and mover_forward is self.head:
            return
        mover_backward.forward = mover_forward
        mover_forward.backward = mover_backward

        pushed.backward = node

        node.backward = self.head
        node.forward = pushed

        self.head.forward = node
        now_head_lru = self.head.backward is self.head
        if now_head_lru:
            raise ValueError("Head became LRU!")

    cpdef add_node(self, data):
        if data in self._mapping:
            self.hit_node(data)
            return
        node = LRUNode._create(data, None, None)

        pushed = self.head.forward
        self.head.forward = node
        node.backward = self.head
        node.forward = pushed
        pushed.backward = node

        self._mapping[node.data] = node

    cpdef get_least_recently_used(self):
        lru = self.head.backward
        if lru is self.head:
            raise ValueError("Head node cannot be LRU!")
        return lru.data

    cpdef hit_node(self, k):
        cdef:
            LRUNode out
            PyObject* tmp
        tmp = PyDict_GetItem(self._mapping, k)
        if tmp == NULL:
            raise KeyError(k)
        else:
            out = <LRUNode>tmp
            self.move_node_up(out)

    cpdef unspool(self):
        chain = []
        current = self.head
        while current.forward is not self.head:
            chain.append(current)
            current = current.forward
        chain.append(current)
        return chain

    cpdef remove_node(self, data):
        cdef:
            LRUNode node, fwd, bck
        node = self._mapping[data]
        fwd = node.forward
        bck = node.backward
        fwd.backward = bck
        bck.forward = fwd
        self._mapping.pop(data)

    cpdef clear(self):
        self.head = LRUNode._create(None, None, None)
        self.head.forward = self.head
        self.head.backward = self.head
        self._mapping.clear()


cdef class LRUMapping(object):
    cdef:
        public size_t max_size
        public dict store
        public LRUCache lru

    def __init__(self, max_size=512):
        self.max_size = max_size
        self.store = dict()
        self.lru = LRUCache()

    def __getitem__(self, key):
        value = self.store[key]
        self.lru.hit_node(key)
        return value

    def __setitem__(self, key, value):
        self.lru.add_node(key)
        self.store[key] = value
        self._check_size()

    def __delitem__(self, key):
        del self.store[key]
        self.lru.remove_node(key)

    cpdef _check_size(self):
        cdef:
            Py_ssize_t n
        n = len(self.store)
        while n > self.max_size:
            key = self.lru.get_least_recently_used()
            self.store.pop(key)
            self.lru.remove_node(key)
            n -= 1

    def __contains__(self, key):
        cdef:
            PyObject* temp

        temp = PyDict_GetItem(self.store, key)
        if temp == NULL:
            return False
        return True

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def get(self, key, default=None):
        try:
            value = self[key]
            return value
        except KeyError:
            return default

    def pop(self, key, default=None):
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            return default

    def clear(self):
        self.store.clear()
        self.lru.clear()