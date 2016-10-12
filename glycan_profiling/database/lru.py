class LRUNode(object):

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


class LRUCache(object):

    def __init__(self):
        self.head = LRUNode(None, None, None)
        self.head.forward = self.head
        self.head.backward = self.head
        self._mapping = dict()

    def move_node_up(self, node):
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

    def add_node(self, data):
        if data in self._mapping:
            self.hit_node(data)
            return
        node = LRUNode(data, None, None)

        pushed = self.head.forward
        self.head.forward = node
        node.backward = self.head
        node.forward = pushed
        pushed.backward = node

        self._mapping[node.data] = node

    def get_least_recently_used(self):
        lru = self.head.backward
        if lru is self.head:
            raise ValueError("Head node cannot be LRU!")
        return lru.data

    def hit_node(self, k):
        out = self._mapping[k]
        self.move_node_up(out)

    def unspool(self):
        chain = []
        current = self.head
        while current.forward is not self.head:
            chain.append(current)
            current = current.forward
        chain.append(current)
        return chain

    def remove_node(self, data):
        # print("Removing ", data, id(data))
        node = self._mapping[data]
        fwd = node.forward
        # assert fwd is not node
        bck = node.backward
        # assert bck is not node
        fwd.backward = bck
        bck.forward = fwd
        self._mapping.pop(data)
        # assert node not in self.unspool()[1:]
        # print("Removed")
