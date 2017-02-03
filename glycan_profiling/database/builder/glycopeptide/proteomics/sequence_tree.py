from contextlib import contextmanager
from collections import defaultdict


def identity(x):
    return x


class TreeReprMixin(object):
    def __repr__(self):
        base = dict(self)
        return repr(base)


class PrefixTree(TreeReprMixin, defaultdict):
    '''
    A hash-based Prefix Tree for testing for
    sequence inclusion. This implementation works for any
    slice-able sequence of hashable objects, not just strings.
    '''
    def __init__(self):
        defaultdict.__init__(self, PrefixTree)
        self.labels = set()

    def add(self, sequence, label=None):
        """Adds a single Sequence-like object to the tree
        structure, placing the ith member of `sequence` on
        the ith level of `self`. At each level, if `label`
        is not None, it is added to the level's `labels` set.

        Parameters
        ----------
        sequence : Sequence
            Sequence-like object to add to the tree
        label : object, optional
            An arbitrary object which denotes or is associated with
            `sequence`

        Returns
        -------
        self
        """
        layer = self
        if label is None:
            label = sequence
        if label:
            layer.labels.add(label)
        for i in range(len(sequence)):
            layer = layer[sequence[i]]
            if label:
                layer.labels.add(label)

        return self

    def add_ngram(self, sequence, label=None):
        """Adds successive prefixes of `sequence`
        to the tree by calling :meth:`add`. This process
        is identical to calling :meth:`add` once for a prefix
        tree.
        
        Parameters
        ----------
        sequence : Sequence
            Sequence-like object to add to the tree
        label : object, optional
            An arbitrary object which denotes or is associated with
            `sequence`
        """
        if label is None:
            label = sequence
        for i in range(1, len(sequence) + 1):
            self.add(sequence[:i], label)

    def __contains__(self, sequence):
        layer = self
        j = 0
        for i in sequence:
            if not dict.__contains__(layer, i):
                break
            layer = layer[i]
            j += 1
        return len(sequence) == j

    def depth_in(self, sequence):
        layer = self
        count = 0
        for i in sequence:
            if not dict.__contains__(layer, i):
                break
            else:
                layer = layer[i]
            count += 1
        return count

    def subsequences_of(self, sequence):
        layer = self
        for i in sequence:
            layer = layer[i]
        return layer.labels

    def __iter__(self):
        return iter(self.labels)


class SuffixTree(PrefixTree):
    '''
    A hash-based Suffix Tree for testing for
    sequence inclusion. This implementation works for any
    slice-able sequence of hashable objects, not just strings.
    '''
    def __init__(self):
        defaultdict.__init__(self, SuffixTree)
        self.labels = set()

    def add_ngram(self, sequence, label=None):
        if label is None:
            label = sequence
        for i in range(len(sequence)):
            self.add(sequence[i:], label=label)


class KeyTransformingPrefixTree(PrefixTree):
    def __init__(self, transformer):
        defaultdict.__init__(self, lambda: KeyTransformingPrefixTree(transformer))
        self.transformer = transformer
        self.labels = set()

    def get(self, key):
        return self[self.transformer(key)]

    def add(self, sequence, label=None):
        layer = self
        t = self.transformer
        if label is None:
            label = sequence
        if label:
            layer.labels.add(label)
        for i in range(len(sequence)):
            layer = layer[t(sequence[i])]
            if label:
                layer.labels.add(label)
        return self

    def __contains__(self, sequence):
        layer = self
        j = 0
        t = self.transformer
        for i in sequence:
            j += 1
            if not dict.__contains__(layer, t(i)):
                break
            layer = layer[i]
        return j == len(sequence)

    def depth_in(self, sequence):
        layer = self
        count = 0
        t = self.transformer
        for i in sequence:
            if not dict.__contains__(layer, t(i)):
                break
            else:
                layer = layer[i]
            count += 1
        return count

    def subsequences_of(self, sequence):
        layer = self
        t = self.transformer
        for i in sequence:
            layer = layer[t(i)]
        return layer.labels

    __repr__ = dict.__repr__

    def __iter__(self):
        return iter(self.labels)

    @contextmanager
    def with_identity(self):
        t = self.transformer
        self.transformer = identity
        yield
        self.transformer = t


class KeyTransformingSuffixTree(KeyTransformingPrefixTree):
    def __init__(self, transformer):
        defaultdict.__init__(self, lambda: KeyTransformingSuffixTree(transformer))
        self.transformer = transformer
        self.labels = set()

    def add_ngram(self, sequence, label=None):
        if label is None:
            label = sequence
        for i in range(len(sequence)):
            self.add(sequence[i:], label=label)
