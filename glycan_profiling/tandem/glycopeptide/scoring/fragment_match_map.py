from collections import defaultdict


class PeakFragmentPair(object):
    __slots__ = ["peak", "fragment", "fragment_name"]

    def __init__(self, peak, fragment):
        self.peak = peak
        self.fragment = fragment
        self.fragment_name = fragment.name

    def __eq__(self, other):
        return (self.peak == other.peak) and (self.fragment_name == other.fragment_name)

    def __hash__(self):
        return hash(self.peak)

    def __reduce__(self):
        return self.__class__, (self.peak, self.fragment)

    def clone(self):
        return self.__class__(self.peak, self.fragment)

    def __repr__(self):
        return "PeakFragmentPair(%r, %r)" % (self.peak, self.fragment)

    def __iter__(self):
        yield self.peak
        yield self.fragment


class FragmentMatchMap(object):
    def __init__(self):
        self.members = set()
        self.by_fragment = defaultdict(list)
        self.by_peak = defaultdict(list)

    def add(self, peak, fragment=None):
        if fragment is not None:
            peak = PeakFragmentPair(peak, fragment)
        if peak not in self.members:
            self.members.add(peak)
            self.by_fragment[peak.fragment].append(peak.peak)
            self.by_peak[peak.peak].append(peak.fragment)

    def fragments_for(self, peak):
        return self.by_peak[peak]

    def peaks_for(self, fragment):
        return self.by_fragment[fragment]

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def items(self):
        for peak, fragment in self.members:
            yield fragment, peak

    def values(self):
        for pair in self.members:
            yield pair.peak

    def fragments(self):
        frags = set()
        for peak, fragment in self:
            frags.add(fragment)
        return iter(frags)

    def remove_fragment(self, fragment):
        peaks = self.peaks_for(fragment)
        for peak in peaks:
            fragments_from_peak = self.by_peak[peak]
            kept = [f for f in fragments_from_peak if f != fragment]
            if len(kept) == 0:
                self.by_peak.pop(peak)
            else:
                self.by_peak[peak] = kept
            self.members.remove(PeakFragmentPair(peak, fragment))
        self.by_fragment.pop(fragment)

    def remove_peak(self, peak):
        fragments = self.fragments_for(peak)
        for fragment in fragments:
            peaks_from_fragment = self.by_fragment[fragment]
            kept = [p for p in peaks_from_fragment if p != peak]
            if len(kept) == 0:
                self.by_fragment.pop(fragment)
            else:
                self.by_fragment[fragment] = kept
            self.members.remove(PeakFragmentPair(peak, fragment))
        self.by_peak.pop(peak)

    def copy(self):
        inst = self.__class__()
        for case in self.members:
            inst.add(case)
        return inst

    def clone(self):
        return self.copy()

    def __repr__(self):
        return "FragmentMatchMap(%s)" % (', '.join(
            f.name for f in self.fragments()),)
