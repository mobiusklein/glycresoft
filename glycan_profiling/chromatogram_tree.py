from abc import ABCMeta
from collections import defaultdict
from operator import attrgetter

import numpy as np

from glypy.utils import uid
from glypy import Composition
from glypy.composition.glycan_composition import FrozenGlycanComposition


intensity_getter = attrgetter("intensity")


class EmptyListException(Exception):
    pass


class DuplicateNodeError(Exception):
    pass


def group_by(ungrouped_list, key_fn=lambda x: x, transform_fn=lambda x: x):
    groups = defaultdict(list)
    for item in ungrouped_list:
        key_value = key_fn(item)
        groups[key_value].append(transform_fn(item))
    return groups


def split_by_charge(peaks):
    return group_by(peaks, lambda x: x.charge)


def count_charge_states(peaks):
    peaks = [j for i in peaks for j in i]
    return len(split_by_charge(peaks))


class MassShiftBase(object):
    def __eq__(self, other):
        try:
            return (self.name == other.name and abs(
                self.mass - other.mass) < 1e-10) or self.composition == other.composition
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)


class MassShift(MassShiftBase):
    def __init__(self, name, composition):
        self.name = intern(name)
        self.composition = composition
        self.mass = composition.mass

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)

    def __mul__(self, n):
        if self.composition == {}:
            return self
        if isinstance(n, int):
            return CompoundMassShift({self: n})
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __add__(self, other):
        if self.composition == {}:
            return other
        elif other.composition == {}:
            return self
        name = "(%s) + (%s)" % (self.name, other.name)
        composition = self.composition + other.composition
        return self.__class__(name, composition)


class CompoundMassShift(MassShiftBase):
    def __init__(self, counts=None):
        if counts is None:
            counts = {}
        self.counts = defaultdict(int, counts)
        self.composition = None
        self.name = None
        self.mass = None

        self._compute_composition()
        self._compute_name()

    def _compute_composition(self):
        composition = Composition()
        for k, v in self.counts.items():
            composition += k.composition * v
        self.composition = composition
        self.mass = composition.mass

    def _compute_name(self):
        parts = []
        for k, v in self.counts.items():
            if v == 1:
                parts.append(k.name)
            else:
                parts.append("%s * %d" % (k.name, v))
        self.name = intern(" + ".join(sorted(parts)))

    def __add__(self, other):
        if other == Unmodified:
            return self
        elif self == Unmodified:
            return other

        if isinstance(other, MassShift):
            counts = defaultdict(int, self.counts)
            counts[other] += 1
            return self.__class__(counts)
        elif isinstance(other, CompoundMassShift):
            counts = defaultdict(int, self.counts)
            for k, v in other.counts.items():
                counts[k] += v
            return self.__class__(counts)
        else:
            return NotImplemented

    def __mul__(self, i):
        if self.composition == {}:
            return self
        if isinstance(i, int):
            counts = defaultdict(int, self.counts)
            for k in counts:
                if k == Unmodified:
                    continue
                counts[k] *= i
            return self.__class__(counts)
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)


Unmodified = MassShift("Unmodified", Composition())
Formate = MassShift("Formate", Composition('HCOOH'))
Ammonium = MassShift("Ammonium", Composition("NH4"))
Sodiated = MassShift("Sodiated", Composition("Na"))


class Chromatogram(object):
    created_at = "new"

    def __init__(self, composition, nodes=None, adducts=None, used_as_adduct=None):
        if nodes is None:
            nodes = ChromatogramTreeList()
        if adducts is None:
            adducts = []
        if used_as_adduct is None:
            used_as_adduct = []
        self.nodes = nodes
        self._adducts = adducts
        self.used_as_adduct = used_as_adduct
        self._infer_adducts()
        self._has_msms = None

        self.composition = composition
        self._total_intensity = None
        self._neutral_mass = None
        self._last_neutral_mass = 0.
        self._most_abundant_member = 0.
        self._charge_states = None
        self._retention_times = None
        self._peaks = None
        self._scan_ids = None

    def _invalidate(self):
        self._total_intensity = None
        self._last_neutral_mass = self._neutral_mass if self._neutral_mass is not None else 0.
        self._neutral_mass = None
        self._charge_states = None
        self._retention_times = None
        self._peaks = None
        self._scan_ids = None
        self._adducts = None
        self._has_msms = None
        self._last_most_abundant_member = self._most_abundant_member
        self._most_abundant_member = None

    def retain_most_abundant_member(self):
        self._neutral_mass = self._last_neutral_mass
        self._most_abundant_member = self._last_most_abundant_member

    @property
    def has_msms(self):
        if self._has_msms is None:
            self._has_msms = [node for node in self.nodes if node.has_msms]
        return self._has_msms

    @property
    def most_abundant_member(self):
        if self._most_abundant_member is None:
            self._most_abundant_member = max(node.max_intensity() for node in self.nodes)
        return self._most_abundant_member

    def _infer_adducts(self):
        adducts = set()
        for node in self.nodes:
            adducts.update(node.node_types())
        self._adducts = list(adducts)

    def __getitem__(self, i):
        return self.nodes[i]

    def __iter__(self):
        return iter(self.nodes)

    def total_signal_for(self, node_type=Unmodified):
        total = 0.
        for node in self.nodes:
            node = node._find(node_type)
            if node is not None:
                total += node._total_intensity_members()
        return total

    def adduct_signal_fractions(self):
        return {
            k: self.total_signal_for(k) for k in self.adducts
        }

    @property
    def adducts(self):
        if self._adducts is None:
            self._infer_adducts()
        return self._adducts

    @property
    def total_signal(self):
        if self._total_intensity is None:
            total = 0.
            for node in self.nodes:
                total += node.total_intensity()
            self._total_intensity = total
        return self._total_intensity

    @property
    def weighted_neutral_mass(self):
        if self._neutral_mass is None:
            self._infer_neutral_mass()
        return self._weighted_neutral_mass

    def _infer_neutral_mass(self, node_type=Unmodified):
        prod = 0
        total = 0
        maximum_intensity = 0
        best_neutral_mass = 0
        for node in self.nodes:
            intensity = node.max_intensity()
            try:
                mass = node.neutral_mass_of_type(node_type)
            except KeyError:
                continue
            prod += intensity * mass
            total += intensity
            if intensity > maximum_intensity:
                maximum_intensity = intensity
                best_neutral_mass = mass
        if total > 0:
            self._weighted_neutral_mass = prod / total - node_type.mass
        else:
            self._weighted_neutral_mass = best_neutral_mass - node_type.mass
        self._last_neutral_mass = self._neutral_mass = best_neutral_mass - node_type.mass
        if self._neutral_mass == 0:
            raise KeyError(node_type)
        return best_neutral_mass

    @property
    def neutral_mass(self):
        if self._neutral_mass is None:
            try:
                self._infer_neutral_mass()
            except KeyError:
                self._infer_neutral_mass(self.adducts[0])
        return self._neutral_mass

    @property
    def charge_states(self):
        if self._charge_states is None:
            states = set()
            for node in self.nodes:
                states.update(node.charge_states())
            self._charge_states = states
        return self._charge_states

    @property
    def n_charge_states(self):
        return len(self.charge_states)

    @property
    def key(self):
        if self.composition is not None:
            return self.composition
        else:
            return self.neutral_mass

    @property
    def retention_times(self):
        if self._retention_times is None:
            self._retention_times = tuple(node.retention_time for node in self.nodes)
        return self._retention_times

    @property
    def scan_ids(self):
        if self._scan_ids is None:
            self._scan_ids = tuple(node.scan_id for node in self.nodes)
        return self._scan_ids

    @property
    def peaks(self):
        if self._peaks is None:
            self._peaks = tuple(node.peaks for node in self.nodes)
        return self._peaks

    @property
    def start_time(self):
        return self.nodes[0].retention_time

    @property
    def end_time(self):
        return self.nodes[-1].retention_time

    def as_arrays(self):
        rts = np.array([node.retention_time for node in self.nodes])
        intens = np.array([node.total_intensity() for node in self.nodes])
        return rts, intens

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return "Chromatogram(%s, %0.4f)" % (self.composition, self.neutral_mass)

    def split_sparse(self, delta_rt=1.):
        chunks = []
        current_chunk = []
        last_rt = self.nodes[0].retention_time

        for node in self.nodes:
            if (node.retention_time - last_rt) > delta_rt:
                x = Chromatogram(self.composition, ChromatogramTreeList(current_chunk))
                x.used_as_adduct = list(self.used_as_adduct)
                chunks.append(x)
                current_chunk = []

            last_rt = node.retention_time
            current_chunk.append(node)

        x = Chromatogram(self.composition, ChromatogramTreeList(current_chunk))
        x.used_as_adduct = list(self.used_as_adduct)
        chunks.append(x)
        for chunk in chunks:
            chunk.created_at = self.created_at
        return chunks

    def truncate_before(self, time):
        _, i = self.nodes.find_time(time)
        if self.nodes[i].retention_time < time:
            i += 1
        self.nodes = ChromatogramTreeList(self.nodes[i:])
        self._invalidate()

    def truncate_after(self, time):
        _, i = self.nodes.find_time(time)
        if self.nodes[i].retention_time < time:
            i += 1
        self.nodes = ChromatogramTreeList(self.nodes[:i])
        self._invalidate()

    def clone(self):
        c = Chromatogram(
            self.composition, self.nodes.clone(), list(self.adducts), list(self.used_as_adduct))
        c.created_at = self.created_at
        return c

    def insert_node(self, node):
        self.nodes.insert_node(node)
        self._invalidate()

    def insert(self, scan_id, peak, retention_time):
        self.nodes.insert(retention_time, scan_id, [peak])
        self._invalidate()

    def merge(self, other, node_type=Unmodified):
        new = self.clone()
        for node in other.nodes.unspool_strip_children():
            node = node.clone()
            node.node_type = node.node_type + node_type
            new.insert_node(node)
        new.created_at = "merge"
        return new

    @classmethod
    def from_parts(cls, composition, retention_times, scan_ids, peaks):
        nodes = ChromatogramTreeList()
        nodes.extend(zip(scan_ids, peaks, retention_times))
        return cls(composition, nodes)

    def slice(self, start, end):
        _, i = self.nodes.find_time(start)
        _, j = self.nodes.find_time(end)
        new = Chromatogram(
            self.composition,
            ChromatogramTreeList(node.clone() for node in self.nodes[i:j + 1]),
            used_as_adduct=list(self.used_as_adduct))
        return new

    def bisect_adduct(self, adduct):
        new_adduct = Chromatogram(self.composition)
        new_no_adduct = Chromatogram(self.composition)
        for node in self:
            for new_node in node._unspool_strip_children():
                if new_node.node_type == adduct:
                    new_adduct.insert_node(new_node)
                else:
                    new_no_adduct.insert_node(new_node)
        return new_adduct, new_no_adduct

    def bisect_charge(self, charge):
        new_charge = Chromatogram(self.composition)
        new_no_charge = Chromatogram(self.composition)
        for node in self.nodes.unspool():
            node_t = node.node_type
            rt = node.retention_time
            scan_id = node.scan_id
            peaks = node.members
            charge_peaks = []
            no_charge_peaks = []
            for peak in peaks:
                if peak.charge == charge:
                    charge_peaks.append(peak)
                else:
                    no_charge_peaks.append(peak)
            charge_node = ChromatogramTreeNode(
                retention_time=rt, scan_id=scan_id, children=None,
                members=charge_peaks, node_type=node_t)
            new_charge.insert_node(charge_node)
            no_charge_node = ChromatogramTreeNode(
                retention_time=rt, scan_id=scan_id, children=None,
                members=no_charge_peaks, node_type=node_t)
            new_no_charge.insert_node(no_charge_node)
        return new_charge, new_no_charge

    def __eq__(self, other):
        if self.key != other.key:
            return False
        else:
            return self.peaks == other.peaks

    def __hash__(self):
        return hash((self.neutral_mass, self.start_time, self.end_time))

    def overlaps_in_time(self, interval):
        cond = ((self.start_time <= interval.start_time and self.end_time >= interval.end_time) or (
            self.start_time >= interval.start_time and self.end_time <= interval.end_time) or (
            self.start_time >= interval.start_time and self.end_time >= interval.end_time and
            self.start_time <= interval.end_time) or (
            self.start_time <= interval.start_time and self.end_time >= interval.start_time) or (
            self.start_time <= interval.end_time and self.end_time >= interval.end_time))
        return cond


class ChromatogramTreeList(object):
    def __init__(self, roots=None):
        if roots is None:
            roots = []
        self.roots = list(roots)

    def find_time(self, retention_time):
        if len(self.roots) == 0:
            raise EmptyListException()
        lo = 0
        hi = len(self.roots)
        while lo != hi:
            i = (lo + hi) / 2
            node = self.roots[i]
            if node.retention_time == retention_time:
                return node, i
            elif (hi - lo) == 1:
                return None, i
            elif node.retention_time < retention_time:
                lo = i
            elif node.retention_time > retention_time:
                hi = i

    def insert_node(self, node):
        try:
            root, i = self.find_time(node.retention_time)
            if root is None:
                if i != 0:
                    self.roots.insert(i + 1, node)
                else:
                    slot = self.roots[i]
                    if slot.retention_time < node.retention_time:
                        i += 1
                    self.roots.insert(i, node)
            else:
                root.add(node)
            return i
        except EmptyListException:
            self.roots.append(node)
            return 0

    def insert(self, retention_time, scan_id, peaks, node_type=Unmodified):
        node = ChromatogramTreeNode(retention_time, scan_id, [], peaks, node_type)
        return self.insert_node(node)

    def extend(self, iterable):
        for scan_id, peaks, retention_time in iterable:
            self.insert(retention_time, scan_id, peaks)

    def __getitem__(self, i):
        return self.roots[i]

    def __len__(self):
        return len(self.roots)

    def __iter__(self):
        return iter(self.roots)

    def clone(self):
        return ChromatogramTreeList(node.clone() for node in self)

    def unspool(self):
        out_queue = []
        for root in self:
            stack = [root]
            while len(stack) != 0:
                node = stack.pop()
                out_queue.append(node)
                stack.extend(node.children)
        return out_queue

    def unspool_strip_children(self):
        out_queue = []
        for root in self:
            stack = [root]
            while len(stack) != 0:
                node = stack.pop()
                node_copy = node.clone()
                node_copy.children = []
                out_queue.append(node_copy)
                stack.extend(node.children)
        return out_queue


class ChromatogramTreeNode(object):
    def __init__(self, retention_time=None, scan_id=None, children=None, members=None,
                 node_type=Unmodified):
        if children is None:
            children = []
        if members is None:
            members = []
        self.retention_time = retention_time
        self.scan_id = scan_id
        self.children = children
        self.members = members
        self.node_type = node_type
        self._most_abundant_member = None
        self._neutral_mass = 0
        self._charge_states = set()
        self._recalculate()
        self._has_msms = None
        self.node_id = uid()

    def clone(self):
        node = ChromatogramTreeNode(
            self.retention_time, self.scan_id, [c.clone() for c in self.children],
            list(self.members), node_type=self.node_type)
        node.node_id = self.node_id
        return node

    def _unspool_strip_children(self):
        node = ChromatogramTreeNode(
            self.retention_time, self.scan_id, [], list(self.members), node_type=self.node_type)
        yield node
        for child in self.children:
            for node in child._unspool_strip_children():
                yield node

    def _calculate_most_abundant_member(self):
        if len(self.members) == 1:
            self._most_abundant_member = self.members[0]
        else:
            if len(self.members) == 0:
                self._most_abundant_member = None
            else:
                self._most_abundant_member = max(self.members, key=intensity_getter)

    def _recalculate(self):
        self._calculate_most_abundant_member()
        self._neutral_mass = self._most_abundant_member.neutral_mass
        self._charge_states = None
        self._has_msms = None

    @property
    def _contained_charge_states(self):
        if self._charge_states is None:
            self._charge_states = set(split_by_charge(self.members))
        return self._charge_states

    @property
    def has_msms(self):
        if self._has_msms is None:
            self._has_msms = self._has_any_peaks_with_msms()
        return self._has_msms

    def _find(self, node_type=Unmodified):
        if self.node_type == node_type:
            return self
        else:
            for child in self.children:
                match = child._find(node_type)
                if match is not None:
                    return match

    def find(self, node_type=Unmodified):
        match = self._find(node_type)
        if match is not None:
            return match
        else:
            raise KeyError(node_type)

    @property
    def neutral_mass(self):
        return self._neutral_mass

    def charge_states(self):
        u = set()
        u.update(self._contained_charge_states)
        for child in self.children:
            u.update(child.charge_states())
        return u

    def neutral_mass_of_type(self, node_type=Unmodified):
        return self.find(node_type)._neutral_mass

    def add(self, node, recalculate=True):
        if self.node_id == node.node_id:
            raise DuplicateNodeError("Duplicate Node %s" % node)
        if node.node_type == self.node_type:
            self.members.extend(node.members)
        else:
            self.children.append(node)
        if recalculate:
            self._recalculate()

    def _total_intensity_members(self):
        total = 0.
        for peak in self.members:
            total += peak.intensity
        return total

    def _total_intensity_children(self):
        total = 0.
        for child in self.children:
            total += child.total_intensity()
        return total

    def max_intensity(self):
        return self._most_abundant_member.intensity

    def total_intensity(self):
        return self._total_intensity_children() + self._total_intensity_members()

    def __eq__(self, other):
        return self.members == other.members

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.uid)

    def _has_any_peaks_with_msms(self):
        for peak in self.members:
            if peak.chosen_for_msms:
                return True
        for child in self.children:
            if child._has_any_peaks_with_msms():
                return True
        return False

    @property
    def peaks(self):
        peaks = list(self.members)
        for child in self.children:
            peaks.extend(child.peaks)
        return peaks

    def __repr__(self):
        return "ChromatogramTreeNode(%f, %r, %s|%d, %d)" % (
            self.retention_time, self.scan_id, self.node_type.name,
            len(self.members), len(self.children))

    def node_types(self):
        kinds = [self.node_type]
        for child in self.children:
            kinds.extend(child.node_types())
        return kinds


class ChromatogramForest(object):
    def __init__(self, chromatograms=None, error_tolerance=1e-5, scan_id_to_rt=lambda x: x):
        if chromatograms is None:
            chromatograms = []
        self.chromatograms = sorted(chromatograms, key=lambda x: x.neutral_mass)
        self.error_tolerance = error_tolerance
        self.scan_id_to_rt = scan_id_to_rt
        self.count = 0

    def __len__(self):
        return len(self.chromatograms)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.chromatograms[i]
        else:
            return [self.chromatograms[j] for j in i]

    def find_insertion_point(self, peak):
        index, matched = binary_search_with_flag(
            self.chromatograms, peak.neutral_mass, self.error_tolerance)
        return index, matched

    def find_minimizing_index(self, peak, indices):
        best_index = None
        best_error = float('inf')
        for index_case in indices:
            chroma = self[index_case]
            err = abs(chroma.neutral_mass - peak.neutral_mass) / peak.neutral_mass
            if err < best_error:
                best_index = index_case
                best_error = err
        return best_index

    def handle_peak(self, scan_id, peak):
        if len(self) == 0:
            index = [0]
            matched = False
        else:
            index, matched = self.find_insertion_point(peak)
        if matched:
            chroma = self.chromatograms[self.find_minimizing_index(peak, index)]
            most_abundant_member = chroma.most_abundant_member
            chroma.insert(scan_id, peak, self.scan_id_to_rt(scan_id))
            if peak.intensity < most_abundant_member:
                chroma.retain_most_abundant_member()
        else:
            chroma = Chromatogram(None)
            chroma.created_at = "forest"
            chroma.insert(scan_id, peak, self.scan_id_to_rt(scan_id))
            self.insert_chromatogram(chroma, index)
        self.count += 1

    def insert_chromatogram(self, chromatogram, index):
        if index[0] != 0:
            self.chromatograms.insert(index[0] + 1, chromatogram)
        else:
            if len(self) == 0:
                new_index = index[0]
            else:
                x = self.chromatograms[index[0]]
                if x.neutral_mass < chromatogram.neutral_mass:
                    new_index = index[0] + 1
                else:
                    new_index = index[0]
            self.chromatograms.insert(new_index, chromatogram)

    def aggregate_unmatched_peaks(self, scan_id_peaks_list, minimum_mass=300, minimum_intensity=1000.):
        unmatched = sorted(scan_id_peaks_list, key=lambda x: x[1].intensity, reverse=True)
        for scan_id, peak in unmatched:
            if peak.neutral_mass < minimum_mass or peak.intensity < minimum_intensity:
                continue
            self.handle_peak(scan_id, peak)


def mask_subsequence(target, masker):
    unmasked_nodes = []
    target_nodes = target.nodes.unspool_strip_children()
    masking_nodes = masker.nodes.unspool()
    for node in target_nodes:
        if node not in masking_nodes:
            unmasked_nodes.append(node)
    new = Chromatogram(target.composition)
    new.created_at = "mask_subsequence"
    map(new.insert_node, unmasked_nodes)
    return new


def is_sorted(mass_list):
    i = 0
    for a, b in zip(mass_list[:-1], mass_list[1:]):
        if not a.neutral_mass <= b.neutral_mass:
            print a.neutral_mass, b.neutral_mass, i
        i += 1
    return True


def is_sparse(mass_list):
    i = 0
    for a, b in zip(mass_list[:-1], mass_list[1:]):
        err = (a.neutral_mass - b.neutral_mass) / b.neutral_mass
        if abs(err) < 1e-5 and a.composition is None:
            print a.neutral_mass, b.neutral_mass, err, i
            raise ValueError("Not sparse")
            return False
        i += 1
    return True


def binary_search_with_flag(array, mass, error_tolerance=1e-5):
        lo = 0
        n = hi = len(array)
        while hi != lo:
            mid = (hi + lo) / 2
            x = array[mid]
            err = (x.neutral_mass - mass) / mass
            if abs(err) <= error_tolerance:
                i = mid - 1
                # Begin Sweep forward
                while i > 0:
                    x = array[i]
                    err = (x.neutral_mass - mass) / mass
                    if abs(err) <= error_tolerance:
                        i -= 1
                        continue
                    else:
                        break
                low_end = i + 1
                i = mid + 1

                # Begin Sweep backward
                while i < n:
                    x = array[i]
                    err = (x.neutral_mass - mass) / mass
                    if abs(err) <= error_tolerance:
                        i += 1
                        continue
                    else:
                        break
                high_end = i
                return list(range(low_end, high_end)), True
            elif (hi - lo) == 1:
                return [mid], False
            elif err > 0:
                hi = mid
            elif err < 0:
                lo = mid
        return 0, False


class ChromatogramGraphNode(object):
    def __init__(self, chromatogram):
        self.chromatogram = chromatogram
        self.edges = set()

    def __eq__(self, other):
        return self.chromatogram == other.chromatogram

    def __hash__(self):
        return hash(self.chromatogram.key)


class ChromatogramInterface(object):
    __metaclass__ = ABCMeta


ChromatogramInterface.register(Chromatogram)


class CachedGlycanComposition(FrozenGlycanComposition):
    _hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(str(self))
        return self._hash


class GlycanCompositionChromatogram(Chromatogram):
    _composition = None
    _parsed_composition = None

    @property
    def composition(self):
        if self._composition is None:
            return None
        elif self._parsed_composition is None:
            self._parsed_composition = CachedGlycanComposition.parse(self._composition)
        return self._parsed_composition

    @composition.setter
    def composition(self, value):
        if value is None:
            self._composition = None
        else:
            self._composition = str(value)
            self._parsed_composition = None
