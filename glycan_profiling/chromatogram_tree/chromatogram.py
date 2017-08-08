from abc import ABCMeta
from collections import defaultdict, Counter
from operator import attrgetter

from six import add_metaclass

import numpy as np

from glypy.utils import uid
from glypy.composition.glycan_composition import HashableGlycanComposition


from .mass_shift import Unmodified


MIN_POINTS_FOR_CHARGE_STATE = 3
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


class SubsequenceMasker(object):
    def __init__(self, target, masker):
        self.target = target
        self.masker = masker

        self.masking_nodes = []

    def masking_nodes_from_masker(self):
        self.masking_nodes = self.masker.nodes.unspool()

    def masking_nodes_from_peaks(self, peaks):
        nodes = []
        for node in self.masker.nodes.unspool_strip_children():
            for peak in node.members:
                if peak in peaks:
                    nodes.append(node)
                    break
        self.masking_nodes = nodes

    def mask(self):
        unmasked_nodes = []
        target_nodes = self.target.nodes.unspool_strip_children()
        for node in target_nodes:
            if node not in self.masking_nodes:
                unmasked_nodes.append(node)

        new = self.target.clone()
        new.clear()
        new.used_as_adduct = []
        new.created_at = "mask_subsequence"
        for node in unmasked_nodes:
            new.insert_node(node)
        return new

    @classmethod
    def mask_subsequence(cls, target, masker, peaks=None):
        inst = cls(target, masker)
        if peaks is None:
            inst.masking_nodes_from_masker()
        else:
            inst.masking_nodes_from_peaks(peaks)
        return inst.mask()


mask_subsequence = SubsequenceMasker.mask_subsequence


class Chromatogram(object):
    created_at = "new"
    glycan_composition = None

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
        self._weighted_neutral_mass = None
        self._last_neutral_mass = 0.
        self._most_abundant_member = 0.
        self._charge_states = None
        self._retention_times = None
        self._peaks = None
        self._scan_ids = None
        self._start_time = None
        self._end_time = None

    def _invalidate(self):
        self._total_intensity = None
        self._last_neutral_mass = self._neutral_mass if self._neutral_mass is not None else 0.
        self._neutral_mass = None
        self._weighted_neutral_mass = None
        self._charge_states = None
        self._retention_times = None
        self._peaks = None
        self._scan_ids = None
        self._adducts = None
        self._has_msms = None
        self._start_time = None
        self._end_time = None
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
        if not adducts:
            adducts = [Unmodified]
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
    def integrated_abundance(self):
        spacing = np.array([
            node.retention_time for node in self.nodes
        ])
        values = np.array([
            node.total_intensity() for node in self.nodes
        ])
        integrated = np.trapz(values, spacing)
        return integrated

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
        if self._weighted_neutral_mass is None:
            try:
                self._infer_neutral_mass()
            except KeyError:
                self._infer_neutral_mass(self.adducts[0])
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
            states = Counter()
            for node in self.nodes:
                states += (Counter(node.charge_states()))
            # Require more than `MIN_POINTS_FOR_CHARGE_STATE` data points to accept any
            # charge state
            collapsed_states = {k for k, v in states.items() if v >= min(MIN_POINTS_FOR_CHARGE_STATE, len(self))}
            if not collapsed_states:
                collapsed_states = states.keys()
            self._charge_states = collapsed_states
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
        if self._start_time is None:
            self._start_time = self.nodes[0].retention_time
        return self._start_time

    @property
    def end_time(self):
        if self._end_time is None:
            self._end_time = self.nodes[-1].retention_time
        return self._end_time

    def as_arrays(self):
        rts = np.array([node.retention_time for node in self.nodes], dtype=np.float64)
        signal = np.array([node.total_intensity() for node in self.nodes], dtype=np.float64)
        return rts, signal

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
                x = self.__class__(self.composition, ChromatogramTreeList(current_chunk))
                x.used_as_adduct = list(self.used_as_adduct)

                chunks.append(x)
                current_chunk = []

            last_rt = node.retention_time
            current_chunk.append(node)

        x = self.__class__(self.composition, ChromatogramTreeList(current_chunk))
        x.used_as_adduct = list(self.used_as_adduct)

        chunks.append(x)
        for chunk in chunks:
            chunk.created_at = self.created_at

        for member in chunks:
            for other in chunks:
                if member == other:
                    continue
                assert not member.overlaps_in_time(other)

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

    def clone(self, cls=None):
        if cls is None:
            cls = self.__class__
        c = cls(
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

    def _merge_missing_only(self, other, node_type=Unmodified):
        new = self.clone()
        ids = set(node.node_id for node in new.nodes.unspool())
        for node in other.nodes.unspool_strip_children():
            if node.node_id in ids:
                continue
            else:
                new_node = node.clone()
                new_node.node_type = node.node_type + node_type
                new.insert_node(new_node)
        new.created_at = "_merge_common_only"
        return new

    @classmethod
    def from_parts(cls, composition, retention_times, scan_ids, peaks):
        nodes = ChromatogramTreeList()
        nodes.extend(zip(scan_ids, peaks, retention_times))
        return cls(composition, nodes)

    def slice(self, start, end):
        _, i = self.nodes.find_time(start)
        _, j = self.nodes.find_time(end)
        new = self.__class__(
            self.composition,
            ChromatogramTreeList(node.clone() for node in self.nodes[i:j + 1]),
            used_as_adduct=list(self.used_as_adduct))
        return new

    def bisect_adduct(self, adduct):
        new_adduct = self.__class__(self.composition)
        new_no_adduct = self.__class__(self.composition)
        for node in self:
            for new_node in node._unspool_strip_children():
                if new_node.node_type == adduct:
                    new_adduct.insert_node(new_node)
                else:
                    new_no_adduct.insert_node(new_node)
        return new_adduct, new_no_adduct

    def bisect_charge(self, charge):
        new_charge = self.__class__(self.composition)
        new_no_charge = self.__class__(self.composition)
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
            if len(charge_peaks):
                charge_node = ChromatogramTreeNode(
                    retention_time=rt, scan_id=scan_id, children=None,
                    members=charge_peaks, node_type=node_t)
                new_charge.insert_node(charge_node)
            if len(no_charge_peaks):
                no_charge_node = ChromatogramTreeNode(
                    retention_time=rt, scan_id=scan_id, children=None,
                    members=no_charge_peaks, node_type=node_t)
                new_no_charge.insert_node(no_charge_node)
        return new_charge, new_no_charge

    def __eq__(self, other):
        if other is None:
            return False
        if self.key != other.key:
            return False
        else:
            return self.peaks == other.peaks

    def __ne__(self, other):
        return not self == other

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

    @property
    def elemental_composition(self):
        return None

    def common_nodes(self, other):
        return self.nodes.common_nodes(other.nodes)

    @property
    def apex_time(self):
        rt, intensity = self.as_arrays()
        return rt[np.argmax(intensity)]

    def extract_components(self):
        adducts = list(self.adducts)
        labels = {}
        rest = self
        for adduct in adducts:
            with_adduct, rest = rest.bisect_adduct(adduct)
            labels[adduct] = with_adduct

        labels[Unmodified] = rest
        adduct_charge_table = defaultdict(dict)
        for adduct, component in labels.items():
            charges = list(component.charge_states)
            rest = component
            for charge in charges:
                selected, rest = rest.bisect_charge(charge)
                adduct_charge_table[adduct][charge] = selected
        return adduct_charge_table

    def clear(self):
        self.nodes.clear()
        self._invalidate()


class ChromatogramTreeList(object):
    def __init__(self, roots=None):
        if roots is None:
            roots = []
        self.roots = list(roots)
        self._node_id_hash = None
        self._peak_hash = None

    def _invalidate(self):
        self._node_id_hash = None
        self._peak_hash = None

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

    def _build_node_id_hash(self):
        node_id_hash = set()
        for node in self.unspool():
            node_id_hash.add(node.node_id)
        self._node_id_hash = frozenset(node_id_hash)

    def _build_peak_hash(self):
        peak_hash = set()
        for node in self.unspool():
            peak_hash.update([(node.scan_id, peak) for peak in node.members])
        self._peak_hash = frozenset(peak_hash)

    @property
    def node_id_hash(self):
        if self._node_id_hash is None:
            self._build_node_id_hash()
        return self._node_id_hash

    @property
    def peak_hash(self):
        if self._peak_hash is None:
            self._build_peak_hash()
        return self._peak_hash

    def insert_node(self, node):
        self._invalidate()
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

    def clear(self):
        self.roots = []
        self._invalidate()

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

    def common_nodes(self, other):
        return not self.node_id_hash.isdisjoint(other.node_id_hash)

    def common_peaks(self, other):
        return not self.peak_hash.isdisjoint(other.peak_hash)

    def __repr__(self):
        return "ChromatogramTreeList(%d nodes, %0.2f-%0.2f)" % (
            len(self), self[0].retention_time, self[-1].retention_time)


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
        if self._neutral_mass is None:
            if self._most_abundant_member is not None:
                self._neutral_mass = self._most_abundant_member.neutral_mass
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


@add_metaclass(ABCMeta)
class ChromatogramInterface(object):
    pass


ChromatogramInterface.register(Chromatogram)


class ChromatogramWrapper(object):
    def __init__(self, chromatogram):
        self.chromatogram = chromatogram

    def __iter__(self):
        return iter(self.chromatogram)

    def __getitem__(self, i):
        return self.chromatogram[i]

    def __len__(self):
        return len(self.chromatogram)

    def __hash__(self):
        return hash(self.chromatogram)

    def __eq__(self, other):
        try:
            return self.chromatogram == get_chromatogram(other)
        except Exception:
            return False

    @property
    def nodes(self):
        return self.chromatogram.nodes

    @property
    def neutral_mass(self):
        return self.chromatogram.neutral_mass

    @property
    def weighted_neutral_mass(self):
        return self.chromatogram.weighted_neutral_mass

    @property
    def n_charge_states(self):
        return self.chromatogram.n_charge_states

    @property
    def charge_states(self):
        return self.chromatogram.charge_states

    @property
    def adducts(self):
        return self.chromatogram.adducts

    @property
    def total_signal(self):
        return self.chromatogram.total_signal

    @property
    def integrated_abundance(self):
        return self.chromatogram.integrated_abundance

    @property
    def start_time(self):
        return self.chromatogram.start_time

    @property
    def end_time(self):
        return self.chromatogram.end_time

    def as_arrays(self):
        return self.chromatogram.as_arrays()

    def overlaps_in_time(self, interval):
        return self.chromatogram.overlaps_in_time(interval)

    @property
    def key(self):
        return self.chromatogram.key

    @property
    def composition(self):
        return self.chromatogram.composition

    @property
    def peaks(self):
        return self.chromatogram.peaks

    @property
    def scan_ids(self):
        return self.chromatogram.scan_ids

    @property
    def retention_times(self):
        return self.chromatogram.retention_times

    def __repr__(self):
        return "{self.__class__.__name__}({self.key}, {self.neutral_mass})".format(self=self)

    @property
    def entity(self):
        return self.chromatogram.entity

    @entity.setter
    def entity(self, value):
        self.chromatogram.entity = value

    @property
    def glycan_composition(self):
        return self.chromatogram.glycan_composition

    @property
    def elemental_composition(self):
        return self.chromatogram.elemental_composition

    def common_nodes(self, other):
        return self.chromatogram.common_nodes(other)

    @property
    def apex_time(self):
        return self.chromatogram.apex_time

    def __getattr__(self, name):
        if name == 'chromatogram':
            raise AttributeError(name)
        else:
            return getattr(self.chromatogram, name)

    def clone(self):
        chromatogram = self.chromatogram.clone()
        new = self.__class__(chromatogram)
        return new

    def clear(self):
        self.chromatogram.clear()


ChromatogramInterface.register(ChromatogramWrapper)


class CachedGlycanComposition(HashableGlycanComposition):
    _hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(str(self))
        return self._hash


class EntityChromatogram(Chromatogram):
    _entity = None

    @property
    def glycan_composition(self):
        return self._entity

    def clone(self, cls=None):
        if cls is None:
            cls = self.__class__
        inst = super(EntityChromatogram, self).clone(cls=cls)
        inst.entity = self.entity
        return inst

    @property
    def entity(self):
        return self._entity

    @entity.setter
    def entity(self, value):
        if isinstance(value, str):
            value = self._parse(value)
        self._entity = value
        if self.composition is None and value is not None:
            self.composition = value

    @property
    def elemental_composition(self):
        try:
            return self.entity.total_composition()
        except AttributeError:
            try:
                return self._parse(self.composition).total_composition()
            except Exception:
                return None


class GlycanCompositionChromatogram(EntityChromatogram):
    @staticmethod
    def _parse(string):
        return CachedGlycanComposition.parse(string)

    @property
    def glycan_composition(self):
        if isinstance(self.composition, str):
            self.composition = self._parse(self.composition)
        return self.composition


class GlycopeptideChromatogram(EntityChromatogram):
    @staticmethod
    def _parse(string):
        from glycopeptidepy.structure.sequence import PeptideSequence
        return PeptideSequence(string)

    @property
    def glycan_composition(self):
        return self._entity.glycan_composition


def get_chromatogram(instance):
    if isinstance(instance, Chromatogram):
        return instance
    elif isinstance(instance, ChromatogramWrapper):
        return instance.chromatogram
    elif isinstance(instance, ChromatogramInterface):
        return instance
    else:
        try:
            return instance.get_chromatogram()
        except AttributeError:
            raise TypeError(
                "%s does not contain a chromatogram" % instance)
