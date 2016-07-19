from collections import defaultdict

import numpy as np

import glypy
from glypy.utils import uid


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


class MassShift(object):
    def __init__(self, name, composition):
        self.name = name
        self.composition = composition
        self.mass = composition.mass

    def __repr__(self):
        return "MassShift(%s, %s)" % (self.name, self.composition)

    def __mul__(self, n):
        if isinstance(n, int):
            return self.__class__("%d * %s" % (n, self.name), self.composition * n)
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")

    def __eq__(self, other):
        return self.name == other.name and abs(self.mass - other.mass) < 1e-6

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)


Unmodified = MassShift("Unmodified", glypy.Composition())
Formate = MassShift("Formate", glypy.Composition('HCOOH'))


class Chromatogram(object):
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

        self.composition = composition
        self._total_intensity = None
        self._neutral_mass = None
        self._charge_states = None
        self._retention_times = None
        self._peaks = None
        self._scan_ids = None

    def _invalidate(self):
        self._total_intensity = None
        self._neutral_mass = None
        self._charge_states = None
        self._retention_times = None
        self._peaks = None
        self._scan_ids = None
        self._adducts = None

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
        self._neutral_mass = best_neutral_mass - node_type.mass
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
        return chunks

    def truncate_before(self, time):
        _, i = self.nodes.find_time(time)
        if self.nodes[i].retention_time < time:
            i += 1
        self.nodes = ChromatogramTreeList(self.nodes[i + 1:])
        self._invalidate()

    def truncate_after(self, time):
        _, i = self.nodes.find_time(time)
        if self.nodes[i].retention_time < time:
            i += 1
        self.nodes = ChromatogramTreeList(self.nodes[:i])
        self._invalidate()

    def clone(self):
        return Chromatogram(
            self.composition, self.nodes.clone(), list(self.adducts), list(self.used_as_adduct))

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
            node.node_type = node_type
            new.insert_node(node)
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


class ChromatogramTreeList(object):
    def __init__(self, roots=None):
        if roots is None:
            roots = []
        self.roots = list(roots)

    def find_time(self, retention_time):
        if len(self.roots) == 0:
            raise ValueError("Empty!")
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
        except ValueError:
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
        self._neutral_mass = 0
        self._charge_states = set()
        self._recalculate()
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

    def _recalculate(self):
        self._neutral_mass = max(self.members, key=lambda x: x.intensity).neutral_mass
        self._charge_states = set(split_by_charge(self.members))

    def _find(self, node_type=Unmodified):
        if self.node_type == node_type:
            return self
        else:
            for child in self.children:
                match = child._find(node_type)
                if match:
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
        u.update(self._charge_states)
        for child in self.children:
            u.update(child.charge_states())
        return u

    def neutral_mass_of_type(self, node_type=Unmodified):
        return self.find(node_type)._neutral_mass

    def add(self, node, recalculate=True):
        assert self.node_id != node.node_id
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
        return max(self.peaks, key=lambda x: x.intensity).intensity

    def total_intensity(self):
        return self._total_intensity_children() + self._total_intensity_members()

    def __eq__(self, other):
        return self.members == other.members

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
            chroma.insert(scan_id, peak, self.scan_id_to_rt(scan_id))
        else:
            chroma = Chromatogram(None)
            chroma.insert(scan_id, peak, self.scan_id_to_rt(scan_id))
            self.insert_chromatogram(chroma, index)
        self.count += 1

    def insert_chromatogram(self, chromatogram, index):
        if index[0] != 0:
            self.chromatograms.insert(index[0] + 1, chromatogram)
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
