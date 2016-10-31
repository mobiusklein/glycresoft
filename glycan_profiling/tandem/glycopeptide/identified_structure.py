from collections import defaultdict

from glycopeptidepy.structure.sequence import _n_glycosylation
from ..identified_structure import IdentifiedStructure, extract_identified_structures as _extract_identified_structures


class IdentifiedGlycopeptide(IdentifiedStructure):
    def __init__(self, structure, spectrum_matches, chromatogram, shared_with=None):
        super(IdentifiedGlycopeptide, self).__init__(
            structure, spectrum_matches, chromatogram, shared_with)
        self.q_value = min(s.q_value for s in self.spectrum_matches)
        self.protein_relation = self.structure.protein_relation

    @property
    def start_position(self):
        return self.protein_relation.start_position

    @property
    def end_position(self):
        return self.protein_relation.end_position

    def __repr__(self):
        return "IdentifiedGlycopeptide(%s, %0.3f, %0.3f, %0.3e)" % (
            self.structure, self.ms2_score, self.ms1_score, self.total_signal)

    def overlaps(self, other):
        return self.protein_relation.overlaps(other.protein_relation)

    def spans(self, position):
        return position in self.protein_relation


def indices_of_glycosylation(glycopeptide):
    i = 0
    out = []
    for res, mods in glycopeptide:
        if _n_glycosylation in mods:
            out.append(i)
        i += 1
    return out


def protein_indices_for_glycosites(glycopeptide):
    protein_relation = glycopeptide.protein_relation
    return [
        site + protein_relation.start_position for site in
        indices_of_glycosylation(glycopeptide)
    ]


class GlycopeptideIndex(object):
    def __init__(self, members=None):
        if members is None:
            members = []
        self.members = list(members)

    def append(self, member):
        self.members.append(member)

    def __iter__(self):
        return iter(self.members)

    def __getitem__(self, i):
        return self.members[i]

    def __setitem__(self, i, v):
        self.members[i] = v

    def __len__(self):
        return len(self.members)

    def with_glycan_composition(self, composition):
        return GlycopeptideIndex([member for member in self if member.glycan_composition == composition])

    def __repr__(self):
        return repr(self.members)

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.members)


class SiteMap(object):
    def __init__(self, store=None):
        if store is None:
            store = defaultdict(GlycopeptideIndex)
        self.store = store

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    @property
    def sites(self):
        return sorted(self.store.keys())

    def __iter__(self):
        return iter(self.store.items())

    def items(self):
        return self.store.items()

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return repr(self.store)

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.store)


class IdentifiedGlycoprotein(object):
    def __init__(self, protein, identified_glycopeptides):
        self.name = protein.name
        self.protein_sequence = protein.protein_sequence
        self.n_glycan_sequon_sites = protein.n_glycan_sequon_sites
        self.identified_glycopeptides = identified_glycopeptides
        self._site_map = SiteMap()
        self.microheterogeneity_map = defaultdict(lambda: defaultdict(float))
        self._map_glycopeptides_to_glycosites()

    @property
    def glycosylation_sites(self):
        return self.n_glycan_sequon_sites

    def _map_glycopeptides_to_glycosites(self):
        site_map = SiteMap()
        for gp in self.identified_glycopeptides:
            sites = set(protein_indices_for_glycosites(gp.structure))
            for site in self.n_glycan_sequon_sites:
                if site in sites:
                    site_map[site].append(gp)
                    sites.remove(site)
            # Deal with left-overs
            for site in sites:
                site_map[site].append(gp)
        self._site_map = site_map

        for site in self.n_glycan_sequon_sites:
            buckets = self._accumulate_glycan_composition_abundance_within_site(site)
            self.microheterogeneity_map[site] = buckets

    def _accumulate_glycan_composition_abundance_within_site(self, site):
        buckets = defaultdict(float)
        for gp in self._site_map[site]:
            buckets[str(gp.glycan_composition)] += gp.total_signal
        return buckets

    def __repr__(self):
        return "IdentifiedGlycoprotein(%s, %s)" % (
            self.name, "[%s]" % ', '.join(
                ["%d: %d" % (k, len(v)) for k, v in sorted(self._site_map.items(), key=lambda x: x[0])]))

    @property
    def site_map(self):
        return self._site_map

    @classmethod
    def aggregate(cls, glycopeptides, index=None):
        agg = defaultdict(list)
        for gp in glycopeptides:
            agg[gp.protein_relation.protein_id].append(gp)

        if index is None:
            return agg

        out = []
        for protein_id, group in agg.items():
            out.append(
                IdentifiedGlycoprotein(index[protein_id], group))
        return out


def extract_identified_structures(tandem_annotated_chromatograms, threshold_fn=lambda x: x.q_value < 0.05):
    return _extract_identified_structures(
        tandem_annotated_chromatograms, threshold_fn, result_type=IdentifiedGlycopeptide)
