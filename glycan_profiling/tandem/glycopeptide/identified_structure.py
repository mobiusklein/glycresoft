from collections import defaultdict

from glycopeptidepy.structure.sequence import (
    _n_glycosylation,
    _o_glycosylation,
    _gag_linker_glycosylation)

from glycopeptidepy.structure.glycan import (
    GlycosylationType)

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


class AmbiguousGlycopeptideGroup(object):
    def __init__(self, members=None):
        if members is None:
            members = []
        self.members = list(members)

    def ambiguous_with(self, glycopeptide):
        for member in self:
            if not member.is_distinct(glycopeptide):
                return True
        return False

    def add(self, glycopeptide):
        self.members.append(glycopeptide)

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]

    def __repr__(self):
        return "{s.__class__.__name__}({size})".format(s=self, size=len(self))

    def __contains__(self, other):
        return other in self.members

    @classmethod
    def aggregate(cls, glycopeptides):
        groups = []
        no_chromatogram = []
        for glycopeptide in glycopeptides:
            if glycopeptide.chromatogram is None:
                no_chromatogram.append(cls([glycopeptide]))
            else:
                for group in groups:
                    if group.ambiguous_with(glycopeptide):
                        group.add(glycopeptide)
                        break
                else:
                    groups.append(cls([glycopeptide]))
        return groups + no_chromatogram


core_type_map = {
    GlycosylationType.n_linked: _n_glycosylation,
    GlycosylationType.o_linked: _o_glycosylation,
    GlycosylationType.glycosaminoglycan: _gag_linker_glycosylation
}


def indices_of_glycosylation(glycopeptide, glycosylation_type=GlycosylationType.n_linked):
    i = 0
    out = []

    core_type = core_type_map[glycosylation_type]

    for res, mods in glycopeptide:
        if core_type in mods:
            out.append(i)
        i += 1
    return out


def protein_indices_for_glycosites(glycopeptide, glycosylation_type=GlycosylationType.n_linked):
    protein_relation = glycopeptide.protein_relation
    return [
        site + protein_relation.start_position for site in
        indices_of_glycosylation(glycopeptide, glycosylation_type)
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


class HeterogeneityMap(object):
    def __init__(self):
        self.store = defaultdict(lambda: defaultdict(float))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = defaultdict(float, value)

    def __iter__(self):
        return iter(self.store)

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def get(self, key, default=None):
        return self.store.get(key, default)

    def pop(self, key):
        return self.store.pop(key)


class IdentifiedGlycoprotein(object):
    glycosylation_types = [
        GlycosylationType.n_linked,
        GlycosylationType.o_linked,
        GlycosylationType.glycosaminoglycan
    ]

    def __init__(self, protein, identified_glycopeptides):
        self.name = protein.name
        self.protein_sequence = protein.protein_sequence
        self.n_glycan_sequon_sites = protein.n_glycan_sequon_sites
        self.o_glycan_sequon_sites = protein.o_glycan_sequon_sites
        self.glycosaminoglycan_sequon_sites = protein.glycosaminoglycan_sequon_sites
        self.identified_glycopeptides = identified_glycopeptides
        self.ambiguous_groups = [
            a for a in AmbiguousGlycopeptideGroup.aggregate(identified_glycopeptides)
            if len(a) > 1
        ]
        self._site_map = defaultdict(lambda: SiteMap())
        self.microheterogeneity_map = HeterogeneityMap()
        for glycotype in self.glycosylation_types:
            self._map_glycopeptides_to_glycosites(glycotype)

    @property
    def glycosylation_sites(self):
        return self.n_glycan_sequon_sites

    def glycosylation_sites_for(self, glycosylation_type, occupied=False):
        sites = self._get_site_list_for(glycosylation_type)
        if not occupied:
            return sites
        out = []
        for site in sites:
            if site not in self.site_map[glycosylation_type]:
                continue
            out.append(site)
        return out

    def _get_site_list_for(self, glycosylation_type):
        if glycosylation_type == GlycosylationType.n_linked:
            return self.n_glycan_sequon_sites
        elif glycosylation_type == GlycosylationType.o_linked:
            return self.o_glycan_sequon_sites
        elif glycosylation_type == GlycosylationType.glycosaminoglycan:
            return self.glycosaminoglycan_sequon_sites
        else:
            raise KeyError("Glycosylation type %r not known" % (glycosylation_type,))

    def _aggregate_buckets(self, site_map, glycosylation_type=GlycosylationType.n_linked):
        microheterogeneity_map = defaultdict(lambda: defaultdict(float))
        for site in self._get_site_list_for(glycosylation_type):
            buckets = self._accumulate_glycan_composition_abundance_within_site(site, site_map)
            microheterogeneity_map[site] = buckets
        return microheterogeneity_map

    def _map_glycopeptides_to_glycosites(self, glycosylation_type=GlycosylationType.n_linked):
        site_map = SiteMap()
        for gp in self.identified_glycopeptides:
            sites = set(protein_indices_for_glycosites(gp.structure, glycosylation_type))
            for site in self._get_site_list_for(glycosylation_type):
                if site in sites:
                    site_map[site].append(gp)
                    sites.remove(site)
            # Deal with left-overs
            for site in sites:
                site_map[site].append(gp)

        self._site_map[glycosylation_type] = site_map
        self.microheterogeneity_map[glycosylation_type] = self._aggregate_buckets(site_map, glycosylation_type)

    def _accumulate_glycan_composition_abundance_within_site(self, site, site_map):
        buckets = defaultdict(float)
        if site not in site_map:
            return buckets
        for gp in site_map[site]:
            buckets[str(gp.glycan_composition)] += gp.total_signal
        return buckets

    def __repr__(self):
        site_list = defaultdict(lambda: defaultdict(int))
        for key, site_map in self._site_map.items():
            for site, members in site_map.items():
                site_list[site][key] = len(members)
        prefix_map = {}
        for key in site_list:
            prefix_map[key] = self.protein_sequence[key]

        string_form = [":".join(map(lambda x: x.name, self._site_map.keys()))]
        for site, observed in sorted(site_list.items()):
            prefix = prefix_map[site]
            components = [observed[k] for k in self._site_map]
            if sum(components) == 0:
                continue
            string_form.append("%d%s=(%s)" % (site, prefix, ', '.join(map(str, components))))

        return "IdentifiedGlycoprotein(%s, %s)" % (
            self.name, "[%s]" % ', '.join(string_form))

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
