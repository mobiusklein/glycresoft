from collections import defaultdict

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


def indices_of_glycosylation(glycopeptide):
    i = 0
    out = []
    for res, mods in glycopeptide:
        if "N-Glycosylation" in mods:
            out.append(i)
        i += 1
    return out


def protein_indices_for_glycosites(glycopeptide):
    protein_relation = glycopeptide.protein_relation
    return [
        site + protein_relation.start_position for site in
        indices_of_glycosylation(glycopeptide)
    ]


class IdentifiedGlycoprotein(object):
    def __init__(self, protein, identified_glycopeptides):
        self.protein = protein
        self.n_glycan_sequon_sites = protein.n_glycan_sequon_sites
        self.identified_glycopeptides = identified_glycopeptides
        self._site_map = defaultdict(list)
        self.microheterogeneity_map = defaultdict(lambda: defaultdict(float))

        self._map_glycopeptides_to_glycosites()

    def _map_glycopeptides_to_glycosites(self):
        site_map = defaultdict(list)
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
            self.protein.name, "[%s]" % ', '.join(
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


def extract_identified_structures(tandem_annotated_chromatograms):
    return _extract_identified_structures(
        tandem_annotated_chromatograms, result_type=IdentifiedGlycopeptide)
