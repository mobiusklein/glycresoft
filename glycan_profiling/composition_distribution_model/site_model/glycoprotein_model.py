import warnings
from collections import defaultdict, namedtuple


from glycopeptidepy.structure.parser import strip_modifications
from glycopeptidepy.utils.collectiontools import groupby

from glycan_profiling import serialize
from glycan_profiling.structure import PeptideProteinRelation
from glycan_profiling.database.builder.glycopeptide.proteomics.fasta import DeflineSuffix
from glycan_profiling.database.builder.glycopeptide.proteomics.sequence_tree import SuffixTree


from .glycosite_model import MINIMUM, GlycosylationSiteModel


ProteinStub = namedtuple("ProteinStub", ("name", ))


def _make_name_suffix_lookup(proteins):
    tree = SuffixTree()
    for prot in proteins:
        name = prot.name
        tree.add_ngram(DeflineSuffix(name, name))
    return tree


class GlycoproteinSiteSpecificGlycomeModel(object):
    def __init__(self, protein, glycosylation_sites=None):
        self.protein = protein
        self._glycosylation_sites = []
        self.glycosylation_sites = glycosylation_sites

    @property
    def glycosylation_sites(self):
        return self._glycosylation_sites

    @glycosylation_sites.setter
    def glycosylation_sites(self, glycosylation_sites):
        self._glycosylation_sites = sorted(
            glycosylation_sites or [], key=lambda x: x.position)

    def __getitem__(self, i):
        return self.glycosylation_sites[i]

    def __len__(self):
        return len(self.glycosylation_sites)

    @property
    def id(self):
        return self.protein.id

    @property
    def name(self):
        return self.protein.name

    def find_sites_in(self, start, end):
        spans = []
        for site in self.glycosylation_sites:
            if start <= site.position <= end:
                spans.append(site)
            elif end < site.position:
                break
        return spans

    def _guess_sites_from_sequence(self, sequence):
        prot_seq = str(self.protein)
        query_seq = strip_modifications(sequence)
        try:
            start = prot_seq.index(query_seq)
            end = start + len(query_seq)
            return PeptideProteinRelation(start, end, self.protein.id, self.protein.hypothesis_id)
        except ValueError:
            return None

    def score(self, glycopeptide, glycan_composition=None):
        if glycan_composition is None:
            glycan_composition = glycopeptide.glycan_composition
        pr = glycopeptide.protein_relation
        sites = self.find_sites_in(pr.start_position, pr.end_position)
        if len(sites) > 1:
            score_acc = 0.0
            warnings.warn(
                "Multiple glycosylation sites are not (yet) supported")
            for site in sites:
                try:
                    rec = site.glycan_map[glycan_composition]
                    score_acc += (rec.score)
                except KeyError:
                    score_acc += (MINIMUM)
            return score_acc / len(sites)
        try:
            site = sites[0]
            try:
                rec = site.glycan_map[glycan_composition]
            except KeyError:
                return MINIMUM
            return rec.score
        except IndexError:
            return MINIMUM

    @classmethod
    def bind_to_hypothesis(cls, session, site_models, hypothesis_id=1, fuzzy=True):
        by_protein_name = defaultdict(list)
        for site in site_models:
            by_protein_name[site.protein_name].append(site)
        protein_models = {}
        proteins = session.query(serialize.Protein).filter(
            serialize.Protein.hypothesis_id == hypothesis_id).all()
        protein_name_map = {prot.name: prot for prot in proteins}
        # if fuzzy:
        #     tree = SuffixTree()
        #     for prot in proteins:
        #         tree.add_ngram(DeflineSuffix(prot.name, prot.name))
        tree = None
        for protein_name, sites in by_protein_name.items():
            try:
                protein = protein_name_map[protein_name]
            except KeyError:
                if fuzzy:
                    if tree is None:
                        tree = _make_name_suffix_lookup(proteins)
                    labels = list(tree.subsequences_of(protein_name))
                    if not labels:
                        continue
                    protein = protein_name_map[labels[0].original]
                else:
                    continue
            model = cls(protein, sites)
            protein_models[model.id] = model
        return protein_models

    def __repr__(self):
        template = "{self.__class__.__name__}({self.name}, {self.glycosylation_sites})"
        return template.format(self=self)

    @classmethod
    def load(cls, fh, session=None, hypothesis_id=1, fuzzy=True):
        site_models = GlycosylationSiteModel.load(fh)
        if session is not None:
            return cls.bind_to_hypothesis(session, site_models, hypothesis_id, fuzzy)
        by_protein_name = groupby(site_models, lambda x: x.protein_name)
        result = []
        for name, models in by_protein_name.items():
            result.append(cls(ProteinStub(name), models))
        return result



class ReversedProteinSiteReflectionGlycoproteinSiteSpecificGlycomeModel(GlycoproteinSiteSpecificGlycomeModel):
    @property
    def glycosylation_sites(self):
        return self._glycosylation_sites

    @glycosylation_sites.setter
    def glycosylation_sites(self, glycosylation_sites):
        temp = []
        if isinstance(self.protein, ProteinStub):
            raise TypeError("Cannot create a reflected glycosite model with a ProteinStub")
        n = len(str(self.protein))
        for site in glycosylation_sites:
            site = site.copy()
            site.position = n - site.position - 1
            temp.append(site)
        self._glycosylation_sites = sorted(
            temp or [], key=lambda x: x.position)
