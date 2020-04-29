import re
import warnings

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from glycopeptidepy.structure.parser import strip_modifications
from glycan_profiling.structure import LRUMapping

from glycan_profiling.tandem.spectrum_match import SpectrumMatchClassification as StructureClassification

from .glycosite_model import MINIMUM, to_decoy_glycan, parse_glycan_composition
from .glycoprotein_model import GlycoproteinSiteSpecificGlycomeModel


class GlycoproteomeModelBase(object):
    def score(self, glycopeptide, glycan_composition=None):
        raise NotImplementedError()

    @classmethod
    def bind_to_hypothesis(cls, session, site_models, hypothesis_id=1, fuzzy=True,
                           site_model_cls=GlycoproteinSiteSpecificGlycomeModel):
        inst = cls(
            site_model_cls.bind_to_hypothesis(
                session, site_models, hypothesis_id, fuzzy))
        return inst


class GlycoproteomeModel(GlycoproteomeModelBase):
    def __init__(self, glycoprotein_models):
        if isinstance(glycoprotein_models, Mapping):
            self.glycoprotein_models = dict(glycoprotein_models)
        else:
            self.glycoprotein_models = {
                ggm.id: ggm for ggm in glycoprotein_models
            }

    def relabel_proteins(self, name_to_id_map):
        remapped = {}
        for ggm in self.glycoprotein_models.values():
            try:
                new_id = name_to_id_map[ggm.name]
                remapped[new_id] = ggm
            except KeyError:
                warnings.warn(
                    "No mapping for %r, it will be omitted" % (ggm.name, ))
        self.glycoprotein_models = remapped

    def find_model(self, glycopeptide):
        if glycopeptide.protein_relation is None:
            return None
        protein_id = glycopeptide.protein_relation.protein_id
        glycoprotein_model = self.glycoprotein_models[protein_id]
        return glycoprotein_model

    def score(self, glycopeptide, glycan_composition=None):
        glycoprotein_model = self.find_model(glycopeptide)
        if glycoprotein_model is None:
            score = MINIMUM
        else:
            score = glycoprotein_model.score(glycopeptide, glycan_composition)
        return score


class SubstringGlycoproteomeModel(GlycoproteomeModelBase):
    def __init__(self, models, cache_size=2**15):
        self.models = models
        self.sequence_to_model = {
            str(model.protein): model for model in models.values()
        }
        self.peptide_to_protein = LRUMapping(cache_size)

    def get_models(self, glycopeptide):
        out = []
        seq = strip_modifications(glycopeptide)
        if seq in self.peptide_to_protein:
            return list(self.peptide_to_protein[seq])
        pattern = re.compile(seq)
        for case in self.sequence_to_model:
            if seq in case:
                bounds = pattern.finditer(case)
                for match in bounds:
                    protein_model = self.sequence_to_model[case]
                    site_models = protein_model.find_sites_in(
                        match.start(), match.end())
                    out.append(site_models)
        self.peptide_to_protein[seq] = tuple(out)
        return out

    def find_proteins(self, glycopeptide):
        out = []
        seq = strip_modifications(glycopeptide)
        for case in self.sequence_to_model:
            if seq in case:
                out.append(self.sequence_to_model[case])
        return out

    def score(self, glycopeptide, glycan_composition=None):
        if glycan_composition is None:
            glycan_composition = glycopeptide.glycan_composition
        models = self.get_models(glycopeptide)
        if len(models) == 0:
            return MINIMUM
        sites = models[0]
        if len(sites) == 0:
            return MINIMUM
        try:
            acc = []
            for site in sites:
                try:
                    rec = site.glycan_map[glycan_composition]
                    acc.append(rec.score)
                except KeyError:
                    pass
            return max(sum(acc) / len(acc), MINIMUM) if acc else MINIMUM
        except IndexError:
            return MINIMUM

    def __call__(self, glycopeptide):
        return self.get_models(glycopeptide)


class GlycoproteomePriorAnnotator(object):
    def __init__(self, target_model, decoy_model):
        self.target_model = target_model
        self.decoy_model = decoy_model

    def select_model(self, glycopeptide, structure_type):
        if structure_type & StructureClassification.decoy_peptide_target_glycan:
            return self.decoy_model
        else:
            return self.target_model

    def score_glycan(self, glycopeptide, structure_type, model):
        if structure_type & StructureClassification.target_peptide_decoy_glycan:
            gc = to_decoy_glycan(glycopeptide.glycan_composition)
        else:
            gc = glycopeptide.glycan_composition
        return model.score(glycopeptide, gc)

    def score(self, glycopeptide, structure_type):
        model = self.select_model(glycopeptide, structure_type)
        return self.score_glycan(glycopeptide, structure_type, model)
