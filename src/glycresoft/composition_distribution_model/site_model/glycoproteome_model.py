import re
import io
import warnings

from typing import Dict, List, Optional, Mapping

from glycopeptidepy.structure.parser import strip_modifications
from glypy.structure.glycan_composition import HashableGlycanComposition

from glycresoft.structure import LRUMapping

from glycresoft.structure.structure_loader import FragmentCachingGlycopeptide
from glycresoft.tandem.spectrum_match import SpectrumMatchClassification as StructureClassification

from .glycosite_model import MINIMUM, GlycosylationSiteModel
from .glycoprotein_model import (
    GlycoproteinSiteSpecificGlycomeModel,
    ReversedProteinSiteReflectionGlycoproteinSiteSpecificGlycomeModel
)


class GlycoproteomeModelBase(object):
    __slots__ = ()
    def score(self, glycopeptide: FragmentCachingGlycopeptide,
              glycan_composition: Optional[HashableGlycanComposition]=None) -> float:
        raise NotImplementedError()

    @classmethod
    def bind_to_hypothesis(cls, session, site_models, hypothesis_id=1, fuzzy=True,
                           site_model_cls=GlycoproteinSiteSpecificGlycomeModel) -> 'GlycoproteomeModelBase':
        inst = cls(
            site_model_cls.bind_to_hypothesis(
                session, site_models, hypothesis_id, fuzzy))
        return inst


class GlycoproteomeModel(GlycoproteomeModelBase):
    glycoprotein_models: Dict[int, GlycoproteinSiteSpecificGlycomeModel]

    def __init__(self, glycoprotein_models):
        if isinstance(glycoprotein_models, Mapping):
            self.glycoprotein_models = dict(glycoprotein_models)
        else:
            self.glycoprotein_models = {
                ggm.id: ggm for ggm in glycoprotein_models
            }

    def relabel_proteins(self, name_to_id_map: Dict[str, int]):
        remapped = {}
        for ggm in self.glycoprotein_models.values():
            try:
                new_id = name_to_id_map[ggm.name]
                remapped[new_id] = ggm
            except KeyError:
                warnings.warn(
                    "No mapping for %r, it will be omitted" % (ggm.name, ))
        self.glycoprotein_models = remapped

    def stub_proteins(self):
        for model in self.glycoprotein_models.values():
            model.stub_protein()

    def find_model(self, glycopeptide: FragmentCachingGlycopeptide):
        if glycopeptide.protein_relation is None:
            return None
        protein_id = glycopeptide.protein_relation.protein_id
        glycoprotein_model = self.glycoprotein_models.get(protein_id)
        return glycoprotein_model

    def score(self, glycopeptide: FragmentCachingGlycopeptide,
              glycan_composition: Optional[HashableGlycanComposition]=None) -> float:
        glycoprotein_model = self.find_model(glycopeptide)
        if glycoprotein_model is None:
            score = MINIMUM
        else:
            score = glycoprotein_model.score(glycopeptide, glycan_composition)
        return score


class SubstringGlycoproteomeModel(GlycoproteomeModelBase):
    sequence_to_model: Dict[str, GlycoproteinSiteSpecificGlycomeModel]

    def __init__(self, models, cache_size=2**15):
        self.models = models
        self.sequence_to_model = {
            str(model.protein): model for model in models.values()
        }
        self.peptide_to_protein = LRUMapping(cache_size)

    def get_models(self, glycopeptide: FragmentCachingGlycopeptide) -> List[GlycoproteinSiteSpecificGlycomeModel]:
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

    def find_proteins(self, glycopeptide: FragmentCachingGlycopeptide) -> List[GlycoproteinSiteSpecificGlycomeModel]:
        out = []
        seq = strip_modifications(glycopeptide)
        for case in self.sequence_to_model:
            if seq in case:
                out.append(self.sequence_to_model[case])
        return out

    def score(self, glycopeptide: FragmentCachingGlycopeptide,
              glycan_composition: Optional[HashableGlycanComposition]=None) -> float:
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
    target_model: GlycoproteomeModel
    decoy_model: GlycoproteomeModel

    @classmethod
    def load(cls, target_session, decoy_session, fh: io.TextIOBase, hypothesis_id=1, fuzzy=True):
        site_models = GlycosylationSiteModel.load(fh)
        target_model = GlycoproteomeModel.bind_to_hypothesis(
            target_session,
            site_models,
            hypothesis_id=hypothesis_id,
            fuzzy=fuzzy)
        decoy_model = GlycoproteomeModel.bind_to_hypothesis(
            decoy_session,
            site_models,
            hypothesis_id=hypothesis_id,
            fuzzy=fuzzy,
            site_model_cls=ReversedProteinSiteReflectionGlycoproteinSiteSpecificGlycomeModel)
        target_model.stub_proteins()
        decoy_model.stub_proteins()
        return cls(target_model, decoy_model)

    def __init__(self, target_model, decoy_model):
        self.target_model = target_model
        self.decoy_model = decoy_model

    def select_model(self, glycopeptide: FragmentCachingGlycopeptide,
                     structure_type: StructureClassification) -> GlycoproteomeModel:
        if structure_type & StructureClassification.decoy_peptide_target_glycan:
            return self.decoy_model
        else:
            return self.target_model

    def score_glycan(self, glycopeptide: FragmentCachingGlycopeptide,
                     structure_type: StructureClassification,
                     model: GlycoproteomeModel) -> float:
        # Treat decoy glycans identical
        gc = glycopeptide.glycan_composition
        return model.score(glycopeptide, gc)

    def score(self, glycopeptide: FragmentCachingGlycopeptide, structure_type: StructureClassification) -> float:
        model = self.select_model(glycopeptide, structure_type)
        return self.score_glycan(glycopeptide, structure_type, model)
