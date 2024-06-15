import re
import os
from functools import partial

from typing import Optional, TYPE_CHECKING

from sqlalchemy import (
    Column, Integer, String, ForeignKey, PickleType,
    func)

from sqlalchemy.orm import relationship, backref, object_session
from sqlalchemy.ext.declarative import declared_attr

import dill

from glycresoft.serialize.base import (
    Base, HasUniqueName)

from glycresoft.serialize.spectrum import SampleRun

from glycresoft.serialize.hypothesis.generic import HasFiles
from glycresoft.serialize.param import HasParameters

from ms_deisotope.data_source import ProcessedRandomAccessScanSource
from ms_deisotope.output import ProcessedMSFileLoader

from glycresoft.structure.enums import AnalysisTypeEnum, GlycopeptideSearchStrategy

if TYPE_CHECKING:
    from glycresoft.composition_distribution_model.site_model.glycoproteome_model import GlycoproteomeModel


DillType = partial(PickleType, pickler=dill)


class _AnalysisParametersProps:
    def open_ms_file(self) -> Optional[ProcessedRandomAccessScanSource]:
        path = self.parameters.get("sample_path")
        if not path:
            return None
        if os.path.exists(path):
            return ProcessedMSFileLoader(path)
        return None

    def glycosite_model_path(self) -> Optional['GlycoproteomeModel']:
        from glycresoft.composition_distribution_model.site_model.glycoproteome_model import (
            GlycoproteomeModel, GlycosylationSiteModel)

        path = self.parameters.get('glycosylation_site_models_path')
        if os.path.exists(path):
            with open(path, 'rt', encoding='utf8') as fh:
                site_models = GlycosylationSiteModel.load(fh)
            target_model = GlycoproteomeModel.bind_to_hypothesis(
                object_session(self),
                site_models,
                hypothesis_id=self.hypothesis_id,
                fuzzy=True)
            return target_model
        return None

    @property
    def mass_shifts(self):
        return self.parameters.get('mass_shifts', [])

    @property
    def ms1_scoring_model(self):
        return self.parameters.get('scoring_model')

    @property
    def msn_scoring_model(self):
        return self.parameters.get('tandem_scoring_model')

    @property
    def retention_time_model(self):
        return self.parameters.get('retention_time_model')

    @property
    def fdr_estimator(self):
        return self.parameters.get('fdr_estimator')

    @property
    def search_strategy(self):
        strategy_str = self.parameters.get('search_strategy')
        strategy = GlycopeptideSearchStrategy[strategy_str]
        return strategy

    @property
    def is_multiscore(self):
        strategy = self.search_strategy
        return strategy == GlycopeptideSearchStrategy.multipart_target_decoy_competition


class Analysis(Base, HasUniqueName, HasFiles, HasParameters, _AnalysisParametersProps):
    __tablename__ = "Analysis"

    id: int = Column(Integer, primary_key=True)
    sample_run_id: int = Column(Integer, ForeignKey(SampleRun.id, ondelete="CASCADE"), index=True)
    sample_run: SampleRun = relationship(SampleRun)
    analysis_type: str = Column(String(128))
    status: str = Column(String(28))

    def __init__(self, **kwargs):
        self._init_parameters(kwargs)
        super(Analysis, self).__init__(**kwargs)

    def __repr__(self):
        sample_run = self.sample_run
        if sample_run:
            sample_run_name = sample_run.name
        else:
            sample_run_name = "<Detached From Sample>"
        return "Analysis(%s, %s)" % (self.name, sample_run_name)

    def _infer_hypothesis_id(self):
        try:
            hypothesis_id = self.parameters['hypothesis_id']
            return hypothesis_id
        except KeyError:
            session = object_session(self)
            if self.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
                from . import IdentifiedGlycopeptide, GlycopeptideHypothesis, Glycopeptide
                hypothesis_id = session.query(func.distinct(GlycopeptideHypothesis.id)).join(
                    Glycopeptide).join(
                    IdentifiedGlycopeptide,
                    Glycopeptide.id == IdentifiedGlycopeptide.structure_id).filter(
                    IdentifiedGlycopeptide.analysis_id == self.id).scalar()
                return hypothesis_id
            elif self.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
                from . import GlycanComposition, GlycanCompositionChromatogram, GlycanHypothesis
                hypothesis_id = session.query(func.distinct(GlycanHypothesis.id)).join(GlycanComposition).join(
                    GlycanCompositionChromatogram,
                    GlycanCompositionChromatogram.glycan_composition_id == GlycanComposition.id).filter(
                    GlycanCompositionChromatogram.analysis_id == self.id).scalar()
                return hypothesis_id
            else:
                raise ValueError(self.analysis_type)

    @property
    def hypothesis_id(self):
        return self._infer_hypothesis_id()

    def get_hypothesis(self):
        from . import GlycopeptideHypothesis, GlycanHypothesis
        hypothesis_id = self.hypothesis_id
        if hypothesis_id is None:
            raise ValueError(
                "Analysis does not have a Hypothesis (analysis id: %r, name: %r)" % (
                    self.id, self.name))
        session = object_session(self)
        if self.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
            return session.query(GlycopeptideHypothesis).get(hypothesis_id)
        elif self.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
            return session.query(GlycanHypothesis).get(hypothesis_id)

    hypothesis = property(get_hypothesis)

    def aggregate_identified_glycoproteins(self, glycopeptides=None):
        from glycresoft.tandem.glycopeptide.identified_structure import IdentifiedGlycoprotein
        from . import Protein
        if glycopeptides is None:
            glycopeptides = self.identified_glycopeptides.all()

        protein_index = {}
        session = object_session(self)
        proteins = session.query(Protein).all()
        for p in proteins:
            protein_index[p.id] = p

        glycoproteome = IdentifiedGlycoprotein.aggregate(glycopeptides, index=protein_index)
        return glycoproteome



class BoundToAnalysis(object):

    @declared_attr
    def analysis_id(self):
        return Column(Integer, ForeignKey(Analysis.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def analysis(self):
        return relationship(Analysis, backref=backref(self._collection_name(), lazy='dynamic'))

    @classmethod
    def _collection_name(cls):
        name = cls.__name__
        collection_name = re.sub(r"(.+?)([A-Z])", lambda match: match.group(1).lower() +
               "_" + match.group(2).lower(), name, 0) + 's'
        return collection_name
