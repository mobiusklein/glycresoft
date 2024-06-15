from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from glycresoft.chromatogram_tree.mass_shift import MassShift, Unmodified
from glycresoft.structure.denovo import Path
from glycresoft.structure.scan import ScanStub
from glycresoft.structure.structure_loader import FragmentCachingGlycopeptide

from ms_deisotope import isotopic_shift
from ms_deisotope.data_source import ProcessedScan

from glycresoft.tandem.oxonium_ions import (
    OxoniumIndex,
    SignatureSpecification,
    gscore_scanner,
    single_signatures,
    compound_signatures
)

from glycresoft.database.mass_collection import NeutralMassDatabase, MassObject

from ..core_search import GlycanCombinationRecord, IndexedGlycanFilteringPeptideMassEstimator, CoreMotifFinder

from .search_space import PeptideMassFilterBase


@dataclass
class PeptideMassNode:
    peptide_mass: float
    mass_shift: float
    delta_mass: float
    scan_id: str
    glycan_score: float


@dataclass(frozen=True)
class PeptideEdge:
    structure: FragmentCachingGlycopeptide
    nodes: Tuple['SpectrumNode', 'SpectrumNode'] = field(hash=False)
    similarity: float = field(compare=False, hash=False, default=0.0)
    coverage: float = field(compare=False, hash=False, default=0.0)


@dataclass
class SpectrumNode:
    scan: Union[ProcessedScan, ScanStub]
    precursor_mass: float
    peptide_masses: NeutralMassDatabase[PeptideMassNode]
    signatures: Dict[SignatureSpecification, float]
    oxonium_score: float

    @property
    def scan_id(self):
        return self.scan.id

    def load(self):
        if isinstance(self.scan, ScanStub):
            self.scan = self.scan.convert()
        return self

    def unload(self):
        if isinstance(self.scan, ProcessedScan):
            self.scan = ScanStub(self.scan.precursor_information, self.scan.source)
        return self

    @property
    def deconvoluted_peak_set(self):
        return self.load().deconvoluted_peak_set

    def overlaps(self, other: 'SpectrumNode') -> List[Tuple[PeptideMassNode, PeptideMassNode]]:
        return intersecting_peptide_masses(self.peptide_masses, other.peptide_masses)


@dataclass
class IdentifiedSpectrumNode(SpectrumNode):
    structure: FragmentCachingGlycopeptide


# Glycan averagose mass
AVERAGE_MONOMER_MASS: float = 185.5677185226898


def score_path(path: Path, delta_mass: float) -> float:
    n = delta_mass / AVERAGE_MONOMER_MASS
    k = 2.0
    d = np.log(n) * n / k
    return np.log10([p.intensity for p in path.peaks]).sum() / d


def merge_peptide_nodes_database_denovo(database_nodes: List[PeptideMassNode],
                                        denovo_nodes: List[PeptideMassNode], error_tolerance: float=5e-6) -> List[PeptideMassNode]:
    accumulator = []

    db_i = 0
    denovo_i = 0
    db_n = len(database_nodes)
    denovo_n = len(denovo_nodes)

    while db_i < db_n and denovo_i < denovo_n:
        db_node = database_nodes[db_i]
        denovo_node = denovo_nodes[denovo_i]
        if abs(db_node.peptide_mass - denovo_node.peptide_mass) / denovo_node.peptide_mass <= error_tolerance:
            accumulator.append(db_node)
            db_i += 1
            denovo_i += 1
        elif db_node.peptide_mass < denovo_node.peptide_mass:
            accumulator.append(db_node)
            db_i += 1
        else:
            accumulator.append(denovo_node)
            denovo_i += 1

    accumulator.extend(denovo_nodes[denovo_i:])
    accumulator.extend(database_nodes[db_i:])
    return accumulator


def intersecting_peptide_masses(query_nodes: Sequence[PeptideMassNode],
                                reference_nodes: Sequence[PeptideMassNode],
                                error_tolerance: float=5e-6) -> List[Tuple[PeptideMassNode, PeptideMassNode]]:
    accumulator = []

    query_i = 0
    reference_i = 0
    query_n = len(query_nodes)
    reference_n = len(reference_nodes)

    checkpoint = None

    while query_i < query_n and reference_i < reference_n:
        query_node = query_nodes[query_i]
        reference_node = reference_nodes[reference_i]
        if abs(query_node.peptide_mass - reference_node.peptide_mass) / reference_node.peptide_mass <= error_tolerance:
            accumulator.append((query_node, reference_node))
            if checkpoint is None:
                checkpoint = reference_i
            reference_i += 1
        elif query_node.peptide_mass < reference_node.peptide_mass:
            if checkpoint is not None and query_i < (query_n - 1) and abs(query_node.peptide_mass - query_nodes[query_i + 1].peptide_mass) / query_node.peptide_mass < error_tolerance:
                reference_i = checkpoint
                checkpoint = None
            query_i += 1
        else:
            reference_i += 1
    return accumulator


class PeptideGraphBuilder(PeptideMassFilterBase):
    motif_finder: CoreMotifFinder
    signatures: Dict[SignatureSpecification, float]

    def __init__(self, glycan_compositions, product_error_tolerance=0.00002, glycan_score_threshold=0.1,
                 min_fragments=2, peptide_masses_per_scan=100, probing_range_for_missing_precursors=3,
                 trust_precursor_fits=True, fragment_weight=0.56, core_weight=1.42, oxonium_ion_index=None,
                 signature_ion_index=None, oxonium_ion_threshold: float = 0.05, signatures: List[SignatureSpecification]=None):
        super().__init__(glycan_compositions, product_error_tolerance, glycan_score_threshold, min_fragments,
                         peptide_masses_per_scan, probing_range_for_missing_precursors, trust_precursor_fits,
                         fragment_weight, core_weight, oxonium_ion_index, signature_ion_index, oxonium_ion_threshold)
        if signatures is None:
            signatures = signature_ion_index.signatures if signature_ion_index is not None else list(
                single_signatures.keys())
        self.motif_finder = CoreMotifFinder(self.monosaccharides, self.product_error_tolerance)
        self.signatures = signatures

    def collect_denovo_peptide_nodes(self, scan: ProcessedScan, mass_shifts: Optional[List[MassShift]]=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        if not scan.precursor_information.defaulted and self.trust_precursor_fits:
            probe = 0
        else:
            probe = self.probing_range_for_missing_precursors
        precursor_mass = scan.precursor_information.neutral_mass
        nodes = []
        for i in range(probe + 1):
            neutron_shift = self.neutron_shift * i
            for mass_shift in mass_shifts:
                mass_shift_mass = mass_shift.mass
                intact_mass = precursor_mass - mass_shift_mass - neutron_shift
                for (peptide_mass, paths) in self.motif_finder.estimate_peptide_mass(
                            scan,
                            topn=self.peptide_masses_per_scan,
                            mass_shift=mass_shift,
                            simplify=False,
                            query_mass=precursor_mass - neutron_shift,
                        ):

                    if len(paths[0].peaks) < (self.min_fragments + 1):
                        continue

                    delta_mass = intact_mass - peptide_mass
                    nodes.append(
                        PeptideMassNode(peptide_mass, mass_shift, delta_mass,
                                        scan.id, score_path(paths[0], delta_mass))
                    )
        nodes.sort(key=lambda x: x.peptide_mass)
        return nodes

    def collect_database_peptide_nodes(self, scan: ProcessedScan, mass_shifts: Optional[List[MassShift]]=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        if not scan.precursor_information.defaulted and self.trust_precursor_fits:
            probe = 0
        else:
            probe = self.probing_range_for_missing_precursors
        precursor_mass = scan.precursor_information.neutral_mass
        nodes = []
        for i in range(probe + 1):
            neutron_shift = self.neutron_shift * i
            for mass_shift in mass_shifts:
                mass_shift_mass = mass_shift.mass
                intact_mass = precursor_mass - mass_shift_mass - neutron_shift
                for peptide_mass_pred in self.peptide_mass_predictor.estimate_peptide_mass(
                            scan,
                            topn=self.peptide_masses_per_scan,
                            mass_shift=mass_shift,
                            threshold=self.glycan_score_threshold,
                            min_fragments=self.min_fragments,
                            simplify=False,
                            query_mass=precursor_mass - neutron_shift,
                        ):
                    peptide_mass = peptide_mass_pred.peptide_mass
                    nodes.append(
                        PeptideMassNode(peptide_mass, mass_shift, intact_mass - peptide_mass, scan.id, peptide_mass_pred.score)
                    )
        nodes.sort(key=lambda x: x.peptide_mass)
        return nodes

    def handle_scan(self, scan: ProcessedScan, mass_shifts: Optional[List[MassShift]] = None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        nodes = self.collect_database_peptide_nodes(scan, mass_shifts)
        nodes.sort(key=lambda x: x.peptide_mass)
        denovo_nodes = self.collect_denovo_peptide_nodes(scan, mass_shifts)
        denovo_nodes.sort(key=lambda x: x.peptide_mass)
        nodes = merge_peptide_nodes_database_denovo(
            nodes, denovo_nodes, self.product_error_tolerance)
        signature_intensities = {}
        if len(scan.deconvoluted_peak_set) > 0:
            base_peak_intensity: float = scan.base_peak.deconvoluted().intensity
        else:
            base_peak_intensity: float = 1.0
        for sig in self.signatures:
            peak = sig.peak_of(scan.deconvoluted_peak_set, self.product_error_tolerance)
            if peak is not None:
                signature_intensities[sig] = peak.intensity / base_peak_intensity
            else:
                signature_intensities[sig] = 0.0
        oxonium_score = gscore_scanner(scan.deconvoluted_peak_set)
        return SpectrumNode(
            scan,
            scan.precursor_information.neutral_mass,
            NeutralMassDatabase(nodes, lambda x: x.peptide_mass),
            signature_intensities,
            oxonium_score
        )


_PeptideMassNode = PeptideMassNode
_intersecting_peptide_masses = intersecting_peptide_masses

from glycresoft._c.tandem.peptide_graph import PeptideMassNode, intersecting_peptide_masses
