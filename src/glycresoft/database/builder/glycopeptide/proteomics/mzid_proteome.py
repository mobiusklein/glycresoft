import os
import re
import operator
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Set, Tuple

from six import string_types as basestring

from glycopeptidepy.io import fasta as glycopeptidepy_fasta
from glycopeptidepy.structure import sequence, modification, residue
from glycopeptidepy.enzyme import Protease, enzyme_rules
from glypy.composition import formula

from glycresoft.serialize import (
    Peptide, Protein, DatabaseBoundOperation)

from glycresoft.task import TaskBase

from ..common import ProteinReversingMixin

from .mzid_parser import Parser
from .peptide_permutation import (
    ProteinDigestor, n_glycan_sequon_sites,
    o_glycan_sequon_sites, gag_sequon_sites, UniprotProteinAnnotator)
from .remove_duplicate_peptides import DeduplicatePeptides
from .share_peptides import PeptideSharer
from .fasta import FastaProteinSequenceResolver

logger = logging.getLogger("mzid")


PeptideSequence = sequence.PeptideSequence
Residue = residue.Residue
Modification = modification.Modification
ModificationNameResolutionError = modification.ModificationNameResolutionError
AnonymousModificationRule = modification.AnonymousModificationRule

GT = "greater"
LT = "lesser"

PROTEOMICS_SCORE = {
    "PEAKS:peptideScore": GT,
    "mascot:score": GT,
    "PEAKS:proteinScore": GT,
    "MS-GF:EValue": LT,
    "Byonic:Score": GT,
    "percolator:Q value": LT,
    r"X\!Tandem:expect": GT,
    "Morpheus:Morpheus score": GT,
    "MetaMorpheus:score": GT,
}


def score_comparator(score_type):
    try:
        preference = PROTEOMICS_SCORE[score_type]
        if preference == LT:
            return operator.lt
        else:
            return operator.gt
    except KeyError:
        raise KeyError("Don't know how to compare score of type %r" % score_type)


WHITELIST_GLYCOSITE_PTMS = [Modification("Deamidation"), Modification("HexNAc")]


class allset(object):

    def __contains__(self, x):
        return True


class ParameterizedProtease(Protease):
    def __init__(self, name, used_missed_cleavages=1, cleavage_start=None, cleavage_end=None):
        super(ParameterizedProtease, self).__init__(name, cleavage_start, cleavage_end)
        self.used_missed_cleavages = used_missed_cleavages


def resolve_database_url(url):
    if url.startswith("file://"):
        path = url.replace("file://", "")
        while path.startswith("/") and len(path) > 0:
            path = path[1:]
        if os.path.exists(path):
            if os.path.isfile(path):
                return path
            else:
                raise IOError("File URI %r points to a directory" % url)
        else:
            raise IOError("File URI %r does not exist on local system" % url)
    elif url.startswith("http"):
        return url
    elif os.path.exists(url):
        if os.path.isfile(url):
            return url
        else:
            raise IOError("File path %r points to a directory" % url)
    else:
        raise IOError("Cannot resolve URL %r" % url)


def protein_names(mzid_path, pattern=r'.*'):
    pattern = re.compile(pattern)
    parser = Parser(mzid_path, retrieve_refs=False,
                    iterative=True, build_id_cache=False, use_index=False)
    for protein in parser.iterfind(
            "DBSequence", retrieve_refs=False, recursive=False, iterative=True):
        name = protein['accession']
        if pattern.match(name):
            yield name


def remove_peptide_sequence_alterations(base_sequence, insert_sites, delete_sites):
    """
    Remove all the sequence insertions and deletions in order to reconstruct the
    original peptide sequence.

    Parameters
    ----------
    base_sequence : str
        The peptide sequence string which contains a combination
        of insertion and deletions.
    insert_sites : list
        A list of (position, None) pairs indicating the position of
        an amino acid insertion to be removed.
    delete_sites : list
        A list of (position, residue) pairs indicating the position
        and identity of amino acids that were deleted and need to be
        re-inserted.

    Returns
    -------
    str
    """
    sequence_copy = list(base_sequence)

    alteration_sites = insert_sites + delete_sites
    alteration_sites.sort()
    shift = 0
    for position, residue_ in alteration_sites:
        if residue_ is None:
            sequence_copy.pop(position - shift)
            shift += 1
        else:
            sequence_copy.insert(position - shift + 1, residue_)
            shift -= 1
    sequence_copy = ''.join(sequence_copy)
    return sequence_copy


class PeptideGroup(object):
    """Maps Protein ID to :class:`Peptide`, keeping track of the top scoring example
    for each :class:`Peptide` instance observed from the set of target Proteins

    Attributes
    ----------
    has_target_match: bool
        Description
    members: dict
        Mapping from Protein ID to :class:`Peptide`
    scores: defaultdict(list)
        Mapping from Protein ID to score
    """

    members: Dict
    scores: DefaultDict[str, List[float]]
    has_target_match: bool

    def __init__(self):
        self.members = dict()
        self.scores = defaultdict(list)
        self.has_target_match = False
        self._first = None
        self._last = None

    def __getitem__(self, key):
        return self.members[key]

    def __setitem__(self, key, value):
        self.members[key] = value
        if self._first is None:
            self._first = value
        self._last = value

    def clear(self):
        self.members.clear()
        self.has_target_match = False
        self._first = None
        self._last = None

    def _update_has_target_match(self, protein_set: Set[str]) -> bool:
        for key in self.members:
            if key in protein_set:
                return True
        return False

    def update_state(self, protein_set: Set[str]):
        had = self.has_target_match
        if not had:
            has = self.has_target_match = self._update_has_target_match(protein_set)
        else:
            has = had
        if not had and has:
            for key in list(self.members):
                if key not in protein_set:
                    self.members.pop(key)
                    self.scores.pop(key)
        else:
            if self._last.protein_id not in protein_set:
                try:
                    self.members.pop(self._last.protein_id)
                    self.scores.pop(self._last.protein_id)
                except KeyError:
                    pass

    def keys(self):
        return self.members.keys()

    def values(self):
        return self.members.values()

    def items(self):
        return self.members.items()

    def bind_scores(self):
        acc = []
        for key, peptide in self.members.items():
            peptide.scores = self.scores[key]
            acc.extend(self.scores[key])
        return acc

    def first(self):
        return self._first

    def __nonzero__(self):
        if self.members:
            return True
        else:
            return False

    def __bool__(self):
        if self.members:
            return True
        else:
            return False

    def __len__(self):
        return len(self.members)

    def add(self, peptide):
        comparator = score_comparator(peptide.peptide_score_type)
        if self:
            first = self.first()
            if comparator(peptide.peptide_score, first.peptide_score):
                self.clear()
                self[peptide.protein_id] = peptide
            elif first.peptide_score == peptide.peptide_score:
                self[peptide.protein_id] = peptide
        else:
            self[peptide.protein_id] = peptide
        self.scores[peptide.protein_id].append(peptide.peptide_score)


class PeptideCollection(object):
    store: DefaultDict[str, PeptideGroup]
    protein_set: Set[str]
    scores: List[float]

    def __init__(self, protein_set):
        self.store = defaultdict(PeptideGroup)
        self.protein_set = protein_set
        self.scores = []

    def bind_scores(self):
        acc = []
        for key, value in self.store.items():
            acc.extend(value.bind_scores())
        self.scores = acc
        return acc

    def add(self, peptide):
        group = self.store[peptide.modified_peptide_sequence]
        group.add(peptide)
        group.update_state(self.protein_set)
        self.store[peptide.modified_peptide_sequence] = group

    def items(self):
        for key, mapping in self.store.items():
            yield key, mapping.values()

    def __len__(self):
        return sum(map(len, self.store.values()))


class MzIdentMLPeptide(object):
    peptide_dict: Dict[str, Any]
    insert_sites: List[Tuple[int, str]]
    deleteion_sites: List[Tuple[int, str]]

    modification_counter: int
    missed_cleavages: Optional[int]

    base_sequence: str
    peptide_sequence: PeptideSequence
    glycosite_candidates: List[int]

    modification_translation_table: Dict[str, Modification]
    enzyme: Optional[Protease]
    mzid_id: Optional[str]

    def __init__(self, peptide_dict, enzyme=None, constant_modifications=None,
                 modification_translation_table=None, process=True):
        if modification_translation_table is None:
            modification_translation_table = dict()
        if constant_modifications is None:
            constant_modifications = list()

        self.peptide_dict = peptide_dict

        self.insert_sites = []
        self.deleteion_sites = []
        self.modification_counter = 0
        self.missed_cleavages = 0

        self.base_sequence = peptide_dict["PeptideSequence"]
        self.peptide_sequence = PeptideSequence(peptide_dict["PeptideSequence"])

        self.glycosite_candidates = sequence.find_n_glycosylation_sequons(
            self.peptide_sequence, WHITELIST_GLYCOSITE_PTMS)

        self.constant_modifications = constant_modifications
        self.modification_translation_table = modification_translation_table
        self.enzyme = enzyme
        self.mzid_id = peptide_dict.get('id')

        if process:
            self.process()

    def handle_missed_cleavages(self):
        if self.enzyme is not None:
            self.missed_cleavages = self.enzyme.missed_cleavages(self.base_sequence)
        else:
            self.missed_cleavages = None

    def handle_substitutions(self):
        if "SubstitutionModification" in self.peptide_dict:
            subs = self.peptide_dict["SubstitutionModification"]
            for sub in subs:
                pos = sub['location'] - 1
                replace = Residue(sub["replacementResidue"])
                self.peptide_sequence.substitute(pos, replace)
                self.modification_counter += 1

    def add_modification(self, modification: Modification, position: int):
        if position == -1:
            targets = modification.rule.n_term_targets
            for t in targets:
                if t.position_modifier is not None and t.amino_acid_targets is None:
                    break
            else:
                position += 1
        if position == len(self.peptide_sequence):
            targets = modification.rule.c_term_targets
            for t in targets:
                if t.position_modifier is not None and t.amino_acid_targets is None:
                    break
            else:
                position -= 1
        if position == -1:
            self.peptide_sequence.n_term = modification
        elif position == len(self.peptide_sequence):
            self.peptide_sequence.c_term = modification
        else:
            self.peptide_sequence.add_modification(position, modification)

    def add_insertion(self, site: int, symbol=None):
        self.insert_sites.append((site, symbol))

    def add_deletion(self, site: int, symbol):
        self.deleteion_sites.append((site, symbol))

    def handle_modifications(self):
        if "Modification" in self.peptide_dict:
            mods: List[Dict[str, Any]] = self.peptide_dict["Modification"]
            for mod in mods:
                pos: int = mod["location"] - 1
                accession = None
                try:
                    if "unknown modification" in mod:
                        try:
                            _name = mod['unknown modification']
                            if _name in self.modification_translation_table:
                                modification = self.modification_translation_table[_name]()
                            else:
                                modification = Modification(str(_name))
                        except ModificationNameResolutionError:
                            raise KeyError("Cannot find key in %r" % (mod,))
                    else:
                        try:
                            _name = mod["name"]
                            accession = getattr(_name, "accession", None)
                            _name = str(_name)
                            if accession is not None:
                                accession = str(accession)
                                try:
                                    modification = Modification(accession)
                                except ModificationNameResolutionError:
                                    modification = Modification(_name)
                            else:
                                modification = Modification(_name)

                        except (KeyError, ModificationNameResolutionError) as e:
                            raise KeyError("Cannot find key %s in %r" % (e, mod))

                    try:
                        rule_text = "%s (%s)" % (_name, mod["residues"][0])
                        if (rule_text not in self.constant_modifications) and not (
                                pos in self.glycosite_candidates and modification in WHITELIST_GLYCOSITE_PTMS):
                            self.modification_counter += 1
                    except KeyError:
                        self.modification_counter += 1

                    self.add_modification(modification, pos)
                except KeyError:
                    if "unknown modification" in mod:
                        mod_description = mod["unknown modification"]
                        insertion = re.search(r"(\S{3})\sinsertion", mod_description)
                        deletion = re.search(r"(\S{3})\sdeletion", mod_description)
                        self.modification_counter += 1
                        if insertion:
                            self.add_insertion(mod['location'] - 1, None)
                        elif deletion:
                            sym = Residue(deletion.groups()[0]).symbol
                            self.add_deletion(mod['location'] - 1, sym)
                        elif 'monoisotopicMassDelta' in mod:
                            mass = float(mod['monoisotopicMassDelta'])
                            modification = AnonymousModificationRule(str(_name), mass)()
                            self.add_modification(modification, pos)
                        else:
                            raise
                    else:
                        raise

    def process(self):
        self.handle_substitutions()
        self.handle_modifications()
        self.handle_missed_cleavages()

    def original_sequence(self):
        sequence_copy = remove_peptide_sequence_alterations(
            self.base_sequence, self.insert_sites, self.deleteion_sites)
        return sequence_copy

    def __repr__(self):
        return f"{self.__class__.__name__}({self.peptide_sequence}, {self.glycosite_candidates})"


class _EvidenceMixin:
    @property
    def score_type(self) -> Tuple[Optional[float], Optional[str]]:
        score = score_type = None
        for k, v in self.peptide_dict.items():
            if k in PROTEOMICS_SCORE:
                score_type = k
                score = v
                break
        return score, score_type

    @property
    def evidence_list(self) -> List[Dict[str, Any]]:
        evidence_list = self.peptide_dict["PeptideEvidenceRef"]
        # Flatten the evidence list if it has extra nesting because of alternative
        # mzid parsing
        if isinstance(evidence_list[0], list):
            evidence_list = [x for sub in evidence_list for x in sub]
        return evidence_list


class PeptideIdentification(MzIdentMLPeptide, _EvidenceMixin):
    def __init__(self, peptide_dict, enzyme=None, constant_modifications=None,
                 modification_translation_table=None, process=True):
        super(PeptideIdentification, self).__init__(
            peptide_dict, enzyme, constant_modifications, modification_translation_table,
            process)


class ProteinStub(object):
    name: str
    id: int
    protein_sequence: str
    n_glycan_sequon_sites: List[int]
    o_glycan_sequon_sites: List[int]
    glycosaminoglycan_sequon_sites: List[int]

    def __len__(self):
        return len(self.protein_sequence)

    def __init__(self, name, id, sequence, n_glycan_sequon_sites, o_glycan_sequon_sites,
                 glycosaminoglycan_sequon_sites):
        self.name = name
        self.id = id
        self.protein_sequence = sequence
        self.n_glycan_sequon_sites = n_glycan_sequon_sites
        self.o_glycan_sequon_sites = o_glycan_sequon_sites
        self.glycosaminoglycan_sequon_sites = glycosaminoglycan_sequon_sites

    @classmethod
    def from_protein(cls, protein):
        return cls(
            str(protein.name),
            getattr(protein, 'id', None),
            str(protein),
            protein.n_glycan_sequon_sites,
            protein.o_glycan_sequon_sites,
            protein.glycosaminoglycan_sequon_sites)

    def __repr__(self):
        trailer = self.protein_sequence[:10] + '...'
        return f"{self.__class__.__name__}({self.name!r}, {self.id}, {trailer!r})"


class ProteinStubLoaderBase(object):
    store: Mapping[str, ProteinStub]

    def __init__(self, store=None):
        if store is None:
            store = dict()
        self.store = store

    def __getitem__(self, key: str) -> ProteinStub:
        return self.get_protein_by_name(key)

    def __contains__(self, key: str):
        return key in self.store

    def _load_protein(self, protein_name: str):
        raise NotImplementedError()

    def get_protein_by_name(self, protein_name: str) -> ProteinStub:
        try:
            return self.store[protein_name]
        except KeyError:
            protein = self._load_protein(protein_name)
            try:
                stub = ProteinStub.from_protein(protein)
            except AttributeError:
                logger.error("Failed to load stub for %s" % (protein_name,))
                raise
            self.store[protein_name] = stub
            return stub


class MemoryProteinStubLoader(ProteinStubLoaderBase):
    def _load_protein(self, protein_name):
        raise KeyError(protein_name)


class DatabaseProteinStubLoader(ProteinStubLoaderBase):
    hypothesis_id: int
    alias_map: Dict[str, str]

    def __init__(self, session, hypothesis_id, store=None, alias_map: Optional[Dict[str, str]]=None):
        if alias_map is None:
            alias_map = {}
        self.session = session
        self.hypothesis_id = hypothesis_id
        self.alias_map = alias_map
        super(DatabaseProteinStubLoader, self).__init__(store)

    def _load_protein(self, protein_name: str) -> Protein:
        if protein_name in self.alias_map:
            protein_name = self.alias_map[protein_name]
        protein = self.session.query(Protein).filter(
            Protein.name == protein_name,
            Protein.hypothesis_id == self.hypothesis_id).first()
        if protein is None:
            breakpoint()
            raise KeyError(protein_name)
        return protein


class FastaProteinStubLoader(ProteinStubLoaderBase):
    def __init__(self, fasta_path, store=None):
        super(FastaProteinStubLoader, self).__init__(store)
        self.parser = glycopeptidepy_fasta.ProteinFastaFileParser(fasta_path, index=True)

    def _load_protein(self, protein_name: str):
        return self.parser[protein_name]


class PeptideMapperBase(object):
    enzyme: Protease
    constant_modifications: Dict[str, Modification]
    modification_translation_table: Dict[str, Modification]

    peptide_grouper: PeptideCollection
    protein_loader: ProteinStubLoaderBase


    counter: int
    peptide_length_range: Tuple[int, int]


    def __init__(self, enzyme=None, constant_modifications=None, modification_translation_table=None,
                 protein_filter=None, peptide_length_range=(5, 60)):
        if protein_filter is None:
            protein_filter = allset()

        self.enzyme = enzyme
        self.constant_modifications = constant_modifications or []
        self.modification_translation_table = modification_translation_table or {}

        self.peptide_grouper = PeptideCollection(protein_filter)
        self.counter = 0

        self.protein_loader = self._make_protein_loader()
        self.peptide_length_range = peptide_length_range

    def _make_protein_loader(self):
        raise NotImplementedError()


class PeptideConverter(PeptideMapperBase):
    hypothesis_id: int
    include_cysteine_n_glycosylation: bool

    def __init__(self, session, hypothesis_id, enzyme=None, constant_modifications=None,
                 modification_translation_table=None, protein_filter=None,
                 peptide_length_range=(5, 60),
                 include_cysteine_n_glycosylation: bool = False,
                 protein_alias_map: Optional[Dict[str, str]]=None):
        self.session = session
        self.hypothesis_id = hypothesis_id
        self.include_cysteine_n_glycosylation = include_cysteine_n_glycosylation
        self.protein_alias_map = protein_alias_map
        super(PeptideConverter, self).__init__(
            enzyme, constant_modifications, modification_translation_table,
            protein_filter, peptide_length_range)

    def _make_protein_loader(self):
        return DatabaseProteinStubLoader(self.session, self.hypothesis_id, alias_map=self.protein_alias_map)

    def get_protein(self, evidence):
        return self.protein_loader[evidence['accession']]

    def pack_peptide(self, peptide_ident: PeptideIdentification, start: int, end: int, score: float,
                     score_type: str, parent_protein: Protein):
        match = Peptide(
            calculated_mass=peptide_ident.peptide_sequence.mass,
            base_peptide_sequence=peptide_ident.base_sequence,
            modified_peptide_sequence=str(peptide_ident.peptide_sequence),
            formula=formula(peptide_ident.peptide_sequence.total_composition()),
            count_glycosylation_sites=None,
            count_missed_cleavages=peptide_ident.missed_cleavages,
            count_variable_modifications=peptide_ident.modification_counter,
            start_position=start,
            end_position=end,
            peptide_score=score,
            peptide_score_type=score_type,
            sequence_length=end - start,
            protein_id=parent_protein.id,
            hypothesis_id=self.hypothesis_id)
        n_glycosites = n_glycan_sequon_sites(
            match, parent_protein, include_cysteine=self.include_cysteine_n_glycosylation)
        o_glycosites = o_glycan_sequon_sites(match, parent_protein)
        gag_glycosites = gag_sequon_sites(match, parent_protein)
        match.count_glycosylation_sites = len(n_glycosites) + len(o_glycosites)
        match.n_glycosylation_sites = sorted(n_glycosites)
        match.o_glycosylation_sites = sorted(o_glycosites)
        match.gagylation_sites = sorted(gag_glycosites)
        return match

    def copy_db_peptide(self, db_peptide: Peptide):
        dup = Peptide(
            calculated_mass=db_peptide.calculated_mass,
            base_peptide_sequence=db_peptide.base_peptide_sequence,
            modified_peptide_sequence=db_peptide.modified_peptide_sequence,
            formula=db_peptide.formula,
            count_glycosylation_sites=db_peptide.count_glycosylation_sites,
            count_missed_cleavages=db_peptide.count_missed_cleavages,
            count_variable_modifications=db_peptide.count_variable_modifications,
            start_position=db_peptide.start_position,
            end_position=db_peptide.end_position,
            peptide_score=db_peptide.peptide_score,
            peptide_score_type=db_peptide.peptide_score_type,
            sequence_length=db_peptide.sequence_length,
            protein_id=db_peptide.protein_id,
            hypothesis_id=db_peptide.hypothesis_id,
            n_glycosylation_sites=db_peptide.n_glycosylation_sites,
            o_glycosylation_sites=db_peptide.o_glycosylation_sites,
            gagylation_sites=db_peptide.gagylation_sites)
        return dup

    def has_occupied_glycosites(self, db_peptide: Peptide):
        occupied_sites = []
        n_glycosylation_sites = db_peptide.n_glycosylation_sites
        peptide_obj = PeptideSequence(db_peptide.modified_peptide_sequence)

        for site in n_glycosylation_sites:
            if peptide_obj[site][1]:
                occupied_sites.append(site)
        return len(occupied_sites) > 0

    def clear_sites(self, db_peptide: Peptide):
        # TODO: Make this a combinatorial generator so that it optionally clears each
        # putative combination of glycosites across N/O forms.
        occupied_sites = []
        n_glycosylation_sites = db_peptide.n_glycosylation_sites
        peptide_obj = PeptideSequence(db_peptide.modified_peptide_sequence)

        for site in n_glycosylation_sites:
            if peptide_obj[site][1]:
                occupied_sites.append(site)

        for site in occupied_sites:
            peptide_obj.drop_modification(site, peptide_obj[site][1][0])

        copy = self.copy_db_peptide(db_peptide)
        copy.calculated_mass = peptide_obj.mass
        copy.modified_peptide_sequence = str(peptide_obj)
        copy.count_variable_modifications -= len(occupied_sites)
        return copy

    def sequence_starts_at(self, sequence: str, parent_protein: Protein) -> int:
        found = parent_protein.protein_sequence.find(sequence)
        if found == -1:
            raise ValueError("Peptide not found in Protein\n%s\n%r\n\n" % (parent_protein.name, sequence))
        return found

    def handle_peptide_dict(self, peptide_dict: Dict[str, Any]):
        peptide_ident = PeptideIdentification(
            peptide_dict, self.enzyme, self.constant_modifications,
            self.modification_translation_table)

        score, score_type = peptide_ident.score_type
        evidence_list = peptide_ident.evidence_list

        for evidence in evidence_list:
            if "skip" in evidence:
                continue
            if evidence.get("isDecoy", False):
                continue

            parent_protein = self.get_protein(evidence)
            if parent_protein is None:
                continue

            start = evidence["start"] - 1
            end = evidence["end"]
            length = len(peptide_ident.base_sequence)
            if not (self.peptide_length_range[0] <= length <= self.peptide_length_range[1]):
                continue

            sequence_copy = peptide_ident.original_sequence()
            found = self.sequence_starts_at(sequence_copy, parent_protein)

            if found != start:
                start = found
                end = start + length

            match = self.pack_peptide(
                peptide_ident, start, end, score, score_type, parent_protein)
            self.add_to_group(match)
            if self.has_occupied_glycosites(match):
                cleared = self.clear_sites(match)
                self.add_to_group(cleared)

    def add_to_group(self, match):
        self.counter += 1
        self.peptide_grouper.add(match)

    def save_accumulation(self):
        acc = []
        self.peptide_grouper.bind_scores()
        for key, group in self.peptide_grouper.items():
            acc.extend(group)
        self.session.bulk_save_objects(acc)
        self.session.commit()


class MzIdentMLProteomeExtraction(TaskBase):
    def __init__(self, mzid_path, reference_fasta=None, peptide_length_range=(5, 60)):
        self.mzid_path = mzid_path
        self.reference_fasta = reference_fasta
        self.parser = Parser(mzid_path, retrieve_refs=True, iterative=True, use_index=True)
        self.enzymes = []
        self.constant_modifications = []
        self.modification_translation_table = {}

        self._protein_resolver = None
        self._ignore_protein_regex = None
        self._used_database_path = None

        self.peptide_length_range = peptide_length_range or (5, 60)

    def load_enzyme(self):
        self.parser.reset()
        # self.enzymes = list({e['name'].lower() for e in self.parser.iterfind(
        #     "EnzymeName", retrieve_refs=True, iterative=True)})
        enzymes = list(self.parser.iterfind("Enzyme", retrieve_refs=True, iterative=True))
        processed_enzymes = []
        for enz in enzymes:
            permitted_missed_cleavages = int(enz.get("missedCleavages", 1))
            # It's a standard enzyme, so we can look it up by name
            if "EnzymeName" in enz:
                try:
                    enz_name = enz.get('EnzymeName')
                    if isinstance(enz_name, dict):
                        if len(enz_name) == 1:
                            enz_name = list(enz_name)[0]
                        else:
                            self.log("Could not interpret Enzyme Name %r" % (enz,))
                    protease = ParameterizedProtease(enz_name.lower(), permitted_missed_cleavages)
                    processed_enzymes.append(protease)
                except (KeyError, re.error) as e:
                    self.log("Could not resolve protease from name %r (%s)" % (enz['EnzymeName'].lower(), e))
            elif "SiteRegexp" in enz:
                pattern = enz['SiteRegexp']
                try:
                    protease = ParameterizedProtease(pattern, permitted_missed_cleavages)
                    processed_enzymes.append(protease)
                except re.error as e:
                    self.log("Could not resolve protease from name %r (%s)" % (enz['SiteRegexp'].lower(), e))
                except KeyError:
                    self.log("No protease information available: %r" % (enz,))
            elif "name" in enz and enz['name'].lower() in enzyme_rules:
                protease = ParameterizedProtease(enz["name"].lower(), permitted_missed_cleavages)
                processed_enzymes.append(protease)
            elif "id" in enz and enz['id'].lower() in enzyme_rules:
                protease = ParameterizedProtease(enz["id"].lower(), permitted_missed_cleavages)
                processed_enzymes.append(protease)
            else:
                self.log("No protease information available: %r" % (enz,))
        self.enzymes = list(set(processed_enzymes))

    def load_modifications(self):
        self.parser.reset()
        search_param_modifications = list(self.parser.iterfind(
            "ModificationParams", retrieve_refs=True, iterative=True))
        constant_modifications = []

        for param in search_param_modifications:
            for mod in param['SearchModification']:
                try:
                    name = mod['name']
                except KeyError:
                    name = mod['unknown modification']
                    try:
                        Modification(str(name))
                    except ModificationNameResolutionError:
                        self.modification_translation_table[name] = modification.AnonymousModificationRule(
                            str(name), mod['massDelta'])

                residues = mod['residues']
                if mod.get('fixedMod', False):
                    identifier = "%s (%s)" % (name, ''.join(residues).replace(" ", ""))
                    constant_modifications.append(identifier)
        self.constant_modifications = constant_modifications

    def _make_protein_resolver(self):
        if self.reference_fasta is not None:
            self._protein_resolver = FastaProteinSequenceResolver(self.reference_fasta)
            return
        else:
            path = self._find_used_database()
            if path is not None:
                self._protein_resolver = FastaProteinSequenceResolver(path)
                return
        raise ValueError("Cannot construct a Protein Resolver. Cannot fetch additional protein information.")

    def _clear_protein_resolver(self):
        self._protein_resolver = None

    def _find_used_database(self):
        if self._used_database_path is not None:
            return self._used_database_path
        self.parser.reset()
        databases = list(self.parser.iterfind("SearchDatabase", iterative=True))
        # use only the first database
        if len(databases) > 1:
            self.log("... %d databases found: %r" % (len(databases), databases))
            self.log("... Using first only")
        database = databases[0]
        if "decoy DB accession regexp" in database:
            self._ignore_protein_regex = re.compile(database["decoy DB accession regexp"])
        if "FASTA format" in database.get('FileFormat', {}):
            self.log("... Database described in FASTA format")
            db_location = database.get("location")
            if db_location is None:
                raise ValueError("No location present for database")
            else:
                try:
                    path = resolve_database_url(db_location)
                    with open(path, 'r') as handle:
                        for i, line in enumerate(handle):
                            if i > 1000:
                                raise ValueError("No FASTA Header before thousandth line. Probably not a FASTA file")
                            if line.startswith(">"):
                                break
                    self._used_database_path = path
                    return path
                except (IOError, ValueError):
                    return None
        else:
            return None

    def resolve_protein(self, name):
        if self._protein_resolver is None:
            self._make_protein_resolver()
        proteins = self._protein_resolver.find(name)
        if len(proteins) > 1:
            self.log("Protein Name %r resolved to multiple proteins: %r. Using first only." % (name, proteins))
        elif len(proteins) == 0:
            raise KeyError(name)
        return proteins[0]

    def _load_peptides(self):
        self.parser.reset()
        i = 0
        try:
            enzyme = self.enzymes[0]
        except IndexError as e:
            logger.exception("Enzyme not found.", exc_info=e)
            enzyme = None
        for peptide in self.parser.iterfind("Peptide", iterative=True, retrieve_refs=True):
            i += 1
            mzid_peptide = MzIdentMLPeptide(
                peptide, enzyme, self.constant_modifications,
                self.modification_translation_table)
            if i % 1000 == 0:
                self.log("Loaded Peptide %r" % (mzid_peptide,))
            yield mzid_peptide

    def _map_peptide_to_proteins(self):
        self.parser.reset()
        i = 0
        peptide_to_proteins = defaultdict(set)
        for evidence in self.parser.iterfind('PeptideEvidence', iterative=True):
            i += 1
            peptide_to_proteins[evidence['peptide_ref']].add(
                evidence['dBSequence_ref'])
        return peptide_to_proteins

    def _handle_protein(self, name, sequence, data):
        raise NotImplementedError()

    def load_proteins(self):
        self.parser.reset()
        self._find_used_database()
        protein_map = {}
        self.parser.reset()
        for protein in self.parser.iterfind("DBSequence", recursive=True, iterative=True):
            seq = protein.pop('Seq', None)
            name = protein.pop('accession')
            if seq is None:
                try:
                    prot = self.resolve_protein(name)
                    seq = prot.protein_sequence
                except KeyError:
                    if self._can_ignore_protein(name):
                        continue
                    else:
                        self.log("Could not resolve protein %r" % (name,))

            if "protein description" in protein:
                name = " ".join(
                    (name,
                     protein.pop("protein description"))
                )

            if name in protein_map:
                if seq != protein_map[name].protein_sequence:
                    self.log("Multiple proteins with the name %r" % name)
                continue
            try:
                p = self._handle_protein(name, seq, protein)
                protein_map[name] = p
            except residue.UnknownAminoAcidException:
                self.log("Unknown Amino Acid in %r" % (name,))
                continue
            except Exception as e:
                self.log("%r skipped: %r" % (name, e))
                continue
        self._clear_protein_resolver()
        return protein_map


class Proteome(DatabaseBoundOperation, MzIdentMLProteomeExtraction):
    def __init__(self, mzid_path, connection, hypothesis_id, include_baseline_peptides=True,
                 target_proteins=None, reference_fasta=None,
                 peptide_length_range=(5, 60), use_uniprot=True,
                 include_cysteine_n_glycosylation: bool=False,
                 uniprot_source_file: Optional[str]=None):
        DatabaseBoundOperation.__init__(self, connection)
        MzIdentMLProteomeExtraction.__init__(self, mzid_path, reference_fasta)
        if target_proteins is None:
            target_proteins = []
        self.hypothesis_id = hypothesis_id
        self.target_proteins = target_proteins
        self.include_baseline_peptides = include_baseline_peptides
        self.peptide_length_range = peptide_length_range or (5, 60)
        self.use_uniprot = use_uniprot
        self.accession_map = {}
        self.include_cysteine_n_glycosylation = include_cysteine_n_glycosylation
        self.uniprot_source_file = uniprot_source_file

    def _can_ignore_protein(self, name):
        if name not in self.target_proteins:
            return True
        elif (self._ignore_protein_regex is not None) and (
                self._ignore_protein_regex.match(name)):
            return True
        return False

    def _make_protein_from_state(self, name: str, seq: str, state: dict):
        p = Protein(
            name=name,
            protein_sequence=seq,
            other=state,
            hypothesis_id=self.hypothesis_id)
        p._init_sites(self.include_cysteine_n_glycosylation)
        return p

    def load_proteins(self):
        self.parser.reset()
        self._find_used_database()
        session = self.session
        protein_map = {}
        self.accession_map = {}
        self.parser.reset()
        for protein in self.parser.iterfind(
                "DBSequence", retrieve_refs=True, recursive=True, iterative=True):
            seq = protein.pop('Seq', None)
            accession = name = protein.pop('accession')
            if seq is None:
                try:
                    prot = self.resolve_protein(name)
                    seq = prot.protein_sequence
                except KeyError:
                    if self._can_ignore_protein(name):
                        continue
                    else:
                        self.log("Could not resolve protein %r" % (name,))

            if "protein description" in protein:
                name = " ".join(
                    (name,
                     protein.pop("protein description"))
                )

            if name in protein_map:
                if seq != protein_map[name].protein_sequence:
                    self.log("Multiple proteins with the name %r" % name)
                continue
            try:
                p = self._make_protein_from_state(name, seq, protein)
                session.add(p)
                session.flush()
                protein_map[name] = p
                self.accession_map[accession] = name
            except residue.UnknownAminoAcidException:
                self.log("Unknown Amino Acid in %r" % (name,))
                continue
            except Exception as e:
                self.log("%r skipped: %r" % (name, e))
                continue
        session.commit()
        self._clear_protein_resolver()

    def _make_peptide_converter(self, session, protein_filter: Set[int], enzyme: Protease, **kwargs):
        return PeptideConverter(
            session, self.hypothesis_id, enzyme,
            self.constant_modifications,
            self.modification_translation_table,
            protein_filter=protein_filter,
            protein_alias_map=self.accession_map, **kwargs)

    def load_spectrum_matches(self):
        self.parser.reset()
        last = 0
        i = 0
        try:
            enzyme = self.enzymes[0]
        except IndexError as e:
            logger.exception("Enzyme not found.", exc_info=e)
            enzyme = None
        session = self.session

        protein_filter = set(self.retrieve_target_protein_ids())

        peptide_converter = self._make_peptide_converter(
            session, protein_filter, enzyme)

        for spectrum_identification in self.parser.iterfind(
                "SpectrumIdentificationItem", retrieve_refs=True, iterative=True):
            peptide_converter.handle_peptide_dict(spectrum_identification)
            i += 1
            if i % 1000 == 0:
                self.log("... %d spectrum matches processed." % i)
            if (peptide_converter.counter - last) > 1000000:
                last = peptide_converter.counter
                self.log("... %d peptides saved. %d distinct cases." % (
                    peptide_converter.counter, len(
                        peptide_converter.peptide_grouper)))
        self.log("... %d peptides saved. %d distinct cases." % (
            peptide_converter.counter, len(
                peptide_converter.peptide_grouper)))
        peptide_converter.save_accumulation()

    def load(self):
        self.log("... Loading Enzyme")
        self.load_enzyme()
        self.log("... Loading Modifications")
        self.load_modifications()
        self.log("... Loading Proteins")
        self.load_proteins()
        self.log("... Loading Spectrum Matches")
        self.load_spectrum_matches()
        self.log("Sharing Common Peptides")
        self.remove_duplicates()
        self.share_common_peptides()
        self.remove_duplicates()
        self.log("... %d Peptides Total" % (self.count_peptides()))
        if self.include_baseline_peptides:
            self.log("... Building Baseline Peptides")
            self.build_baseline_peptides()
            self.remove_duplicates()
            self.log("... %d Peptides Total" % (self.count_peptides()))
            if self.use_uniprot:
                self.split_proteins()
        self.log("... Removing Duplicate Peptides")
        self.remove_duplicates()
        self.log("... %d Peptides Total" % (self.count_peptides()))

    def retrieve_target_protein_ids(self):
        if not self.target_proteins:
            return [
                i[0] for i in
                self.query(Protein.id).filter(
                    Protein.hypothesis_id == self.hypothesis_id).all()
            ]
        else:
            result = []
            for target in self.target_proteins:
                if isinstance(target, basestring):
                    if target in self.accession_map:
                        target = self.accession_map[target]
                    match = self.query(Protein.id).filter(
                        Protein.name == target,
                        Protein.hypothesis_id == self.hypothesis_id).first()
                    if match:
                        result.append(match[0])
                    else:
                        self.log("Could not locate protein '%s'" % target)
                elif isinstance(target, int):
                    result.append(target)
            return result

    def get_target_proteins(self):
        ids = self.retrieve_target_protein_ids()
        return [self.session.query(Protein).get(i) for i in ids]

    def count_peptides(self):
        peptide_count = self.session.query(Peptide).filter(
            Peptide.hypothesis_id == self.hypothesis_id).count()
        return peptide_count

    def make_restricted_modification_rules(self, rule_strings: List[str]):
        standard_rules = []
        alternative_rules: DefaultDict[str, List[str]] = DefaultDict(list)
        for rule_str in rule_strings:
            name = modification.RestrictedModificationTable.extract_name(rule_str)
            if name in self.modification_translation_table:
                if isinstance(self.modification_translation_table[name], AnonymousModificationRule):
                    alternative_rules[name].append(rule_str)
                else:
                    standard_rules.append(rule_str)
            else:
                standard_rules.append(rule_str)

        mod_table = modification.RestrictedModificationTable(
            constant_modifications=standard_rules,
            variable_modifications=[])
        rules = [mod_table[c] for c in standard_rules]
        if alternative_rules:
            for name, rule_strs in alternative_rules.items():
                rule: AnonymousModificationRule = self.modification_translation_table[name].clone()
                rule.targets = {modification.RestrictedModificationTable.extract_target(s) for s in rule_strs}
                rules.append(rule)
        return rules

    def build_baseline_peptides(self):
        const_modifications = self.make_restricted_modification_rules(self.constant_modifications)
        digestor = ProteinDigestor(
            self.enzymes[0], const_modifications,
            max_missed_cleavages=self.enzymes[0].used_missed_cleavages,
            include_cysteine_n_glycosylation=self.include_cysteine_n_glycosylation,
            min_length=self.peptide_length_range[0],
            max_length=self.peptide_length_range[1])
        accumulator = []
        i = 0
        for protein in self.get_target_proteins():
            for peptide in digestor.process_protein(protein):
                peptide.hypothesis_id = self.hypothesis_id
                accumulator.append(peptide)
                i += 1
                if len(accumulator) > 5000:
                    self.session.bulk_save_objects(accumulator)
                    self.session.commit()
                    accumulator = []
                if i % 1000 == 0:
                    self.log("... %d Baseline Peptides Created" % i)

        self.session.bulk_save_objects(accumulator)
        self.session.commit()

    def share_common_peptides(self):
        sharer = PeptideSharer(self._original_connection, self.hypothesis_id)
        proteins = self.get_target_proteins()
        i = 0
        n = len(proteins)
        for protein in proteins:
            i += 1
            # self.log("... Accumulating Proteins for %r" % protein)
            sharer.find_contained_peptides(protein)
            if i % 5 == 0:
                self.log("... %0.3f%% Done (%s)" % (i / float(n) * 100., protein.name))

    def split_proteins(self):
        const_modifications = self.make_restricted_modification_rules(self.constant_modifications)
        protein_ids = self.retrieve_target_protein_ids()
        if self.use_uniprot:
            annotator = UniprotProteinAnnotator(
                self,
                protein_ids,
                const_modifications,
                [],
                uniprot_source_file=self.uniprot_source_file
            )
            annotator.run()

    def remove_duplicates(self):
        DeduplicatePeptides(self._original_connection, self.hypothesis_id).run()


@dataclass
class _PositionedPeptideFacade:
    start_position: int
    end_position: int
    modified_peptide_sequence: str


class ReverseMzIdentMLPeptide(MzIdentMLPeptide):
    peptide_dict: Dict[str, Any]
    insert_sites: List[Tuple[int, str]]
    deleteion_sites: List[Tuple[int, str]]

    modification_counter: int
    missed_cleavages: Optional[int]

    base_sequence: str
    peptide_sequence: PeptideSequence
    glycosite_candidates: List[int]

    modification_translation_table: Dict[str, Modification]
    enzyme: Optional[Protease]
    mzid_id: Optional[str]

    def __init__(self, peptide_dict, protein: Protein, enzyme=None, constant_modifications=None,
                 modification_translation_table=None, process=True):
        if modification_translation_table is None:
            modification_translation_table = dict()
        if constant_modifications is None:
            constant_modifications = list()

        facade = self.update_sequence(peptide_dict, protein)

        self.peptide_dict = peptide_dict

        self.insert_sites = []
        self.deleteion_sites = []
        self.modification_counter = 0
        self.missed_cleavages = 0

        self.base_sequence = peptide_dict["PeptideSequence"]
        self.peptide_sequence = PeptideSequence(peptide_dict["PeptideSequence"])

        self.glycosite_candidates = n_glycan_sequon_sites(facade, protein)

        self.constant_modifications = constant_modifications
        self.modification_translation_table = modification_translation_table
        self.enzyme = enzyme
        self.mzid_id = peptide_dict.get('id')

        if process:
            self.process()

    def _reflect_position(self, position: int) -> int:
        n = len(self.base_sequence)
        return n - position

    def update_sequence(self, peptide_dict: dict, protein: Protein):
        seq = peptide_dict['PeptideSequence']
        start = protein.protein_sequence[::-1].find(seq)
        n = len(protein)
        # Reflect the coordinates around the reversal point
        end = n - start
        start = end - len(seq)
        seq = protein.protein_sequence[start + 1:end + 1]
        peptide_dict['PeptideSequence'] = seq
        return _PositionedPeptideFacade(start, end, seq)

    def handle_substitutions(self):
        if "SubstitutionModification" in self.peptide_dict:
            subs = self.peptide_dict["SubstitutionModification"]
            for sub in subs:
                pos = sub['location'] - 1
                pos = max(self._reflect_position(pos) - 2, 0)
                replace = Residue(sub["replacementResidue"])
                self.peptide_sequence.substitute(pos, replace)
                self.modification_counter += 1

    def add_modification(self, modification: Modification, position: int):
        if position == -1:
            targets = modification.rule.n_term_targets
            for t in targets:
                if t.position_modifier is not None and t.amino_acid_targets is None:
                    break
            else:
                position += 1
        if position == len(self.peptide_sequence):
            targets = modification.rule.c_term_targets
            for t in targets:
                if t.position_modifier is not None and t.amino_acid_targets is None:
                    break
            else:
                position -= 1
        if position == -1:
            self.peptide_sequence.n_term = modification
        elif position == len(self.peptide_sequence):
            self.peptide_sequence.c_term = modification
        else:
            new_position = max(self._reflect_position(position) - 2, 0)
            self.peptide_sequence.add_modification(new_position, modification)

    def add_insertion(self, site: int, symbol=None):
        site = self._reflect_position(site)
        self.insert_sites.append((site, symbol))

    def add_deletion(self, site: int, symbol):
        site = self._reflect_position(site)
        self.deleteion_sites.append((site, symbol))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.peptide_sequence}, {self.glycosite_candidates})"


class ReversePeptideIdentification(ReverseMzIdentMLPeptide, _EvidenceMixin):
    pass


class ReversePeptideConverter(PeptideConverter):
    def _get_evidence_list(self, peptide_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence_list = peptide_dict["PeptideEvidenceRef"]
        # Flatten the evidence list if it has extra nesting because of alternative
        # mzid parsing
        if isinstance(evidence_list[0], list):
            evidence_list = [x for sub in evidence_list for x in sub]
        return evidence_list

    def __init__(self, session, hypothesis_id, enzyme=None, constant_modifications=None,
                 modification_translation_table=None, protein_filter=None,
                 peptide_length_range=(5, 60),
                 include_cysteine_n_glycosylation: bool = False,
                 protein_alias_map: Optional[Dict[str, str]]=None):
        super().__init__(
            session, hypothesis_id, enzyme=enzyme, constant_modifications=constant_modifications,
            modification_translation_table=modification_translation_table, protein_filter=protein_filter,
            peptide_length_range=peptide_length_range,
            include_cysteine_n_glycosylation=include_cysteine_n_glycosylation,
            protein_alias_map=protein_alias_map
        )

    def handle_peptide_dict(self, peptide_dict: Dict[str, Any]):
        evidence_list = self._get_evidence_list(peptide_dict)
        for evidence in evidence_list:
            if "skip" in evidence:
                continue
            if evidence.get("isDecoy", False):
                continue

            parent_protein = self.get_protein(evidence)
            if parent_protein is None:
                continue

            peptide_ident = ReversePeptideIdentification(
                peptide_dict,
                parent_protein,
                self.enzyme,
                self.constant_modifications,
                self.modification_translation_table
            )

            score, score_type = peptide_ident.score_type

            start = evidence["start"] - 1
            end = evidence["end"]
            length = len(peptide_ident.base_sequence)
            if not (self.peptide_length_range[0] <= length <= self.peptide_length_range[1]):
                continue

            sequence_copy = peptide_ident.original_sequence()
            found = self.sequence_starts_at(sequence_copy, parent_protein)

            if found != start:
                start = found
                end = start + length

            match = self.pack_peptide(
                peptide_ident, start, end, score, score_type, parent_protein)
            self.add_to_group(match)
            if self.has_occupied_glycosites(match):
                cleared = self.clear_sites(match)
                self.add_to_group(cleared)


class ReverseProteome(Proteome, ProteinReversingMixin):
    def _make_protein_from_state(self, name: str, seq: str, state: dict):
        p = super()._make_protein_from_state(name, seq, state)
        p.sites = []
        self.reverse_protein(p)
        return p

    def _make_peptide_converter(self, session, protein_filter: Set[int], enzyme: Protease, **kwargs):
        return ReversePeptideConverter(
            session,
            self.hypothesis_id,
            enzyme=enzyme,
            constant_modifications=self.constant_modifications,
            modification_translation_table=self.modification_translation_table,
            protein_filter=protein_filter,
            protein_alias_map=self.accession_map, **kwargs)
