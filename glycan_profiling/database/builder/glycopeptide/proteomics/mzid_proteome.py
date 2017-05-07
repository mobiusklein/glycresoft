import os
import re
import operator
import logging
from collections import defaultdict
from six import string_types as basestring

from glycopeptidepy.structure import sequence, modification, residue
from glycopeptidepy.enzyme import Protease
from glypy.composition import formula

from glycan_profiling.serialize import (
    Peptide, Protein, DatabaseBoundOperation)

from glycan_profiling.task import TaskBase

from .mzid_parser import Parser
from .peptide_permutation import ProteinDigestor
from .remove_duplicate_peptides import DeduplicatePeptides
from .share_peptides import PeptideSharer
from .fasta import ProteinSequenceListResolver

logger = logging.getLogger("mzid")


PeptideSequence = sequence.PeptideSequence
Residue = residue.Residue
Modification = modification.Modification
ModificationNameResolutionError = modification.ModificationNameResolutionError

GT = "greater"
LT = "lesser"

PROTEOMICS_SCORE = {
    "PEAKS:peptideScore": GT,
    "mascot:score": GT,
    "PEAKS:proteinScore": GT,
    "MS-GF:EValue": LT
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


def parent_sequence_aware_n_glycan_sequon_sites(peptide, protein):
    sites = set(sequence.find_n_glycosylation_sequons(peptide.modified_peptide_sequence))
    sites |= set(site - peptide.start_position for site in protein.glycosylation_sites
                 if peptide.start_position <= site < peptide.end_position)
    return list(sites)


def o_glycan_sequon_sites(peptide, protein=None):
    sites = sequence.find_o_glycosylation_sequons(peptide.modified_peptide_sequence)
    return sites


def gag_sequon_sites(peptide, protein=None):
    sites = sequence.find_glycosaminoglycan_sequons(peptide.modified_peptide_sequence)
    return sites


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
    def __init__(self):
        self.members = dict()
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

    def _update_has_target_match(self, protein_set):
        for key in self.members:
            if key in protein_set:
                return True
        return False

    def update_state(self, protein_set):
        had = self.has_target_match
        if not had:
            has = self.has_target_match = self._update_has_target_match(protein_set)
        else:
            has = had
        if not had and has:
            for key in list(self.members):
                if key not in protein_set:
                    self.members.pop(key)
        else:
            if self._last.protein_id not in protein_set:
                try:
                    self.members.pop(self._last.protein_id)
                except KeyError:
                    pass

    def values(self):
        return self.members.values()

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


class PeptideCollection(object):
    def __init__(self, protein_set):
        self.store = defaultdict(PeptideGroup)
        self.protein_set = protein_set

    def add(self, peptide):
        comparator = score_comparator(peptide.peptide_score_type)
        group = self.store[peptide.modified_peptide_sequence]
        if group:
            first = group.first()
            if comparator(peptide.peptide_score, first.peptide_score):
                group.clear()
                group[peptide.protein_id] = peptide
                self.store[peptide.modified_peptide_sequence] = group
            elif first.peptide_score == peptide.peptide_score:
                group[peptide.protein_id] = peptide
        else:
            group[peptide.protein_id] = peptide
        group.update_state(self.protein_set)

    def items(self):
        for key, mapping in self.store.items():
            yield key, mapping.values()

    def __len__(self):
        return sum(map(len, self.store.values()))


class PeptideIdentification(object):
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

    def handle_modifications(self):
        if "Modification" in self.peptide_dict:
            mods = self.peptide_dict["Modification"]
            for mod in mods:
                pos = mod["location"] - 1
                try:
                    try:
                        _name = mod["name"]
                        modification = Modification(_name)
                    except KeyError as e:
                        if "unknown modification" in mod:
                            try:
                                _name = mod['unknown modification']
                                if _name in self.modification_translation_table:
                                    modification = self.modification_translation_table[_name]()
                                else:
                                    modification = Modification(_name)
                            except ModificationNameResolutionError:
                                raise KeyError("Cannot find key %s in %r" % (e, mod))
                        else:
                            raise KeyError("Cannot find key %s in %r" % (e, mod))

                    try:
                        rule_text = "%s (%s)" % (_name, mod["residues"][0])
                        if (rule_text not in self.constant_modifications) and not (
                                pos in self.glycosite_candidates and modification in WHITELIST_GLYCOSITE_PTMS):
                            self.modification_counter += 1
                    except KeyError:
                        self.modification_counter += 1

                    if pos == -1:
                        self.peptide_sequence.n_term = modification
                    elif pos == len(self.peptide_sequence):
                        self.peptide_sequence.c_term = modification
                    else:
                        self.peptide_sequence.add_modification(pos, modification)
                except KeyError:
                    if "unknown modification" in mod:
                        mod_description = mod["unknown modification"]
                        insertion = re.search(r"(\S{3})\sinsertion", mod_description)
                        deletion = re.search(r"(\S{3})\sdeletion", mod_description)
                        self.modification_counter += 1
                        if insertion:
                            self.insert_sites.append((mod['location'] - 1, None))
                        elif deletion:
                            sym = Residue(deletion.groups()[0]).symbol
                            self.deleteion_sites.append((mod['location'] - 1, sym))
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

    @property
    def score_type(self):
        score = score_type = None
        for k, v in self.peptide_dict.items():
            if k in PROTEOMICS_SCORE:
                score_type = k
                score = v
                break
        return score, score_type

    @property
    def evidence_list(self):
        evidence_list = self.peptide_dict["PeptideEvidenceRef"]
        # Flatten the evidence list if it has extra nesting because of alternative
        # mzid parsing
        if isinstance(evidence_list[0], list):
            evidence_list = [x for sub in evidence_list for x in sub]
        return evidence_list

    def __repr__(self):
        return "PeptideIdentification(%s, %s)" % (self.peptide_sequence, self.glycosite_candidates)


class ProteinStub(object):
    def __init__(self, name, id, sequence, glycosylation_sites):
        self.name = name
        self.id = id
        self.protein_sequence = sequence
        self.glycosylation_sites = glycosylation_sites

    @classmethod
    def from_protein(cls, protein):
        return cls(protein.name, protein.id, protein.protein_sequence, protein.glycosylation_sites)


class ProteinStubLoader(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id
        self.store = dict()

    def _get_protein_from_db(self, protein_name):
        protein = self.session.query(Protein).filter(
            Protein.name == protein_name,
            Protein.hypothesis_id == self.hypothesis_id).first()
        return protein

    def get_protein_by_name(self, protein_name):
        try:
            return self.store[protein_name]
        except KeyError:
            db_prot = self._get_protein_from_db(protein_name)
            stub = ProteinStub.from_protein(db_prot)
            self.store[protein_name] = stub
            return stub

    def __getitem__(self, key):
        return self.get_protein_by_name(key)


class PeptideConverter(object):
    def __init__(self, session, hypothesis_id, enzyme=None, constant_modifications=None,
                 modification_translation_table=None, protein_filter=None):
        if protein_filter is None:
            protein_filter = allset()
        self.session = session
        self.hypothesis_id = hypothesis_id

        self.protein_loader = ProteinStubLoader(self.session, self.hypothesis_id)

        self.enzyme = enzyme
        self.constant_modifications = constant_modifications or []
        self.modification_translation_table = modification_translation_table or {}

        self.peptide_grouper = PeptideCollection(protein_filter)

        self.accumulator = []
        self.chunk_size = 500
        self.counter = 0

    def get_protein(self, evidence):
        return self.protein_loader[evidence['accession']]

    def sequence_starts_at(self, sequence, parent_protein):
        found = parent_protein.protein_sequence.find(sequence)
        if found == -1:
            print(len(parent_protein.protein_sequence))
            raise ValueError("Peptide not found in Protein\n%s\n%r\n\n" % (parent_protein.name, sequence))
        return found

    def pack_peptide(self, peptide_ident, start, end, score, score_type, parent_protein):
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
        n_glycosites = parent_sequence_aware_n_glycan_sequon_sites(
            match, parent_protein)
        o_glycosites = o_glycan_sequon_sites(match, parent_protein)
        gag_glycosites = gag_sequon_sites(match, parent_protein)
        match.count_glycosylation_sites = len(n_glycosites) + len(o_glycosites)
        match.n_glycosylation_sites = sorted(n_glycosites)
        match.o_glycosylation_sites = sorted(o_glycosites)
        match.gagylation_sites = sorted(gag_glycosites)
        return match

    def copy_db_peptide(self, db_peptide):
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

    def has_occupied_glycosites(self, db_peptide):
        occupied_sites = []
        n_glycosylation_sites = db_peptide.n_glycosylation_sites
        peptide_obj = PeptideSequence(db_peptide.modified_peptide_sequence)

        for site in n_glycosylation_sites:
            if peptide_obj[site][1]:
                occupied_sites.append(site)
        return len(occupied_sites) > 0

    def clear_sites(self, db_peptide):
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

    def handle_peptide_dict(self, peptide_dict):
        peptide_ident = PeptideIdentification(
            peptide_dict, self.enzyme, self.constant_modifications,
            self.modification_translation_table)

        score, score_type = peptide_ident.score_type
        evidence_list = peptide_ident.evidence_list

        for evidence in evidence_list:
            if "skip" in evidence:
                continue
            if evidence["isDecoy"]:
                continue

            parent_protein = self.get_protein(evidence)

            if parent_protein is None:
                continue
            start = evidence["start"] - 1
            end = evidence["end"]

            sequence_copy = peptide_ident.original_sequence()
            found = self.sequence_starts_at(sequence_copy, parent_protein)

            if found != start:
                start = found
                end = start + len(peptide_ident.base_sequence)
            match = self.pack_peptide(peptide_ident, start, end, score, score_type, parent_protein)
            self.add_to_save_queue(match)
            if self.has_occupied_glycosites(match):
                cleared = self.clear_sites(match)
                self.add_to_save_queue(cleared)

    def add_to_save_queue(self, match):
        self.counter += 1
        self.peptide_grouper.add(match)
        # assert abs(match.calculated_mass - match.convert().mass) < 0.01, abs(
        #     match.calculated_mass - match.convert().mass)
        if len(self.accumulator) > self.chunk_size:
            self.save_accumulation()

    def save_accumulation(self):
        acc = []
        for key, group in self.peptide_grouper.items():
            acc.extend(group)
        self.accumulator.extend(acc)
        self.session.bulk_save_objects(self.accumulator)
        self.session.commit()
        self.accumulator = []


class MzIdentMLProteomeExtraction(TaskBase):
    def __init__(self, mzid_path, reference_fasta=None):
        self.mzid_path = mzid_path
        self.reference_fasta = reference_fasta
        self.parser = Parser(mzid_path, retrieve_refs=True, iterative=True, use_index=True)
        self.enzymes = []
        self.constant_modifications = []
        self.modification_translation_table = {}

        self._protein_resolver = None
        self._ignore_protein_regex = None
        self._used_database_path = None

    def load_enzyme(self):
        self.parser.reset()
        # self.enzymes = list({e['name'].lower() for e in self.parser.iterfind(
        #     "EnzymeName", retrieve_refs=True, iterative=True)})
        enzymes = list(self.parser.iterfind("Enzyme", retrieve_refs=True, iterative=True))
        processed_enzymes = []
        for enz in enzymes:
            # It's a standard enzyme, so we can look it up by name
            if "EnzymeName" in enz:
                try:
                    protease = Protease(enz['EnzymeName'].lower())
                    processed_enzymes.append(protease)
                except (KeyError, re.error) as e:
                    self.log("Could not resolve protease from name %r (%s)" % (enz['EnzymeName'].lower(), e))
            else:
                try:
                    pattern = enz['SiteRegexp']
                    try:
                        protease = Protease(pattern)
                        processed_enzymes.append(protease)
                    except re.error as e:
                        self.log("Could not resolve protease from name %r (%s)" % (enz['EnzymeName'].lower(), e))
                except KeyError:
                    self.log("No protease information available: %r" % (enz,))
        self.enzymes = processed_enzymes

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
                        Modification(name)
                    except ModificationNameResolutionError:
                        self.modification_translation_table[name] = modification.AnonymousModificationRule(
                            name, mod['massDelta'])

                residues = mod['residues']
                if mod.get('fixedMod', False):
                    identifier = "%s (%s)" % (name, ''.join(residues).replace(" ", ""))
                    constant_modifications.append(identifier)
        self.constant_modifications = constant_modifications

    def _make_protein_resolver(self):
        if self.reference_fasta is not None:
            self._protein_resolver = ProteinSequenceListResolver.build_from_fasta(self.reference_fasta)
            return
        else:
            path = self._find_used_database()
            if path is not None:
                self._protein_resolver = ProteinSequenceListResolver.build_from_fasta(path)
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


class Proteome(DatabaseBoundOperation, MzIdentMLProteomeExtraction):
    def __init__(self, mzid_path, connection, hypothesis_id, include_baseline_peptides=True,
                 target_proteins=None, reference_fasta=None):
        DatabaseBoundOperation.__init__(self, connection)
        MzIdentMLProteomeExtraction.__init__(self, mzid_path, reference_fasta)
        self.hypothesis_id = hypothesis_id
        self.target_proteins = target_proteins
        self.include_baseline_peptides = include_baseline_peptides

    def _can_ignore_protein(self, name):
        if name not in self.target_proteins:
            return True
        elif (self._ignore_protein_regex is not None) and (
                self._ignore_protein_regex.match(name)):
            return True
        return False

    def load_proteins(self):
        self.parser.reset()
        self._find_used_database()
        session = self.session
        protein_map = {}
        self.parser.reset()
        for protein in self.parser.iterfind(
                "DBSequence", retrieve_refs=True, recursive=True, iterative=True):
            # check = protein.copy()
            seq = protein.pop('Seq', None)
            name = protein.pop('accession')
            # name += protein.pop("protein description", "")
            if seq is None:
                try:
                    prot = self.resolve_protein(name)
                    seq = prot.protein_sequence
                except KeyError:
                    if self._can_ignore_protein(name):
                        continue
                    else:
                        self.log("Could not resolve protein %r" % (name,))

            if name in protein_map:
                if seq != protein_map[name].protein_sequence:
                    self.log("Multiple proteins with the name %r" % name)
                continue
            try:
                p = Protein(
                    name=name,
                    protein_sequence=seq,
                    other=protein,
                    hypothesis_id=self.hypothesis_id)
                session.add(p)
                session.flush()
                protein_map[name] = p
            except residue.UnknownAminoAcidException:
                self.log("Unknown Amino Acid in %r" % (name,))
                continue
            except Exception as e:
                self.log("%r skipped: %r" % (name, e))
                continue
        session.commit()
        self._clear_protein_resolver()

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

        peptide_converter = PeptideConverter(
            session, self.hypothesis_id, enzyme,
            self.constant_modifications,
            self.modification_translation_table,
            protein_filter=protein_filter)

        for spectrum_identification in self.parser.iterfind(
                "SpectrumIdentificationItem", retrieve_refs=True, iterative=True):
            peptide_converter.handle_peptide_dict(spectrum_identification)
            i += 1
            if i % 1000 == 0:
                self.log("%d spectrum matches processed." % i)
            if (peptide_converter.counter - last) > 1000000:
                last = peptide_converter.counter
                self.log("%d peptides saved. %d distinct cases." % (
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
        if self.include_baseline_peptides:
            self.log("... Building Baseline Peptides")
            self.build_baseline_peptides()
        self.log("... Removing Duplicate Peptides")
        self.remove_duplicates()

    def retrieve_target_protein_ids(self):
        if len(self.target_proteins) == 0:
            return [
                i[0] for i in
                self.query(Protein.id).filter(
                    Protein.hypothesis_id == self.hypothesis_id).all()
            ]
        else:
            result = []
            for target in self.target_proteins:
                if isinstance(target, basestring):
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

    def build_baseline_peptides(self):
        mod_table = modification.RestrictedModificationTable(
            constant_modifications=self.constant_modifications,
            variable_modifications=[])
        const_modifications = [mod_table[c] for c in self.constant_modifications]
        digestor = ProteinDigestor(self.enzymes[0], const_modifications, max_missed_cleavages=1)
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

    def remove_duplicates(self):
        DeduplicatePeptides(self._original_connection, self.hypothesis_id).run()
