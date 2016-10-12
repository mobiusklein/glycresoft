import re
import logging

from glycopeptidepy.structure import sequence, modification, residue
from glypy.composition import formula

from glycan_profiling.serialize import (
    Peptide, Protein, DatabaseBoundOperation)

from glycan_profiling.task import TaskBase

from .enzyme import expasy_rules
from .mzid_parser import Parser
from .peptide_permutation import ProteinDigestor
from .remove_duplicate_peptides import DeduplicatePeptides
from .share_peptides import PeptideSharer


logger = logging.getLogger("mzid")


PeptideSequence = sequence.PeptideSequence
Residue = residue.Residue
Modification = modification.Modification
ModificationNameResolutionError = modification.ModificationNameResolutionError


PROTEOMICS_SCORE = ["PEAKS:peptideScore", "mascot:score", "PEAKS:proteinScore"]
WHITELIST_GLYCOSITE_PTMS = [Modification("Deamidation"), Modification("HexNAc")]


def protein_names(mzid_path, pattern=r'.*'):
    pattern = re.compile(pattern)
    parser = Parser(mzid_path, retrieve_refs=False,
                    iterative=False, build_id_cache=False, use_index=False)
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


def remove_peptide_sequence_alterations(base_sequence, insert_sites, delete_sites):
    """
    Remove all the sequence insertions and deletions in order to reconstruct the
    original peptide sequence.

    Parameters
    ----------
    base_sequence : str
        The peptide sequence string which contains a combination
        of insertion and deletions
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
            self.missed_cleavages = len(self.enzyme.findall(self.base_sequence))
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
                    except KeyError, e:
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


class PeptideConverter(object):
    def __init__(self, session, hypothesis_id, enzyme=None, constant_modifications=None,
                 modification_translation_table=None):
        self.session = session
        self.hypothesis_id = hypothesis_id
        self.enzyme = enzyme
        self.constant_modifications = constant_modifications or []
        self.modification_translation_table = modification_translation_table or {}

        self.accumulator = []
        self.chunk_size = 500
        self.counter = 0

    def get_protein(self, evidence):
        parent_protein = self.session.query(Protein).filter(
            Protein.name == evidence['accession'],
            Protein.hypothesis_id == self.hypothesis_id).first()
        return parent_protein

    def sequence_starts_at(self, sequence, parent_protein):
        found = parent_protein.protein_sequence.find(sequence)
        if found == -1:
            raise ValueError("Peptide not found in Protein\n%s\n%s\n\n" % (
                parent_protein.name, parent_protein.protein_sequence, (
                    sequence
                )))
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
        match.protein = parent_protein
        glycosites = parent_sequence_aware_n_glycan_sequon_sites(
            match, parent_protein)
        match.count_glycosylation_sites = len(glycosites)
        match.n_glycosylation_sites = list(glycosites)
        match.made = "packed"
        match.identification = peptide_ident
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
            n_glycosylation_sites=db_peptide.n_glycosylation_sites)
        dup.made = 'copied'
        dup.identification = db_peptide
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
                # match = self.pack_peptide(cleared, start, end, score, score_type, parent_protein)
                self.add_to_save_queue(cleared)

    def add_to_save_queue(self, match):
        self.counter += 1
        self.accumulator.append(match)
        assert abs(match.calculated_mass - match.convert().mass) < 0.01, abs(
            match.calculated_mass - match.convert().mass)
        if len(self.accumulator) > self.chunk_size:
            self.save_accumulation()

    def save_accumulation(self):
        self.session.add_all(self.accumulator)
        self.session.commit()
        self.accumulator = []


class Proteome(DatabaseBoundOperation, TaskBase):
    def __init__(self, mzid_path, connection, hypothesis_id, include_baseline_peptides=True,
                 target_proteins=None):
        DatabaseBoundOperation.__init__(self, connection)
        self.mzid_path = mzid_path
        self.hypothesis_id = hypothesis_id
        self.parser = Parser(mzid_path, retrieve_refs=True, iterative=False, build_id_cache=True)
        self.enzymes = []
        self.constant_modifications = []
        self.modification_translation_table = {}
        self.target_proteins = target_proteins

        self.include_baseline_peptides = True

    def load_enzyme(self):
        self.enzymes = list({e['name'].lower() for e in self.parser.iterfind(
            "EnzymeName", retrieve_refs=True, iterative=True)})

    def load_modifications(self):
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
                        self.modification_conversion_map[name] = modification.AnonymousModificationRule(
                            name, mod['massDelta'])

                residues = mod['residues']
                if mod.get('fixedMod', False):
                    identifier = "%s (%s)" % (name, ''.join(residues).replace(" ", ""))
                    constant_modifications.append(identifier)
        self.constant_modifications = constant_modifications

    def load_proteins(self):
        session = self.session
        for protein in self.parser.iterfind(
                "ProteinDetectionHypothesis", retrieve_refs=True, recursive=False, iterative=True):
            seq = protein.pop('Seq')
            try:
                p = Protein(
                    name=protein.pop('accession'),
                    protein_sequence=seq,
                    other=protein,
                    hypothesis_id=self.hypothesis_id)
                session.add(p)
                session.flush()
                # self.log("... Extracted %r" % p)
            except residue.UnknownAminoAcidException:
                continue
        session.commit()

    def load_spectrum_matches(self):
        last = 0
        i = 0
        try:
            enzyme = re.compile(expasy_rules.get(self.enzymes[0]))
        except KeyError, e:
            logger.exception("Enzyme not found.", exc_info=e)
            enzyme = None
        session = self.session
        peptide_converter = PeptideConverter(
            session, self.hypothesis_id, enzyme,
            self.constant_modifications,
            self.modification_translation_table)

        for spectrum_identification in self.parser.iterfind(
                "SpectrumIdentificationItem", retrieve_refs=True, iterative=True):
            peptide_converter.handle_peptide_dict(spectrum_identification)
            i += 1
            if i % 1000 == 0:
                self.log("%d spectrum matches processed." % i)
            if (peptide_converter.counter - last) > 1000:
                last = peptide_converter.counter
                self.log("%d peptides saved." % peptide_converter.counter)
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
        self.log("... Sharing Common Peptides")
        self.share_common_peptides()

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
                if isinstance(target, str):
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
                    self.session.add_all(accumulator)
                    self.session.commit()
                    accumulator = []
                if i % 10 == 0:
                    self.log("... %d Baseline Peptides Created" % i)

        self.session.add_all(accumulator)
        self.session.commit()

    def share_common_peptides(self):
        sharer = PeptideSharer(self._original_connection, self.hypothesis_id)
        for protein in self.get_target_proteins():
            self.log("... Accumulating Proteins for %r" % protein)
            sharer.find_contained_peptides(protein)

    def remove_duplicates(self):
        DeduplicatePeptides(self._original_connection, self.hypothesis_id).run()
