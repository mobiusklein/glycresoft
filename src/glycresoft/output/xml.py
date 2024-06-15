import io
import os
import re
import bisect

from collections import defaultdict, OrderedDict, namedtuple, deque
from typing import Any, Set, List, Union, Dict, Optional

from brainpy import mass_charge_ratio

from glypy.composition import formula
from glypy.io.nomenclature import identity
from glypy.structure.glycan_composition import (
    MonosaccharideResidue,
    FrozenMonosaccharideResidue, SubstituentResidue, FrozenGlycanComposition)

from glycopeptidepy.structure import parser, modification, PeptideSequence

from psims.mzid import components
from psims.mzid.writer import MzIdentMLWriter
from psims.controlled_vocabulary.controlled_vocabulary import load_gno

from ms_deisotope.output import mzml, ProcessedMSFileLoader

from glycresoft import task, serialize, version
from glycresoft.chromatogram_tree import Unmodified
from glycresoft.chromatogram_tree.chromatogram import group_by
from glycresoft.structure import FragmentCachingGlycopeptide


GlycopeptideType = Union[PeptideSequence, FragmentCachingGlycopeptide, serialize.Glycopeptide]
Props = Dict[str, Any]


class mass_term_pair(namedtuple("mass_term_pair", ('mass', 'term'))):
    def __lt__(self, other):
        return self.mass < float(other)

    def __gt__(self, other):
        return self.mass > float(other)

    def __float__(self):
        return self.mass


valid_monosaccharide_names = [
    "Hex",
    "HexNAc",
    "dHex",
    "NeuAc",
    "NeuGc",
    "Pen",
    "Fuc",
    # "HexA",
    # "HexN",
]

valid_monosaccharides = {FrozenMonosaccharideResidue.from_iupac_lite(v): v
                         for v in valid_monosaccharide_names}


def monosaccharide_to_term(monosaccharide):
    try:
        return valid_monosaccharides[monosaccharide]
    except KeyError:
        value = str(monosaccharide)
        return value


substituent_map = {
    "S": "sulfate",
    "P": "phosphate",
    # "Me": "methyl",
    # "Ac": "acetyl",
}

inverted_substituent_map = {
    v: k for k, v in substituent_map.items()
}

substituent_map['Sulpho'] = "sulfate"
substituent_map['Phospho'] = "phosphate"


def mparam(name, value=None, accession=None, cvRef="PSI-MS", **kwargs):
    if isinstance(name, dict):
        value = name.pop('value', None)
        accession = name.pop('accession')
        cvRef = name.pop('cvRef', cvRef)
        name_ = name.pop("name")
        kwargs.update(kwargs)
        name = name_
    return components.CVParam(
        name=name,
        value=value,
        accession=accession,
        cvRef=cvRef,
        **kwargs)


def parse_glycan_formula(glycan_formula):
    gc = FrozenGlycanComposition()
    if glycan_formula.startswith("\""):
        glycan_formula = glycan_formula[1:-1]
    for mono, count in re.findall(r"([^0-9]+)\((\d+)\)", glycan_formula):
        count = int(count)
        if mono in substituent_map:
            parsed = SubstituentResidue(substituent_map[mono])
        elif mono in ("Sia", ):
            continue
        elif mono in ("Pent", ):
            mono = "Pen"
            parsed = FrozenMonosaccharideResidue.from_iupac_lite(mono)
        elif mono == 'Xxx':
            continue
        elif mono == 'X':
            continue
        else:
            parsed = FrozenMonosaccharideResidue.from_iupac_lite(mono)
        gc[parsed] += count
    return gc


class GNOmeResolver(object):
    def __init__(self, cv=None):
        if cv is None:
            cv = load_gno()
        self.cv = cv
        self.build_mass_search_index()
        self.add_glycan_compositions()

    def add_glycan_compositions(self):
        formula_key = "GNO:00000202"
        for term in self.cv.terms.values():
            glycan_formula = term.get(formula_key)
            if glycan_formula:
                term['glycan_composition'] = parse_glycan_formula(glycan_formula)

    def build_mass_search_index(self):
        mass_index = []

        for term in self.cv.terms.values():
            match = re.search(r"weight of (\d+\.\d+) Da", term.definition)
            if match:
                mass = float(match.group(1))
                term['mass'] = mass
                mass_index.append(mass_term_pair(mass, term))

        mass_index.sort()
        self.mass_index = mass_index

    def _find_mass_match(self, mass):
        i = bisect.bisect_left(self.mass_index, mass)
        lo = self.mass_index[i - 1]
        lo_err = abs(lo.mass - mass)
        hi = self.mass_index[i]
        hi_err = abs(hi.mass - mass)
        if hi_err < lo_err:
            term = hi.term
        elif hi_err > lo_err:
            term = lo.term
        else:
            raise ValueError(
                "Ambiguous duplicate masses (%0.2f, %0.2f)" % (lo.mass, hi.mass))
        return term

    def resolve_gnome(self, glycan_composition):
        mass = glycan_composition.mass()
        term = self._find_mass_match(mass)
        recast = glycan_composition.clone().reinterpret(valid_monosaccharides)
        visit_queue = deque(term.children)
        while visit_queue:
            child = visit_queue.popleft()
            gc = child.get("glycan_composition")
            if gc is None:
                visit_queue.extend(child.children)
            elif gc == recast:
                return child

    def glycan_composition_to_terms(self, glycan_composition):
        out = []
        term = self.resolve_gnome(glycan_composition)
        if term is not None:
            out.append({
                "accession": term.id,
                "name": term.name,
                "cvRef": term.vocabulary.name
            })
        reinterpreted = glycan_composition.clone().reinterpret(valid_monosaccharides)
        for mono, count in reinterpreted.items():
            if isinstance(mono, SubstituentResidue):
                subst = inverted_substituent_map.get(
                    mono.name.replace("@", ""))
                if subst is not None:
                    out.append({
                        "name": "monosaccharide count",
                        "value": ("%s:%d" % (subst, count)),
                        "accession": "MS:XXXXX2",
                        "cvRef": "PSI-MS"
                    })
                else:
                    out.append({
                        "name": "unknown monosaccharide count",
                        "value": ("%s:%0.3f:%d" % (mono.name.replace("@", ""), mono.mass(), count)),
                        "accession": "MS:XXXXX3",
                        "cvRef": "PSI-MS"
                    })
            elif isinstance(mono, MonosaccharideResidue):
                for known in valid_monosaccharides:
                    if identity.is_a(mono, known):
                        out.append({
                            "name": "monosaccharide count",
                            "value": ("%s:%d" % (monosaccharide_to_term(known), count)),
                            "accession": "MS:XXXXX2",
                            "cvRef": "PSI-MS"
                        })
                        break
                else:
                    out.append({
                        "name": "unknown monosaccharide count",
                        "value": ("%s:%0.3f:%d" % (monosaccharide_to_term(mono), mono.mass(), count)),
                        "accession": "MS:XXXXX3",
                        "cvRef": "PSI-MS"
                    })
            else:
                raise TypeError("Cannot handle unexpected component of type %s" % (type(mono), ))
        return out


class SequenceIdTracker(object):
    mapping: Dict[str, int]

    def __init__(self):
        self.mapping = dict()

    def convert(self, glycopeptide: GlycopeptideType) -> int:
        s = str(glycopeptide)
        if s in self.mapping:
            return self.mapping[s]
        else:
            self.mapping[s] = glycopeptide.id
            return self.mapping[s]

    def __call__(self, glycopeptide: GlycopeptideType) -> int:
        return self.convert(glycopeptide)

    def dump(self):
        for key, value in self.mapping.items():
            print(value, key)


class MzMLExporter(task.TaskBase):
    def __init__(self, source, outfile):
        self.reader = ProcessedMSFileLoader(source)
        self.outfile = outfile
        self.writer = None
        self.n_spectra = None

    def make_writer(self):
        self.writer = mzml.MzMLScanSerializer(
            self.outfile, sample_name=self.reader.sample_run.name,
            n_spectra=self.n_spectra)

    def aggregate_scan_bunches(self, scan_ids):
        scans = defaultdict(list)
        for scan_id in scan_ids:
            scan = self.reader.get_scan_by_id(scan_id)
            scans[scan.precursor_information.precursor_scan_id].append(
                scan)
        bunches = []
        for precursor_id, products in scans.items():
            products.sort(key=lambda x: x.scan_time)
            precursor = self.reader.get_scan_by_id(precursor_id)
            bunches.append(mzml.ScanBunch(precursor, products))
        bunches.sort(key=lambda bunch: bunch.precursor.scan_time)
        return bunches

    def begin(self, scan_bunches):
        self.n_spectra = sum(len(b.products) for b in scan_bunches) + len(scan_bunches)
        self.make_writer()
        for bunch in scan_bunches:
            self.put_scan_bunch(bunch)

    def put_scan_bunch(self, bunch):
        self.writer.save_scan_bunch(bunch)

    def extract_chromatograms_from_identified_glycopeptides(self, glycopeptide_list):
        by_chromatogram = group_by(
            glycopeptide_list, lambda x: (
                x.chromatogram.chromatogram if x.chromatogram is not None else None))
        i = 0
        for chromatogram, members in by_chromatogram.items():
            if chromatogram is None:
                continue
            self.enqueue_chromatogram(chromatogram, i, params=[
                {"name": "GlycReSoft:profile score", "value": members[0].ms1_score},
                {"name": "GlycReSoft:assigned entity", "value": str(members[0].structure)}
            ])
            i += 1

    def enqueue_chromatogram(self, chromatogram, chromatogram_id, params=None):
        if params is None:
            params = []
        chromatogram_data = dict()
        rt, signal = chromatogram.as_arrays()
        chromatogram_dict = OrderedDict(zip(rt, signal))
        chromatogram_data['chromatogram'] = chromatogram_dict
        chromatogram_data['chromatogram_type'] = 'selected ion current chromatogram'
        chromatogram_data['id'] = chromatogram_id
        chromatogram_data['params'] = params

        self.writer.chromatogram_queue.append(chromatogram_data)

    def complete(self):
        self.writer.complete()
        self.writer.format()


glycosylation_type_to_term_map = {
    "N-Linked": {
        "name": "N-glycan",
        "accession": "MS:XXXXX5",
        "cvRef": "PSI-MS",
    },
    "O-Linked": {
        "name": "mucin O-glycan",
        "accession": "MS:XXXXX6",
        "cvRef": "PSI-MS",
    },
    "GAG linker": {
        "name": "glycosaminoglycan",
        "accession": "MS:XXXXX7",
        "cvRef": "PSI-MS",
    },
}

glycosylation_type_to_term_map["N-Glycan"] = glycosylation_type_to_term_map["N-Linked"]
glycosylation_type_to_term_map["O-Glycan"] = glycosylation_type_to_term_map["O-Linked"]


def glycosylation_type_to_term(glycosylation_type):
    return glycosylation_type_to_term_map[glycosylation_type]


class MzIdentMLSerializer(task.TaskBase):
    outfile: Union[os.PathLike, io.IOBase]

    analysis: serialize.Analysis
    database_handle: serialize.DatabaseBoundOperation
    gnome_resolver: GNOmeResolver

    _id_tracker: SequenceIdTracker
    _glycopeptide_list: List[serialize.IdentifiedGlycopeptide]
    _peptide_evidence: List[Props]

    protein_list: List[serialize.Protein]
    glycan_list: List[serialize.GlycanCombination]

    scan_ids: Set[str]

    q_value_threshold: float
    ms2_score_threshold: float

    export_mzml: bool
    source_mzml_path: Optional[os.PathLike]
    output_mzml_path: Optional[os.PathLike]

    embed_protein_sequences: bool
    report_top_match_per_glycopeptide: bool

    def __init__(self, outfile: Union[os.PathLike, io.IOBase],
                 glycopeptide_list: List[serialize.IdentifiedGlycopeptide],
                 analysis: serialize.Analysis,
                 database_handle: serialize.DatabaseBoundOperation,
                 q_value_threshold: float=0.05,
                 ms2_score_threshold: float=0,
                 export_mzml: bool=True,
                 source_mzml_path: Optional[os.PathLike]=None,
                 output_mzml_path: Optional[os.PathLike]=None,
                 embed_protein_sequences: bool=True,
                 report_top_match_per_glycopeptide: bool=True):

        self.outfile = outfile

        self.database_handle = database_handle
        self.analysis = analysis
        self.gnome_resolver = GNOmeResolver()

        self._glycopeptide_list = glycopeptide_list
        self.protein_list = []
        self.glycan_list = []
        self._peptide_evidence = []
        self.scan_ids = set()
        self._id_tracker = SequenceIdTracker()

        self.q_value_threshold = q_value_threshold
        self.ms2_score_threshold = ms2_score_threshold

        self.export_mzml = export_mzml
        self.source_mzml_path = source_mzml_path
        self.output_mzml_path = output_mzml_path

        self.report_top_match_per_glycopeptide = report_top_match_per_glycopeptide
        self.embed_protein_sequences = embed_protein_sequences

    def _coerce_orm(self, obj):
        if isinstance(obj, serialize.Base):
            obj = obj.convert()
        return obj

    @property
    def glycopeptide_list(self):
        return self._glycopeptide_list

    def extract_proteins(self):
        self.protein_list = self.database_handle.query(serialize.Protein).all()

    def extract_glycans(self):
        self.glycan_list = self.database_handle.query(serialize.GlycanCombination).all()

    def convert_to_peptide_dict(self, glycopeptide: GlycopeptideType, id_tracker: SequenceIdTracker) -> Props:
        data = {
            "id": glycopeptide.id,
            "peptide_sequence": parser.strip_modifications(glycopeptide),
            "modifications": []
        }

        glycopeptide = self._coerce_orm(glycopeptide)

        i = 0
        # TODO: handle N-terminal and C-terminal modifications
        glycosylation_event_count = len(glycopeptide.glycosylation_manager)
        glycosylation_events_handled = 0
        for _pos, mods in glycopeptide:
            i += 1
            if not mods:
                continue
            else:
                mod = mods[0]
            if mod.rule.is_a("glycosylation"):
                glycosylation_events_handled += 1
                is_aggregate_stub = False
                mod_params = [
                    glycosylation_type_to_term(
                        str(mod.rule.glycosylation_type))
                ]
                if mod.rule.is_core:

                    mod_params.extend(
                        self.gnome_resolver.glycan_composition_to_terms(glycopeptide.glycan_composition.clone()))

                    mass = glycopeptide.glycan_composition.mass()
                    if glycosylation_event_count == 1:
                        mod_params.append({
                            "name": "glycan composition",
                            "cvRef": "PSI-MS",
                            "accession": "MS:XXXX14"
                        })
                    else:
                        mod_params.append({
                            "name": "glycan aggregate",
                            "cvRef": "PSI-MS",
                            "accession": "MS:XXXX15"
                        })
                        if glycosylation_events_handled > 1:
                            mass = 0
                            is_aggregate_stub = True

                    if not is_aggregate_stub:
                        mod_params.append({
                            "accession": 'MS:1000864',
                            "cvRef": "PSI-MS",
                            "name": "chemical formula",
                            "value": formula(glycopeptide.glycan_composition.total_composition()),
                        })

                else:
                    mod_params.append({
                        "accession": 'MS:1000864',
                        "cvRef": "PSI-MS",
                        "name": "chemical formula",
                        "value": formula(mod.rule.composition),
                    })
                    if mod.rule.is_composition:
                        mod_params.extend(self.gnome_resolver.glycan_composition_to_terms(mod.rule.glycan.clone()))
                        mod_params.append({
                            "name": "glycan composition",
                            "cvRef": "PSI-MS",
                            "accession": "MS:XXXX14"
                        })
                    else:
                        mod_params.append({
                            "name": "glycan structure",
                            "cvRef": "PSI-MS",
                            "accession": "MS:XXXXXXX"
                        })
                    mass = mod.mass

                mod_dict = {
                    "monoisotopic_mass_delta": mass,
                    "location": i,
                    # "name": "unknown modification",
                    "name": "glycosylation modification",
                    "params": [components.CVParam(**x) for x in mod_params]
                }
                data['modifications'].append(mod_dict)
            else:
                mod_dict = {
                    "monoisotopic_mass_delta": mod.mass,
                    "location": i,
                    "name": mod.name,
                }
                data['modifications'].append(mod_dict)
        return data

    def _encode_score_set(self, spectrum_match: serialize.GlycopeptideSpectrumMatch) -> List[Union[Props, components.CVParam]]:
        score_params = [
            mparam("GlycReSoft:peptide score",
                   spectrum_match.score_set.peptide_score, "MS:XXX10C"),
            mparam("GlycReSoft:glycan score",
                   spectrum_match.score_set.glycan_score, "MS:XXX10B"),
            mparam("GlycReSoft:glycan coverage",
                   spectrum_match.score_set.glycan_coverage, "MS:XXX10H"),
            mparam("GlycReSoft:joint q-value",
                   spectrum_match.q_value, "MS:XXX10G"),
            mparam("GlycReSoft:peptide q-value",
                   spectrum_match.q_value_set.peptide_q_value,
                   "MS:XXX10E"),
            mparam("GlycReSoft:glycan q-value",
                   spectrum_match.q_value_set.glycan_q_value, "MS:XXX10F"),
            mparam("GlycReSoft:glycopeptide q-value",
                   spectrum_match.q_value_set.glycopeptide_q_value, "MS:XXX10D"),
        ]
        return score_params

    def convert_to_identification_item_dict(self, spectrum_match: serialize.GlycopeptideSpectrumMatch,
                                            seen_targets: Optional[Set[int]] = None,
                                            id_tracker: Optional[SequenceIdTracker] = None) -> Props:
        if seen_targets is None:
            seen_targets = set()
        if spectrum_match.target.id not in seen_targets:
            return None
        charge = spectrum_match.scan.precursor_information.charge
        data = {
            "charge_state": charge,
            "experimental_mass_to_charge": mass_charge_ratio(
                spectrum_match.scan.precursor_information.neutral_mass, charge),
            "calculated_mass_to_charge": mass_charge_ratio(
                spectrum_match.target.total_mass, charge),
            "peptide_id": id_tracker(spectrum_match.target),
            "peptide_evidence_id": spectrum_match.target.id,
            "score": mparam({
                "name": "GlycReSoft:total score",
                "value": spectrum_match.score,
                "accession": "MS:XXX10A",
            }),
            "params": [
                components.CVParam(**{
                    "name": "glycan dissociating, peptide preserving",
                    "accession": "MS:XXX111", "cvRef": "PSI-MS"}),
                components.CVParam(**{
                    "name": "glycan eliminated, peptide dissociating",
                    "accession": "MS:XXX114", "cvRef": "PSI-MS"}),
                {
                    "name": "scan start time",
                    "value": spectrum_match.scan.scan_time,
                    "unit_name": "minute"
                }
            ],
            "id": spectrum_match.id
        }
        if spectrum_match.is_multiscore():
            score_params = self._encode_score_set(spectrum_match)
            data['params'].extend(score_params)
        else:
            data['params'].extend([
                mparam("GlycReSoft:glycopeptide q-value",
                    spectrum_match.q_value, "MS:XXX10D"),
            ])
        if spectrum_match.mass_shift.name != Unmodified.name:
            data['params'].append(
                mparam("GlycReSoft:mass shift", "%s:%0.3f:%0.3f" % (
                    spectrum_match.mass_shift.name,
                    spectrum_match.mass_shift.mass,
                    spectrum_match.mass_shift.tandem_mass),
                    "MS:XXX10I"))
        return data

    def convert_to_spectrum_identification_dict(self, spectrum_solution_set: serialize.GlycopeptideSpectrumSolutionSet,
                                                seen_targets: Optional[Set[int]] = None,
                                                id_tracker: Optional[SequenceIdTracker] = None) -> Props:
        data = {
            "spectra_data_id": 1,
            "spectrum_id": spectrum_solution_set.scan.scan_id,
            "id": spectrum_solution_set.id
        }
        idents = []
        for item in spectrum_solution_set:
            d = self.convert_to_identification_item_dict(
                item, seen_targets=seen_targets, id_tracker=id_tracker)
            if d is None:
                continue
            idents.append(d)
        data['identifications'] = idents
        return data

    def convert_to_peptide_evidence_dict(self, glycopeptide: GlycopeptideType, id_tracker: SequenceIdTracker) -> Props:
        data = {
            "start_position": glycopeptide.protein_relation.start_position,
            "end_position": glycopeptide.protein_relation.end_position,
            "peptide_id": id_tracker(glycopeptide),
            "db_sequence_id": glycopeptide.protein_relation.protein_id,
            "is_decoy": False,
            "id": glycopeptide.id
        }
        return data

    def convert_to_protein_dict(self, protein: serialize.Protein, include_sequence: bool = True) -> Props:
        data = {
            "id": protein.id,
            "accession": protein.name,
            "search_database_id": 1,
        }
        if include_sequence:
            data["sequence"] = protein.protein_sequence
        return data

    def extract_peptides(self):
        self.log("Extracting Proteins")
        self.extract_proteins()
        self._peptides = []
        seen = set()

        self.log("Extracting Peptides")
        for gp in self.glycopeptide_list:
            d = self.convert_to_peptide_dict(gp.structure, self._id_tracker)

            if self._id_tracker(gp.structure) == gp.structure.id:
                self._peptides.append(d)
                seen.add(gp.structure.id)

        self.log("Extracting PeptideEvidence")
        self._peptide_evidence = [
            self.convert_to_peptide_evidence_dict(
                gp.structure, self._id_tracker) for gp in self.glycopeptide_list
        ]

        self._proteins = [
            self.convert_to_protein_dict(prot, self.embed_protein_sequences)
            for prot in self.protein_list
        ]

    def extract_spectrum_identifications(self):
        self.log("Extracting SpectrumIdentificationResults")
        spectrum_identifications = []
        seen_scans = set()
        accepted_solution_ids = {gp.structure.id for gp in self.glycopeptide_list}
        gp_list = self.glycopeptide_list
        n = len(gp_list)
        j = 0
        for i, gp in enumerate(gp_list):
            if i % 25 == 0:
                self.log("... Processed %d glycopeptide features with %d spectra matched (%0.2f%%)" % (i, j, i * 100.0 / n))
            if self.report_top_match_per_glycopeptide:
                spectrum_matches: List[serialize.GlycopeptideSpectrumSolutionSet] = [
                    gp.best_spectrum_match.solution_set]
            else:
                spectrum_matches: List[serialize.GlycopeptideSpectrumSolutionSet] = gp.spectrum_matches
            for solution in spectrum_matches:
                j += 1
                if solution.scan.scan_id in seen_scans:
                    continue
                if solution.best_solution().q_value > self.q_value_threshold:
                    continue
                if solution.score < self.ms2_score_threshold:
                    continue
                seen_scans.add(solution.scan.scan_id)
                d = self.convert_to_spectrum_identification_dict(
                    solution, seen_targets=accepted_solution_ids,
                    id_tracker=self._id_tracker)
                if len(d['identifications']):
                    spectrum_identifications.append(d)

        self.scan_ids = seen_scans
        self._spectrum_identification_list = {
            "id": 1,
            "identification_results": spectrum_identifications
        }

    def software_entry(self) -> List[Props]:
        software = {
            "name": "GlycReSoft",
            "version": version.version,
            "uri": None
        }
        return [software]

    def search_database(self) -> Props:
        hypothesis = self.analysis.hypothesis
        spec = {
            "name": hypothesis.name,
            "location": self.database_handle._original_connection,
            "id": 1
        }
        if "fasta_file" in hypothesis.parameters:
            spec['file_format'] = 'fasta format'
            spec['location'] = hypothesis.parameters['fasta_file']
        elif "mzid_file" in hypothesis.parameters:
            spec['file_format'] = 'mzIdentML format'
        return spec

    def source_file(self) -> Props:
        spec = {
            "location": self.database_handle._original_connection,
            "file_format": "data stored in database",
            "id": 1
        }
        return spec

    def spectra_data(self) -> Props:
        spec = {
            "location": self.analysis.parameters['sample_path'],
            "file_format": 'mzML format',
            "spectrum_id_format": "multiple peak list nativeID format",
            "id": 1
        }
        return spec

    def protocol(self) -> Props:
        hypothesis = self.analysis.hypothesis
        analysis = self.analysis
        mods = []

        def transform_modification(mod):
            if isinstance(mod, str):
                mod_inst = modification.Modification(mod)
                target = modification.extract_targets_from_rule_string(mod)
                new_rule = mod_inst.rule.clone({target})
                return new_rule
            return mod

        def pack_modification(mod, fixed=True):
            mod_spec = {
                "fixed": fixed,
                "mass_delta": mod.mass,
                "residues": [res.symbol for rule in mod.targets
                             for res in rule.amino_acid_targets],
                "params": [
                    mod.name
                ]
            }
            return mod_spec

        def pack_glycan_modification(glycan_composition, fixed=False):
            params = self.gnome_resolver.glycan_composition_to_terms(self._coerce_orm(glycan_composition))
            glycan_types = sorted({str(b) for a in glycan_composition.component_classes for b in a})
            residues = []
            for gt in glycan_types:
                gt = str(gt)
                if gt == "N-Glycan":
                    residues.append("N")
                if gt == "O-Glycan":
                    residues.extend("ST")
                if gt == 'GAG-Linker':
                    residues.append("S")
                params.append(glycosylation_type_to_term(gt))
            if glycan_composition.count == 1:
                params.append({
                    "name": "glycan composition",
                    "cvRef": "PSI-MS",
                    "accession": "MS:XXXX14"
                })
            else:
                params.append({
                    "name": "glycan aggregate",
                    "cvRef": "PSI-MS",
                    "accession": "MS:XXXX15"
                })
            mod_spec = {
                "fixed": fixed,
                "mass_delta": glycan_composition.dehydrated_mass(),
                "name": "glycosylation modification",
                # "accession": "MS:XXXXX1",
                "residues": residues,
                "params": [components.CVParam(**x) for x in params]
            }
            return mod_spec


        for mod in hypothesis.parameters.get('constant_modifications', []):
            mod = transform_modification(mod)
            mods.append(pack_modification(mod, True))
        for mod in hypothesis.parameters.get('variable_modifications', []):
            mod = transform_modification(mod)
            mods.append(pack_modification(mod, False))

        for gc in sorted(self.glycan_list, key=lambda x: (x.mass(), x.id)):
            mods.append(pack_glycan_modification(gc, False))

        strategy = analysis.parameters.get("search_strategy")
        if strategy == "multipart-target-decoy-competition":
            fdr_params = [
                {"name": "glycopeptide false discovery rate control strategy",
                 "accession": "MS:XXX108", "cvRef": "PSI-MS"},
                {"name": "peptide glycopeptide false discovery rate control strategy",
                 "accession": "MS:XXX106", "cvRef": "PSI-MS"},
                {"name": "glycan glycopeptide false discovery rate control strategy",
                 "accession": "MS:XXX107", "cvRef": "PSI-MS"},
                {"name": "joint glycopeptide false discovery rate control strategy",
                 "accession": "MS:XXX11A", "cvRef": "PSI-MS"},
            ]
        else:
            fdr_params = [
                {"name": "glycopeptide false discovery rate control strategy",
                 "accession": "MS:XXX108", "cvRef": "PSI-MS"},
            ]
        spec = {
            "enzymes": [
                {"name": getattr(e, 'name', e), "missed_cleavages": hypothesis.parameters.get(
                    'max_missed_cleavages', None), "id": i}
                for i, e in enumerate(hypothesis.parameters.get('enzymes'))
            ],
            "fragment_tolerance": (analysis.parameters['fragment_error_tolerance'] * 1e6, None, "parts per million"),
            "parent_tolerance": (analysis.parameters['mass_error_tolerance'] * 1e6, None, "parts per million"),
            "modification_params": mods,
            "id": 1,
            "additional_search_params": [
                {
                    "name": "glycopeptide search",
                    "accession": "MS:XXXX98",
                    "cvRef": "PSI-MS",
                }
            ] + fdr_params + [
                {
                    "name": "param: b ion",
                    "accession": "MS:1001118",
                    "cvRef": "PSI-MS",
                },
                {
                    "name": "param: y ion",
                    "accession": "MS:1001262",
                    "cvRef": "PSI-MS",
                },
                {
                    "name": "param: peptide + glycan Y ion",
                    "accession": "MS:XXXX17",
                    "cvRef": "PSI-MS",
                },
                {
                    "name": "param: oxonium ion",
                    "accession": "MS:XXXX22",
                    "cvRef": "PSI-MS",
                },
            ]
        }
        spec['additional_search_params'] = [components.CVParam(**x) for x in spec['additional_search_params']]
        return spec

    def run(self):
        f = MzIdentMLWriter(self.outfile, vocabularies=[
            components.CV(
                id='GNO', uri="http://purl.obolibrary.org/obo/gno.obo", full_name='GNO'),
        ])
        self.log("Loading Spectra Data")
        spectra_data = self.spectra_data()
        self.log("Loading Search Database")
        search_database = self.search_database()
        self.log("Building Protocol")
        self.extract_glycans()
        protocol = self.protocol()
        source_file = self.source_file()
        self.extract_peptides()
        self.extract_spectrum_identifications()

        had_specified_mzml_path = self.source_mzml_path is None
        if self.source_mzml_path is None:
            self.source_mzml_path = spectra_data['location']

        if self.source_mzml_path is None:
            did_resolve_mzml_path = False
        else:
            did_resolve_mzml_path = os.path.exists(self.source_mzml_path)
        if not did_resolve_mzml_path:
            self.log("Could not locate source mzML file.")
            if not had_specified_mzml_path:
                self.log("If you did not specify an alternative location to "
                         "find the mzML path, please do so.")

        if self.export_mzml and did_resolve_mzml_path:
            if self.output_mzml_path is None:
                prefix = os.path.splitext(self.outfile.name)[0]
                self.output_mzml_path = "%s.export.mzML" % (prefix,)
            exporter = None
            self.log("Begin Exporting mzML")
            with open(self.output_mzml_path, 'wb') as handle:
                exporter = MzMLExporter(self.source_mzml_path, handle)
                self.log("... Aggregating Scan Bunches")
                scan_bunches = exporter.aggregate_scan_bunches(self.scan_ids)
                self.log("... Exporting Spectra")
                exporter.begin(scan_bunches)
                self.log("... Exporting Chromatograms")
                exporter.extract_chromatograms_from_identified_glycopeptides(
                    self.glycopeptide_list)
                self.log("... Finalizing mzML")
                exporter.complete()
            self.log("mzML Export Finished")

        analysis = [[spectra_data['id']], [search_database['id']]]

        with f:
            f.controlled_vocabularies()
            f.providence(software=self.software_entry())

            f.register("SpectraData", spectra_data['id'])
            f.register("SearchDatabase", search_database['id'])
            f.register("SpectrumIdentificationList", self._spectrum_identification_list['id'])

            with f.sequence_collection():
                for prot in self._proteins:
                    f.write_db_sequence(**prot)
                for pep in self._peptides:
                    f.write_peptide(**pep)
                for pe in self._peptide_evidence:
                    f.write_peptide_evidence(**pe)

            with f.analysis_protocol_collection():
                f.spectrum_identification_protocol(**protocol)

            with f.element("AnalysisCollection"):
                f.SpectrumIdentification(*analysis).write(f)

            with f.element("DataCollection"):
                f.inputs(source_file, search_database, spectra_data)
                with f.element("AnalysisData"):
                    with f.spectrum_identification_list(id=self._spectrum_identification_list['id']):
                        for result_ in self._spectrum_identification_list['identification_results']:
                            result = dict(result_)
                            identifications = result.pop("identifications")
                            result = f.spectrum_identification_result(**result)
                            with result:
                                for item in identifications:
                                    f.write_spectrum_identification_item(**item)

        f.outfile.close()
