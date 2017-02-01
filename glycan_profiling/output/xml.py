from glycan_profiling import task, serialize, version

from glypy.composition import formula

from glycopeptidepy.structure import parser

from brainpy import mass_charge_ratio

from psims.mzid import components
from psims.mzid.writer import MzIdentMLWriter


def convert_to_protein_dict(protein):
    data = {
        "id": protein.id,
        "accession": protein.name,
        "search_database_id": 1,
        "sequence": protein.protein_sequence
    }
    return data


def convert_to_peptide_dict(glycopeptide):
    data = {
        "id": glycopeptide.id,
        "peptide_sequence": parser.strip_modifications(glycopeptide),
        "modifications": [

        ]
    }

    i = 0
    for pos, mods in glycopeptide:
        i += 1
        if not mods:
            continue
        else:
            mod = mods[0]
        if mod.rule.is_a("glycosylation"):
            mod_dict = {
                "monoisotopic_mass_delta": glycopeptide.glycan_composition.mass(),
                "location": i,
                "name": "unknown modification",
                "params": [
                    components.UserParam(
                        name='GlycosylationType', value=str(mod)),
                    components.UserParam(name='GlycanComposition', value=str(
                        glycopeptide.glycan_composition)),
                    components.UserParam(name='Formula', value=formula(
                        glycopeptide.glycan_composition.total_composition()))
                ]
            }
            data['modifications'].append(mod_dict)
        else:
            print("Not Implemented", glycopeptide.id, mod)
    return data


def convert_to_peptide_evidence_dict(glycopeptide):
    data = {
        "start_position": glycopeptide.protein_relation.start_position,
        "end_position": glycopeptide.protein_relation.end_position,
        "peptide_id": glycopeptide.id,
        "db_sequence_id": glycopeptide.protein_relation.protein_id,
        "is_decoy": False,
        "id": glycopeptide.id
    }
    return data


def convert_to_identification_item_dict(spectrum_match, seen=None):
    if seen is None:
        seen = set()
    charge = spectrum_match.scan.precursor_information.charge
    if spectrum_match.target.id not in seen:
        return None
    data = {
        "charge_state": charge,
        "experimental_mass_to_charge": mass_charge_ratio(
            spectrum_match.scan.precursor_information.neutral_mass, charge),
        "calculated_mass_to_charge": mass_charge_ratio(
            spectrum_match.target.total_mass, charge),
        "peptide_id": spectrum_match.target.id,
        "peptide_evidence_id": spectrum_match.target.id,
        "score": {"name": "GlycReSoft:score", "value": spectrum_match.score},
        "params": [
            {"name": "GlycReSoft:q-value", "value": spectrum_match.q_value},
        ],
        "id": spectrum_match.id
    }
    return data


def convert_to_spectrum_identification_dict(spectrum_solution_set, seen=None):
    data = {
        "spectra_data_id": 1,
        "spectrum_id": spectrum_solution_set.scan.id,
        "id": spectrum_solution_set.id
    }
    idents = []
    for item in spectrum_solution_set:
        d = convert_to_identification_item_dict(item, seen=seen)
        if d is None:
            continue
        idents.append(d)
    data['identifications'] = idents
    return data


class MzIdentMLSerializer(task.TaskBase):
    def __init__(self, outfile, glycopeptide_list, analysis, database_handle):
        self.outfile = outfile
        self.database_handle = database_handle
        self._glycopeptide_list = glycopeptide_list
        self.protein_list = None
        self.analysis = analysis
        self.scan_ids = set()

    @property
    def glycopeptide_list(self):
        return self._glycopeptide_list

    def extract_proteins(self):
        self.protein_list = [self.database_handle.query(
            serialize.Protein).get(i) for i in
            {gp.protein_relation.protein_id for gp in self.glycopeptide_list}]

    def extract_peptides(self):
        self.extract_proteins()
        self._peptides = []
        seen = set()
        for gp in self.glycopeptide_list:
            d = convert_to_peptide_dict(gp.structure)
            self._peptides.append(d)
            seen.add(gp.structure.id)
        self._peptide_evidence = [
            convert_to_peptide_evidence_dict(gp.structure) for gp in self.glycopeptide_list
        ]

        self._proteins = [convert_to_protein_dict(prot) for prot in self.protein_list]

    def extract_spectrum_identifications(self):
        spectrum_identifications = []
        seen_scans = set()
        accepted_solution_ids = {gp.structure.id for gp in self.glycopeptide_list}
        for gp in self.glycopeptide_list:
            for solution in gp.spectrum_matches:
                if solution.scan.id in seen_scans:
                    continue
                seen_scans.add(solution.scan.id)
                d = convert_to_spectrum_identification_dict(
                    solution, seen=accepted_solution_ids)
                if len(d['identifications']):
                    spectrum_identifications.append(d)
        self.scan_ids = seen_scans
        self._spectrum_identification_list = {
            "id": 1,
            "identification_results": spectrum_identifications
        }

    def software_entry(self):
        software = {
            "name": "GlycReSoft",
            "version": version.version,
            "uri": None
        }
        return [software]

    def search_database(self):
        hypothesis = self.analysis.hypothesis
        spec = {
            "name": hypothesis.name,
            "location": self.database_handle._original_connection,
            "id": 1
        }
        if "fasta_file" in hypothesis.parameters:
            spec['file_format'] = 'fasta format'
            spec['location'] = hypothesis.parameters['fasta_file']
        return spec

    def source_file(self):
        spec = {
            "location": self.database_handle._original_connection,
            "file_format": "data stored in database",
            "id": 1
        }
        return spec

    def spectra_data(self):
        spec = {
            "location": self.analysis.parameters['sample_path'],
            "file_format": 'mzML format',
            "spectrum_id_format": "multiple peak list nativeID format",
            "id": 1
        }
        return spec

    def protocol(self):
        hypothesis = self.analysis.hypothesis
        analysis = self.analysis
        mods = []
        for mod in hypothesis.parameters['constant_modifications']:
            mod_spec = {
                "fixed": True,
                "mass_delta": mod.mass,
                "residues": [res.symbol for rule in mod.targets
                             for res in rule.amino_acid_targets],
                "params": [
                    mod.name
                ]
            }
            mods.append(mod_spec)
        for mod in hypothesis.parameters['variable_modifications']:
            mod_spec = {
                "fixed": False,
                "mass_delta": mod.mass,
                "residues": [res.symbol for rule in mod.targets
                             for res in rule.amino_acid_targets],
                "params": [
                    mod.name
                ]
            }
            mods.append(mod_spec)
        spec = {
            "enzymes": [
                {"name": e, "missed_cleavages": hypothesis.parameters['max_missed_cleavages']}
                for e in hypothesis.parameters['enzymes']],
            "fragment_tolerance": (analysis.parameters['fragment_error_tolerance'] * 1e6, None, "parts per million"),
            "parent_tolerance": (analysis.parameters['mass_error_tolerance'] * 1e6, None, "parts per million"),
            "modification_params": mods,
            "id": 1
        }
        return spec

    def run(self):
        f = MzIdentMLWriter(self.outfile)
        spectra_data = self.spectra_data()
        search_database = self.search_database()
        protocol = self.protocol()
        source_file = self.source_file()
        analysis = [[spectra_data['id']], [search_database['id']]]

        self.extract_peptides()
        self.extract_spectrum_identifications()

        with f:
            f.controlled_vocabularies()
            f.providence(software=self.software_entry())

            f.register("SpectraData", spectra_data['id'])
            f.register("SearchDatabase", search_database['id'])
            f.register("SpectrumIdentificationList", self._spectrum_identification_list['id'])

            f.sequence_collection(self._proteins, self._peptides, self._peptide_evidence)

            with f.analysis_protocol_collection():
                f.spectrum_identification_protocol(**protocol)

            with f.element("AnalysisCollection"):
                f.SpectrumIdentification(*analysis).write(f)

            with f.element("DataCollection"):
                f.inputs(source_file, search_database, spectra_data)
                with f.element("AnalysisData"):
                    f.spectrum_identification_list(**self._spectrum_identification_list)
        f.outfile.close()
        f.format()
