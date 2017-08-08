import os
from collections import defaultdict, OrderedDict

from glycan_profiling import task, serialize, version

from glypy.composition import formula

from glycopeptidepy.structure import parser, modification
from glycan_profiling.chromatogram_tree.chromatogram import group_by

from brainpy import mass_charge_ratio

from psims.mzid import components
from psims.mzid.writer import MzIdentMLWriter

from ms_deisotope.output import mzml


def convert_to_protein_dict(protein):
    data = {
        "id": protein.id,
        "accession": protein.name,
        "search_database_id": 1,
        "sequence": protein.protein_sequence
    }
    return data


def convert_to_peptide_dict(glycopeptide, id_tracker):
    data = {
        "id": glycopeptide.id,
        "peptide_sequence": parser.strip_modifications(glycopeptide),
        "modifications": [

        ]
    }

    i = 0
    # TODO: handle N-terminal and C-terminal modifications
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
            mod_dict = {
                "monoisotopic_mass_delta": mod.mass,
                "location": i,
                "name": mod.name,
            }
            data['modifications'].append(mod_dict)
    return data


def convert_to_peptide_evidence_dict(glycopeptide, id_tracker):
    data = {
        "start_position": glycopeptide.protein_relation.start_position,
        "end_position": glycopeptide.protein_relation.end_position,
        "peptide_id": id_tracker(glycopeptide),
        "db_sequence_id": glycopeptide.protein_relation.protein_id,
        "is_decoy": False,
        "id": glycopeptide.id
    }
    return data


def convert_to_identification_item_dict(spectrum_match, seen=None, id_tracker=None):
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
        "peptide_id": id_tracker(spectrum_match.target),
        "peptide_evidence_id": spectrum_match.target.id,
        "score": {"name": "GlycReSoft:score", "value": spectrum_match.score},
        "params": [
            {"name": "GlycReSoft:q-value", "value": spectrum_match.q_value},
        ],
        "id": spectrum_match.id
    }
    return data


def convert_to_spectrum_identification_dict(spectrum_solution_set, seen=None, id_tracker=None):
    data = {
        "spectra_data_id": 1,
        "spectrum_id": spectrum_solution_set.scan.id,
        "id": spectrum_solution_set.id
    }
    idents = []
    for item in spectrum_solution_set:
        d = convert_to_identification_item_dict(item, seen=seen, id_tracker=id_tracker)
        if d is None:
            continue
        idents.append(d)
    data['identifications'] = idents
    return data


class MzMLExporter(task.TaskBase):
    def __init__(self, source, outfile):
        self.reader = mzml.ProcessedMzMLDeserializer(source)
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


class SequenceIdTracker(object):
    def __init__(self):
        self.mapping = dict()

    def convert(self, glycopeptide):
        s = str(glycopeptide)
        if s in self.mapping:
            return self.mapping[s]
        else:
            self.mapping[s] = glycopeptide.id
            return self.mapping[s]

    def __call__(self, glycopeptide):
        return self.convert(glycopeptide)

    def dump(self):
        for key, value in self.mapping.items():
            print(value, key)


class MzIdentMLSerializer(task.TaskBase):
    def __init__(self, outfile, glycopeptide_list, analysis, database_handle,
                 q_value_threshold=0.05, ms2_score_threshold=0,
                 export_mzml=True, source_mzml_path=None,
                 output_mzml_path=None):
        self.outfile = outfile
        self.database_handle = database_handle
        self._glycopeptide_list = glycopeptide_list
        self.protein_list = None
        self.analysis = analysis
        self.scan_ids = set()
        self._id_tracker = SequenceIdTracker()
        self.q_value_threshold = q_value_threshold
        self.ms2_score_threshold = ms2_score_threshold
        self.export_mzml = export_mzml
        self.source_mzml_path = source_mzml_path
        self.output_mzml_path = output_mzml_path

    @property
    def glycopeptide_list(self):
        return self._glycopeptide_list

    def extract_proteins(self):
        self.protein_list = [self.database_handle.query(
            serialize.Protein).get(i) for i in
            {gp.protein_relation.protein_id for gp in self.glycopeptide_list}]

    def extract_peptides(self):
        self.log("Extracting Proteins")
        self.extract_proteins()
        self._peptides = []
        seen = set()

        self.log("Extracting Peptides")
        for gp in self.glycopeptide_list:
            d = convert_to_peptide_dict(gp.structure, self._id_tracker)

            if self._id_tracker(gp.structure) == gp.structure.id:
                self._peptides.append(d)
                seen.add(gp.structure.id)

        self.log("Extracting PeptideEvidence")
        self._peptide_evidence = [
            convert_to_peptide_evidence_dict(
                gp.structure, self._id_tracker) for gp in self.glycopeptide_list
        ]

        self._proteins = [convert_to_protein_dict(prot) for prot in self.protein_list]

    def extract_spectrum_identifications(self):
        self.log("Extracting SpectrumIdentificationResults")
        spectrum_identifications = []
        seen_scans = set()
        accepted_solution_ids = {gp.structure.id for gp in self.glycopeptide_list}
        for gp in self.glycopeptide_list:
            for solution in gp.spectrum_matches:
                if solution.scan.scan_id in seen_scans:
                    continue
                if solution.best_solution().q_value > self.q_value_threshold:
                    continue
                if solution.score < self.ms2_score_threshold:
                    continue
                seen_scans.add(solution.scan.scan_id)
                d = convert_to_spectrum_identification_dict(
                    solution, seen=accepted_solution_ids,
                    id_tracker=self._id_tracker)
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
        elif "mzid_file" in hypothesis.parameters:
            spec['file_format'] = 'mzIdentML format'
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

        def transform_modification(mod):
            if isinstance(mod, basestring):
                mod_inst = modification.Modification(mod)
                target = modification.extract_targets_from_string(mod)
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

        for mod in hypothesis.parameters.get('constant_modifications', []):
            mod = transform_modification(mod)
            mods.append(pack_modification(mod, True))
        for mod in hypothesis.parameters.get('variable_modifications', []):
            mod = transform_modification(mod)
            mods.append(pack_modification(mod, False))
        spec = {
            "enzymes": [
                {"name": getattr(e, 'name', e), "missed_cleavages": hypothesis.parameters.get(
                    'max_missed_cleavages', None)}
                for e in hypothesis.parameters.get('enzymes')
            ],
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
