import csv

from glycan_profiling.task import TaskBase


class CSVSerializerBase(TaskBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        self.outstream = outstream
        self.writer = csv.writer(self.outstream, delimiter=delimiter)
        self._entities_iterable = entities_iterable

    def writerows(self, iterable):
        self.writer.writerows(iterable)

    def get_header(self):
        raise NotImplementedError()

    def convert_object(self, obj):
        raise NotImplementedError()

    @property
    def header(self):
        return self.get_header()

    def run(self):
        if self.header:
            self.writer.writerow(self.header)
        gen = (self.convert_object(entity) for entity in self._entities_iterable)
        self.writerows(gen)


class GlycanHypothesisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(GlycanHypothesisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)

    def get_header(self):
        return [
            "composition",
            "calculated_mass",
            "classes"
        ]

    def convert_object(self, obj):
        attribs = [
            obj.composition,
            obj.calculated_mass,
            ";".join([sc.name for sc in obj.structure_classes])
        ]
        return map(str, attribs)


class ImportableGlycanHypothesisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable):
        super(ImportableGlycanHypothesisCSVSerializer, self).__init__(outstream, entities_iterable, "\t")

    def get_header(self):
        return False

    def convert_object(self, obj):
        attribs = [
            obj.composition,
            "\t".join([sc.name for sc in obj.structure_classes])
        ]
        return map(str, attribs)


class GlycopeptideHypothesisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(GlycopeptideHypothesisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)

    def get_header(self):
        return [
            "glycopeptide",
            "calculated_mass",
            "start_position",
            "end_position",
            "protein",
        ]

    def convert_object(self, obj):
        attribs = [
            obj.glycopeptide_sequence,
            obj.calculated_mass,
            obj.peptide.start_position,
            obj.peptide.end_position,
            obj.peptide.protein.name
        ]
        return map(str, attribs)


class GlycanLCMSAnalysisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(GlycanLCMSAnalysisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)

    def get_header(self):
        return [
            "composition",
            "neutral_mass",
            "mass_accuracy",
            "score",
            "total_signal",
            "start_time",
            "end_time",
            "apex_time",
            "charge_states",
            "adducts",
        ]

    def convert_object(self, obj):
        attribs = [
            obj.glycan_composition,
            obj.weighted_neutral_mass,
            0 if obj.glycan_composition is None else (
                (obj.glycan_composition.mass() - obj.weighted_neutral_mass
                 ) / obj.weighted_neutral_mass),
            obj.score,
            obj.total_signal,
            obj.start_time,
            obj.end_time,
            obj.apex_time,
            ';'.join(map(str, obj.charge_states)),
            ';'.join([a.name for a in obj.adducts]),
        ]
        return map(str, attribs)


class GlycopeptideLCMSMSAnalysisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, protein_name_resolver, delimiter=','):
        super(GlycopeptideLCMSMSAnalysisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)
        self.protein_name_resolver = protein_name_resolver

    def get_header(self):
        return [
            "glycopeptide",
            "neutral_mass",
            "mass_accuracy",
            "ms1_score",
            "ms2_score",
            "q_value",
            "total_signal",
            "start_time",
            "end_time",
            "apex_time",
            "charge_states",
            "msms_count",
            "peptide_start",
            "peptide_end",
            "protein_name",
        ]

    def convert_object(self, obj):
        attribs = [
            str(obj.structure),
            obj.weighted_neutral_mass,
            ((obj.structure.total_mass - obj.weighted_neutral_mass
              ) / obj.weighted_neutral_mass),
            obj.ms1_score,
            obj.ms2_score,
            obj.q_value,
            obj.total_signal,
            obj.start_time,
            obj.end_time,
            obj.apex_time,
            ";".join(map(str, obj.charge_states)),
            len(obj.spectrum_matches),
            obj.protein_relation.start_position,
            obj.protein_relation.end_position,
            self.protein_name_resolver[obj.protein_relation.protein_id]
        ]
        return map(str, attribs)


class GlycopeptideSpectrumMatchAnalysisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, protein_name_resolver, delimiter=','):
        super(GlycopeptideSpectrumMatchAnalysisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)
        self.protein_name_resolver = protein_name_resolver

    def get_header(self):
        return [
            "glycopeptide",
            "neutral_mass",
            "mass_accuracy",
            "scan_id",
            "charge",
            "ms2_score",
            "q_value",
            "precursor_abundance",
            "peptide_start",
            "peptide_end",
            "protein_name",
        ]

    def convert_object(self, obj):
        precursor_mass = obj.scan.precursor_information.extracted_neutral_mass
        attribs = [
            str(obj.target),
            precursor_mass,
            ((obj.target.total_mass - precursor_mass
              ) / precursor_mass),
            obj.scan.id,
            obj.scan.precursor_information.extracted_charge,
            obj.score,
            obj.q_value,
            obj.scan.precursor_information.extracted_intensity,
            obj.target.protein_relation.start_position,
            obj.target.protein_relation.end_position,
            self.protein_name_resolver[obj.target.protein_relation.protein_id]
        ]
        return map(str, attribs)

    def filter(self, iterable):
        seen = set()
        for row in iterable:
            key = (row[0], row[2])
            if key in seen:
                continue
            seen.add(key)
            yield row

    def run(self):
        self.writer.writerow(self.header)
        gen = (self.convert_object(entity) for entity in self._entities_iterable)
        self.writerows(self.filter(gen))
