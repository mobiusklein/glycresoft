import csv

from .task import TaskBase


class CSVSerializerBase(TaskBase):
    def __init__(self, outstream, entities_iterable):
        self.outstream = outstream
        self.writer = csv.writer(self.outstream)
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
        self.writer.writerow(self.header)
        gen = (self.convert_object(entity) for entity in self._entities_iterable)
        self.writerows(gen)


class GlycanLCMSAnalysisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable):
        super(GlycanLCMSAnalysisCSVSerializer, self).__init__(outstream, entities_iterable)

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
    def __init__(self, outstream, entities_iterable, protein_name_resolver):
        super(GlycopeptideLCMSMSAnalysisCSVSerializer, self).__init__(outstream, entities_iterable)
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
            "protein_name",
        ]

    def convert_object(self, obj):
        attribs = [
            str(obj.structure),
            obj.chromatogram.weighted_neutral_mass,
            ((obj.structure.total_mass - obj.chromatogram.weighted_neutral_mass
              ) / obj.chromatogram.weighted_neutral_mass),
            obj.ms1_score,
            obj.ms2_score,
            obj.q_value,
            obj.total_signal,
            obj.chromatogram.start_time,
            obj.chromatogram.end_time,
            obj.chromatogram.apex_time,
            ";".join(map(str, obj.chromatogram.charge_states)),
            self.protein_name_resolver[obj.protein_relation.protein_id]
        ]
        return map(str, attribs)
