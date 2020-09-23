import csv
import logging

from io import TextIOWrapper

from six import PY2

from glycan_profiling.task import TaskBase


status_logger = logging.getLogger("glycresoft.status")


def csv_stream(outstream):
    if 'b' in outstream.mode:
        if not PY2:
            return TextIOWrapper(outstream, 'utf8', newline="")
        else:
            return outstream
    else:
        import warnings
        warnings.warn("Opened CSV stream in text mode!")
        return outstream


class CSVSerializerBase(TaskBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        self.outstream = outstream
        try:
            self.is_binary = 'b' in self.outstream.mode
        except AttributeError:
            self.is_binary = True
        if self.is_binary:
            try:
                self.outstream = TextIOWrapper(outstream, 'utf8', newline="")
            except AttributeError:
                # must be Py2
                pass
        self.writer = csv.writer(self.outstream, delimiter=delimiter)
        self._entities_iterable = entities_iterable

    def status_update(self, message):
        status_logger.info(message)

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
            header = self.header
            self.writer.writerow(header)
        gen = (self.convert_object(entity) for entity in self._entities_iterable)
        for i, row in enumerate(gen):
            if i % 100 == 0 and i != 0:
                self.status_update("Handled %d Entities" % i)
                self.outstream.flush()
            self.writer.writerow(row)


class GlycanHypothesisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(GlycanHypothesisCSVSerializer, self).__init__(
            outstream, entities_iterable, delimiter)

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
        return list(map(str, attribs))


class ImportableGlycanHypothesisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable):
        super(ImportableGlycanHypothesisCSVSerializer, self).__init__(
            outstream, entities_iterable, "\t")

    def get_header(self):
        return False

    def convert_object(self, obj):
        attribs = [
            obj.composition,
        # Use list concatenation to ensure that the glycan classes
        # are treated as independent entries not containing whitespace
        # so that they are not quoted
        ] + [sc.name for sc in obj.structure_classes]
        return list(map(str, attribs))


class GlycopeptideHypothesisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(GlycopeptideHypothesisCSVSerializer, self).__init__(
            outstream, entities_iterable, delimiter)

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
        return list(map(str, attribs))


class SimpleChromatogramCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(SimpleChromatogramCSVSerializer, self).__init__(
            outstream, entities_iterable, delimiter)

    def get_header(self):
        return [
            "neutral_mass",
            "total_signal",
            "charge_states",
            "start_time",
            "apex_time",
            "end_time"
        ]

    def convert_object(self, obj):
        attribs = [
            obj.weighted_neutral_mass,
            obj.total_signal,
            ';'.join(map(str, obj.charge_states)),
            obj.start_time,
            obj.apex_time,
            obj.end_time,
        ]
        return map(str, attribs)


class SimpleScoredChromatogramCSVSerializer(SimpleChromatogramCSVSerializer):

    def get_header(self):
        headers = super(SimpleScoredChromatogramCSVSerializer, self).get_header()
        headers += [
            "score",
            "line_score",
            "isotopic_fit",
            "spacing_fit",
            "charge_count",
        ]
        return headers

    def convert_object(self, obj):
        attribs = super(SimpleScoredChromatogramCSVSerializer, self).convert_object(obj)
        scores = obj.score_components()
        more_attribs = [
            obj.score,
            scores.line_score,
            scores.isotopic_fit,
            scores.spacing_fit,
            scores.charge_count,
        ]
        attribs = list(attribs) + list(map(str, more_attribs))
        return attribs


class GlycanLCMSAnalysisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, delimiter=','):
        super(GlycanLCMSAnalysisCSVSerializer, self).__init__(
            outstream, entities_iterable, delimiter)

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
            "mass_shifts",
            "line_score",
            "isotopic_fit",
            "spacing_fit",
            "charge_count",
            "ambiguous_with",
            "used_as_mass_shift"
        ]

    def convert_object(self, obj):
        scores = obj.score_components()
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
            ';'.join([a.name for a in obj.mass_shifts]),
            scores.line_score,
            scores.isotopic_fit,
            scores.spacing_fit,
            scores.charge_count,
            ';'.join("%s:%s" % (p[0], p[1].name) for p in obj.ambiguous_with),
            ';'.join("%s:%s" % (p[0], p[1].name) for p in obj.used_as_mass_shift)
        ]
        return list(map(str, attribs))


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
            "mass_shifts",
        ]

    def convert_object(self, obj):
        try:
            weighted_neutral_mass = obj.weighted_neutral_mass
        except Exception:
            weighted_neutral_mass = obj.tandem_solutions[0].scan.precursor_information.neutral_mass
        try:
            charge_states = obj.charge_states
        except Exception:
            charge_states = (obj.tandem_solutions[0].scan.precursor_information.charge,)
        attribs = [
            str(obj.structure),
            weighted_neutral_mass,
            (obj.structure.total_mass - weighted_neutral_mass) / weighted_neutral_mass,
            obj.ms1_score,
            obj.ms2_score,
            obj.q_value,
            obj.total_signal,
            obj.start_time if obj.chromatogram else '',
            obj.end_time if obj.chromatogram else '',
            obj.apex_time if obj.chromatogram else '',
            ";".join(map(str, charge_states)),
            len(obj.spectrum_matches),
            obj.protein_relation.start_position,
            obj.protein_relation.end_position,
            self.protein_name_resolver[obj.protein_relation.protein_id],
            ';'.join([a.name for a in obj.mass_shifts]),
        ]
        return list(map(str, attribs))


class MultiScoreGlycopeptideLCMSMSAnalysisCSVSerializer(GlycopeptideLCMSMSAnalysisCSVSerializer):
    def get_header(self):
        headers = super(MultiScoreGlycopeptideLCMSMSAnalysisCSVSerializer, self).get_header()
        headers.extend([
            "glycopeptide_score",
            "peptide_score",
            "glycan_score",
            "glycan_coverage",
            "total_q_value",
            "peptide_q_value",
            "glycan_q_value",
            "glycopeptide_q_value",
        ])
        return headers

    def convert_object(self, obj):
        fields = super(MultiScoreGlycopeptideLCMSMSAnalysisCSVSerializer, self).convert_object(obj)
        score_set = obj.score_set
        q_value_set = obj.q_value_set
        new_fields = [
            score_set.glycopeptide_score,
            score_set.peptide_score,
            score_set.glycan_score,
            score_set.glycan_coverage,
            q_value_set.total_q_value,
            q_value_set.peptide_q_value,
            q_value_set.glycan_q_value,
            q_value_set.glycopeptide_q_value,
        ]
        fields.extend(map(str, new_fields))
        return fields


class GlycopeptideSpectrumMatchAnalysisCSVSerializer(CSVSerializerBase):
    def __init__(self, outstream, entities_iterable, protein_name_resolver, delimiter=','):
        super(GlycopeptideSpectrumMatchAnalysisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)
        self.protein_name_resolver = protein_name_resolver

    def get_header(self):
        return [
            "glycopeptide",
            "neutral_mass",
            "mass_accuracy",
            "mass_shift_name",
            "scan_id",
            "scan_time",
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
        try:
            mass_shift_name = obj.mass_shift.name
        except Exception:
            mass_shift_name = "Unmodified"
        attribs = [
            str(obj.target),
            precursor_mass,
            (obj.target.total_mass - precursor_mass) / precursor_mass,
            mass_shift_name,
            obj.scan.scan_id,
            obj.scan.scan_time,
            obj.scan.precursor_information.extracted_charge,
            obj.score,
            obj.q_value,
            obj.scan.precursor_information.extracted_intensity,
            obj.target.protein_relation.start_position,
            obj.target.protein_relation.end_position,
            self.protein_name_resolver[obj.target.protein_relation.protein_id]
        ]
        return list(map(str, attribs))

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
