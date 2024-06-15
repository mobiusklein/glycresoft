import csv
import logging

from io import TextIOWrapper
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, TextIO

from six import PY2

from glycresoft.task import TaskBase

from glycresoft.scoring.elution_time_grouping import (
    GlycopeptideChromatogramProxy
)

if TYPE_CHECKING:
    from glycresoft.serialize import Analysis

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
    entity_label: str = "entities"
    log_interval: int = 100
    outstream: TextIO
    writer: csv.writer
    delimiter: str
    _entities_iterable: Iterable

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

    def status_update(self, message: str):
        status_logger.info(message)

    def writerows(self, iterable: Iterable[Iterable[str]]):
        self.writer.writerows(iterable)

    def writerow(self, row: Iterable[str]):
        self.writer.writerow(row)

    def get_header(self) -> List[str]:
        raise NotImplementedError()

    def convert_object(self, obj):
        raise NotImplementedError()

    @property
    def header(self):
        return self.get_header()

    def run(self):
        if self.header:
            header = self.header
            self.writerow(header)
        entity_label = self.entity_label
        log_interval = self.log_interval
        gen = (self.convert_object(entity) for entity in self._entities_iterable)
        i = 0
        for i, row in enumerate(gen, 1):
            if i % log_interval == 0 and i != 0:
                self.status_update(f"Handled {i} {entity_label}")
            self.writerow(row)
        self.status_update(f"Handled {i} {entity_label}")
        self.outstream.flush() # If the TextIOWrapper is buffering, we need to flush it here


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
            outstream, entities_iterable, delimiter="\t")

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
    analysis: 'Analysis'
    protein_name_resolver: Mapping[int, str]
    include_group: bool

    def __init__(self, outstream, entities_iterable, protein_name_resolver, analysis, delimiter=',',
                 include_group: bool=True):
        super(GlycopeptideLCMSMSAnalysisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)
        self.protein_name_resolver = protein_name_resolver
        self.analysis = analysis
        self.retention_time_model = analysis.parameters.get("retention_time_model")
        self.include_group = include_group

    def get_header(self):
        headers = [
            "glycopeptide",
            "analysis",
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
            "n_glycosylation_sites",
            "mass_shifts",
        ]
        if self.include_group:
            headers.append("group_id")
        if self.retention_time_model:
            headers.extend([
                "predicted_apex_interval_start",
                "predicted_apex_interval_end",
                "retention_time_score",
            ])
        return headers

    def convert_object(self, obj):
        has_chrom = obj.has_chromatogram()
        if has_chrom:
            weighted_neutral_mass = obj.weighted_neutral_mass
            charge_states = obj.charge_states
        else:
            weighted_neutral_mass = obj.tandem_solutions[0].scan.precursor_information.neutral_mass
            charge_states = (obj.tandem_solutions[0].scan.precursor_information.charge,)

        attribs = [
            str(obj.structure),
            self.analysis.name,
            weighted_neutral_mass,
            (obj.structure.total_mass - weighted_neutral_mass) / weighted_neutral_mass,
            obj.ms1_score,
            obj.ms2_score,
            obj.q_value,
            obj.total_signal,
            obj.start_time if has_chrom else '',
            obj.end_time if has_chrom else '',
            obj.apex_time if has_chrom else '',
            ";".join(map(str, charge_states)),
            len(obj.spectrum_matches),
            obj.protein_relation.start_position,
            obj.protein_relation.end_position,
            self.protein_name_resolver[obj.protein_relation.protein_id],
            ';'.join([
                str(i + obj.start_position) for i in obj.structure.n_glycan_sequon_sites
            ]),
            ';'.join([a.name for a in obj.mass_shifts]),
        ]
        try:
            attribs.append(obj.ambiguous_id)
        except AttributeError:
            attribs.append(-1)
        if self.retention_time_model:
            if has_chrom:
                proxy = GlycopeptideChromatogramProxy.from_chromatogram(obj)
                rt_start, rt_end = self.retention_time_model.predict_interval(proxy, 0.01)
                rt_score = self.retention_time_model.score_interval(proxy, 0.01)
                attribs.append(rt_start)
                attribs.append(rt_end)
                attribs.append(rt_score)
            else:
                attribs.append("-")
                attribs.append("-")
                attribs.append("-")
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
            "localizations"
        ])
        return headers

    def convert_object(self, obj):
        fields = super(MultiScoreGlycopeptideLCMSMSAnalysisCSVSerializer, self).convert_object(obj)
        score_set = obj.score_set
        q_value_set = obj.q_value_set
        new_fields = [
            score_set.glycopeptide_score if score_set is not None else "?",
            score_set.peptide_score if score_set is not None else "?",
            score_set.glycan_score if score_set is not None else "?",
            score_set.glycan_coverage if score_set is not None else "?",
            q_value_set.total_q_value if q_value_set is not None else "?",
            q_value_set.peptide_q_value if q_value_set is not None else "?",
            q_value_set.glycan_q_value if q_value_set is not None else "?",
            q_value_set.glycopeptide_q_value if q_value_set is not None else "?",
            ';'.join(map(str, obj.localizations))
        ]
        fields.extend(map(str, new_fields))
        return fields


class GlycopeptideSpectrumMatchAnalysisCSVSerializer(CSVSerializerBase):
    entity_label = "glycopeptide spectrum matches"
    log_interval = 5000

    include_rank: bool
    include_group: bool
    analysis: 'Analysis'
    protein_name_resolver: Dict[int, str]

    def __init__(self, outstream, entities_iterable, protein_name_resolver, analysis, delimiter=',',
                 include_rank: bool = True, include_group: bool = True):
        super(GlycopeptideSpectrumMatchAnalysisCSVSerializer, self).__init__(outstream, entities_iterable, delimiter)
        self.protein_name_resolver = protein_name_resolver
        self.analysis = analysis
        self.include_rank = include_rank
        self.include_group = include_group

    def get_header(self):
        names = [
            "glycopeptide",
            "analysis",
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
            "n_glycosylation_sites",
            "is_best_match",
            "is_precursor_fit",
        ]
        if self.include_rank:
            names.append("rank")
        if self.include_group:
            names.append("group_id")
        return names

    def convert_object(self, obj):
        try:
            mass_shift = obj.mass_shift
            mass_shift_mass = mass_shift.mass
            mass_shift_name = mass_shift.name
        except Exception:
            mass_shift_name = "Unmodified"
            mass_shift_mass = 0
        target = obj.target
        scan = obj.scan
        precursor = scan.precursor_information
        precursor_mass = precursor.neutral_mass
        attribs = [
            str(obj.target),
            self.analysis.name,
            precursor_mass,
            ((target.total_mass + mass_shift_mass) -
             precursor_mass) / precursor_mass,
            mass_shift_name,
            scan.scan_id,
            scan.scan_time,
            scan.precursor_information.extracted_charge,
            obj.score,
            obj.q_value,
            scan.precursor_information.intensity,
            target.protein_relation.start_position,
            target.protein_relation.end_position,
            self.protein_name_resolver[target.protein_relation.protein_id],
            ';'.join([
                str(i + obj.target.protein_relation.start_position)
                for i in obj.target.n_glycan_sequon_sites
            ]),
            obj.is_best_match,
            not precursor.defaulted
        ]
        if self.include_rank:
            try:
                rank = obj.rank
            except AttributeError:
                rank = -1
            attribs.append(rank)
        if self.include_group:
            try:
                group = obj.cluster_id
            except AttributeError:
                group = -1
            attribs.append(group)
        return list(map(str, attribs))


class MultiScoreGlycopeptideSpectrumMatchAnalysisCSVSerializer(GlycopeptideSpectrumMatchAnalysisCSVSerializer):

    def get_header(self):
        header = super(MultiScoreGlycopeptideSpectrumMatchAnalysisCSVSerializer, self).get_header()
        header.extend([
            "peptide_score",
            "glycan_score",
            "glycan_coverage",
            "peptide_q_value",
            "glycan_q_value",
            "glycopeptide_q_value",
            "localizations",
        ])
        return header

    def convert_object(self, obj):
        fields = super(MultiScoreGlycopeptideSpectrumMatchAnalysisCSVSerializer, self).convert_object(obj)
        score_set = obj.score_set
        fdr_set = obj.q_value_set
        fields.extend([
            score_set.peptide_score,
            score_set.glycan_score,
            score_set.glycan_coverage,
            fdr_set.peptide_q_value,
            fdr_set.glycan_q_value,
            fdr_set.glycopeptide_q_value,
            ';'.join(map(str, obj.localizations))
        ])
        return fields
