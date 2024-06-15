import os
import sys
import time
import logging

import click

from glycresoft.cli.base import cli
from glycresoft.cli.validators import get_by_name_or_id, DatabaseConnectionParam

from glycresoft import serialize
from glycresoft.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    Analysis,
    AnalysisTypeEnum,
    GlycanCompositionChromatogram,
    Protein,
    Glycopeptide,
    IdentifiedGlycopeptide,
    GlycopeptideSpectrumMatch,
    AnalysisDeserializer,
    GlycanComposition,
    GlycanCombination,
    GlycanCombinationGlycanComposition,
    UnidentifiedChromatogram)

from glycresoft.output import (
    GlycanHypothesisCSVSerializer, ImportableGlycanHypothesisCSVSerializer,
    GlycopeptideHypothesisCSVSerializer, GlycanLCMSAnalysisCSVSerializer,
    GlycopeptideLCMSMSAnalysisCSVSerializer,
    GlycopeptideSpectrumMatchAnalysisCSVSerializer,
    MultiScoreGlycopeptideLCMSMSAnalysisCSVSerializer,
    MultiScoreGlycopeptideSpectrumMatchAnalysisCSVSerializer,
    MzIdentMLSerializer,
    GlycanChromatogramReportCreator,
    GlycopeptideDatabaseSearchReportCreator,
    TrainingMGFExporter,
    SpectrumAnnotatorExport,
    CSVSpectrumAnnotatorExport)

from glycresoft.output.csv_format import csv_stream

from glycresoft.cli.utils import ctxstream


status_logger = logging.getLogger('glycresoft.status')


def database_connection_arg(fn):
    arg = click.argument(
        "database-connection",
        type=DatabaseConnectionParam(exists=True),
        doc_help=(
            "A connection URI for a database, or a path on the file system"))
    return arg(fn)


def analysis_identifier_arg(analysis_type):
    def wrapper(fn):
        arg = click.argument("analysis-identifier", doc_help=(
            "The ID number or name of the %s analysis to use" % (analysis_type,)))
        return arg(fn)
    return wrapper


def hypothesis_identifier_arg(hypothesis_type):
    def wrapper(fn):
        arg = click.argument("hypothesis-identifier", doc_help=(
            "The ID number or name of the %s hypothesis to use" % (hypothesis_type,)))
        return arg(fn)
    return wrapper


@cli.group(short_help='Write Data Collections To Text Files')
def export():
    pass


@export.command("glycan-hypothesis",
                short_help="Export theoretical glycan composition database to CSV")
@database_connection_arg
@hypothesis_identifier_arg("glycan")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-i", "--importable", is_flag=True,
              help="Make the file importable for later re-use, with less information.")
def glycan_hypothesis(database_connection, hypothesis_identifier, output_path=None, importable=False):
    """Write each theoretical glycan composition in CSV format"""
    database_connection = DatabaseBoundOperation(database_connection)
    hypothesis = get_by_name_or_id(database_connection, GlycanHypothesis, hypothesis_identifier)
    if importable:
        task_type = ImportableGlycanHypothesisCSVSerializer
    else:
        task_type = GlycanHypothesisCSVSerializer
    if output_path is None or output_path == '-':
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = task_type(output_stream, hypothesis.glycans)
        job.run()


@export.command("glycopeptide-hypothesis",
                short_help='Export theoretical glycopeptide database to CSV')
@database_connection_arg
@hypothesis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def glycopeptide_hypothesis(database_connection, hypothesis_identifier, output_path, multifasta=False):
    """Write each theoretical glycopeptide in CSV format"""
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    hypothesis = get_by_name_or_id(session, GlycopeptideHypothesis, hypothesis_identifier)

    def generate():
        interval = 100000
        i = 0
        while True:
            session.expire_all()
            chunk = hypothesis.glycopeptides.slice(i, i + interval).all()
            if len(chunk) == 0:
                break
            for glycopeptide in chunk:
                yield glycopeptide
            i += interval
    if output_path is None or output_path == '-':
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = GlycopeptideHypothesisCSVSerializer(output_stream, generate())
        job.run()


@export.command("glycan-identification",
                short_help="Exports assigned LC-MS features of Glycan Compositions to CSV or HTML")
@database_connection_arg
@analysis_identifier_arg("glycan")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-r", "--report", is_flag=True, help="Export an HTML report instead of a CSV")
@click.option("-t", "--threshold", type=float, default=0, help="Minimum score threshold to include")
def glycan_composition_identification(database_connection, analysis_identifier, output_path=None,
                                      threshold=0, report=False):
    """Write each glycan chromatogram in CSV format"""
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    analysis_id = analysis.id
    if output_path is None or output_path == '-':
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')

    if report:
        with output_stream:
            job = GlycanChromatogramReportCreator(
                database_connection._original_connection,
                analysis_id, output_stream, threshold=threshold)
            job.run()
    else:
        def generate():
            i = 0
            interval = 100
            query = session.query(GlycanCompositionChromatogram).filter(
                GlycanCompositionChromatogram.analysis_id == analysis_id,
                GlycanCompositionChromatogram.score > threshold)

            while True:
                session.expire_all()
                chunk = query.slice(i, i + interval).all()
                if len(chunk) == 0:
                    break
                for gcs in chunk:
                    yield gcs.convert()
                i += interval

            i = 0
            query = session.query(UnidentifiedChromatogram).filter(
                UnidentifiedChromatogram.analysis_id == analysis_id,
                UnidentifiedChromatogram.score > threshold)

            while True:
                session.expire_all()
                chunk = query.slice(i, i + interval).all()
                if len(chunk) == 0:
                    break
                for gcs in chunk:
                    yield gcs.convert()
                i += interval

        with output_stream:
            job = GlycanLCMSAnalysisCSVSerializer(output_stream, generate())
            job.run()


@export.command("glycopeptide-identification",
                short_help="Exports assigned LC-MS/MS features of Glycopeptides to CSV or HTML")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-r", "--report", is_flag=True, help="Export an HTML report instead of a CSV")
@click.option("-m", "--mzml-path", type=click.Path(exists=True), default=None, help=(
    "Path to read processed spectra from instead of the path embedded in the analysis metadata"))
@click.option("-t", "--threshold", type=float, default=0, help='The minimum MS2 score threshold (larger is better)')
@click.option("-f", "--fdr-threshold", type=float, default=1, help='The maximum FDR threshold (smaller is better)')
def glycopeptide_identification(database_connection, analysis_identifier, output_path=None,
                                report=False, mzml_path=None, threshold=0, fdr_threshold=1):
    """Write each distinct identified glycopeptide in CSV format"""
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    analysis_id = analysis.id
    is_multiscore = analysis.is_multiscore
    if output_path is None or output_path == '-':
        output_stream = ctxstream(click.get_binary_stream('stdout'))
    else:
        output_stream = open(output_path, 'wb')
    if report:
        with output_stream:
            if mzml_path is None:
                mzml_path = analysis.parameters['sample_path']
                if not os.path.exists(mzml_path):
                    raise click.ClickException(
                        ("Sample path {} not found. Pass the path to"
                         " this file as `-m/--mzml-path` for this command.").format(
                             mzml_path))
            job = GlycopeptideDatabaseSearchReportCreator(
                database_connection._original_connection, analysis_id,
                stream=output_stream,
                threshold=threshold,
                fdr_threshold=fdr_threshold,
                mzml_path=mzml_path)
            job.run()
    else:
        query = session.query(Protein.id, Protein.name).join(Protein.glycopeptides).join(
            IdentifiedGlycopeptide).filter(
                IdentifiedGlycopeptide.analysis_id == analysis.id)
        protein_index = dict(query)

        if is_multiscore:
            job_type = MultiScoreGlycopeptideLCMSMSAnalysisCSVSerializer
        else:
            job_type = GlycopeptideLCMSMSAnalysisCSVSerializer

        def generate():
            i = 0
            interval = 100
            query = session.query(IdentifiedGlycopeptide).filter(
                IdentifiedGlycopeptide.analysis_id == analysis_id)
            if threshold != 0:
                query = query.filter(IdentifiedGlycopeptide.ms2_score >= threshold)
            if fdr_threshold < 1:
                query = query.filter(IdentifiedGlycopeptide.q_value <= fdr_threshold)
            while True:
                session.expunge_all()
                chunk = query.slice(i, i + interval).all()
                if len(chunk) == 0:
                    break
                # chunk = IdentifiedGlycopeptide.bulk_convert(chunk)
                for glycopeptide in chunk:
                    yield glycopeptide
                i += interval
        with output_stream:
            job = job_type(output_stream, generate(), protein_index, analysis)
            job.run()


@export.command("glycopeptide-spectrum-matches",
                short_help="Exports individual MS/MS assignments of Glycopeptides to CSV")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-t", "--threshold", type=float, default=0, help='The minimum MS2 score threshold (larger is better)')
@click.option("-f", "--fdr-threshold", type=float, default=1, help='The maximum FDR threshold (smaller is better)')
def glycopeptide_spectrum_matches(database_connection, analysis_identifier, output_path=None, threshold=0,
                                  fdr_threshold=1.0):
    """Write each matched glycopeptide spectrum in CSV format"""
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    analysis_id = analysis.id
    is_multiscore = analysis.is_multiscore
    start = time.monotonic()
    protein_query = session.query(Protein.id, Protein.name).join(Protein.glycopeptides).join(
        GlycopeptideSpectrumMatch).filter(
            GlycopeptideSpectrumMatch.analysis_id == analysis.id)
    status_logger.info("Loading protein index")
    protein_index = dict(protein_query.all())

    def generate():
        i = 0
        interval = 100000
        gpsm_query = session.query(GlycopeptideSpectrumMatch.id).filter(
            GlycopeptideSpectrumMatch.analysis_id == analysis_id).order_by(
                GlycopeptideSpectrumMatch.scan_id)
        if threshold != 0:
            gpsm_query = gpsm_query.filter(GlycopeptideSpectrumMatch.score >= threshold)
        if fdr_threshold < 1:
            gpsm_query = gpsm_query.filter(GlycopeptideSpectrumMatch.q_value <= fdr_threshold)

        mass_shift_cache = {
            m.id: m.convert()
            for m in session.query(serialize.CompoundMassShift).all()
        }
        scan_cache = {}
        structure_cache = {}
        peptide_relation_cache = {}

        status_logger.info("Reading spectrum matches")
        j = 0
        while True:
            session.expire_all()
            chunk = gpsm_query.slice(i, i + interval).all()
            chunk = [x[0] for x in chunk]
            chunk = GlycopeptideSpectrumMatch.bulk_load(
                session,
                chunk,
                mass_shift_cache=mass_shift_cache,
                scan_cache=scan_cache,
                structure_cache=structure_cache,
                peptide_relation_cache=peptide_relation_cache
            )
            if not chunk:
                break
            chunk = [x[1] for x in sorted(chunk.items())]
            for glycopeptide in chunk:
                j += 1
                yield glycopeptide
            i += interval

    if is_multiscore:
        job_type = MultiScoreGlycopeptideSpectrumMatchAnalysisCSVSerializer
    else:
        job_type = GlycopeptideSpectrumMatchAnalysisCSVSerializer

    if output_path is None or output_path == '-':
        output_stream = ctxstream(click.get_binary_stream('stdout'))
    else:
        output_stream = open(output_path, 'wb')

    with output_stream:
        job = job_type(output_stream, generate(), protein_index, analysis)
        job.run()

    end = time.monotonic()
    elapsed = end - start
    status_logger.info(f"{elapsed:0.2f} seconds elapsed")


@export.command("mzid", short_help="Export a Glycopeptide Analysis as MzIdentML")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.argument("output-path")
@click.option("--embed-protein-sequences/--exclude-protein-sequences", is_flag=True, default=True,
              help="Include protein sequences in the output file")
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
def glycopeptide_mzidentml(database_connection, analysis_identifier, output_path=None,
                           mzml_path=None, embed_protein_sequences=True):
    """
    Write identified glycopeptides as mzIdentML file, and associated MSn spectra
    to a paired mzML file if the matched data are available. If an mzML file is written
    it will also contain the extracted ion chromatograms for each glycopeptide with an
    extracted elution profile.
    """
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    loader = AnalysisDeserializer(
        database_connection._original_connection, analysis_id=analysis.id)
    click.echo("Loading Identifications")
    # glycopeptides = loader.load_identified_glycopeptides()
    glycopeptides = loader.query(IdentifiedGlycopeptide).filter(
        IdentifiedGlycopeptide.analysis_id == analysis_identifier).all()
    with open(output_path, 'wb') as outfile:
        writer = MzIdentMLSerializer(
            outfile, glycopeptides, analysis, loader,
            source_mzml_path=mzml_path,
            embed_protein_sequences=embed_protein_sequences)
        writer.run()


@export.command("glycopeptide-training-mgf", short_help=("Export identified MS2 spectra as MGF with"
                                                         " matched glycopeptide as metadata"))
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
@click.option("-t", "--threshold", type=float, default=None, help='Minimum MS2 score to export')
def glycopeptide_training_mgf(database_connection, analysis_identifier, output_path=None,
                              mzml_path=None, threshold=None):
    """
    Export identified glycopeptide spectrum matches for model training as
    an annotated MGF file.
    """
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    if output_path is None or output_path == '-':
        output_stream = ctxstream(click.get_binary_stream('stdout'))
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        TrainingMGFExporter.from_analysis(
            database_connection, analysis.id, output_stream, mzml_path, threshold).run()


@export.command("identified-glycans-from-glycopeptides", short_help="Export glycans on identified glycopeptides")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def export_identified_glycans_from_glycopeptides(database_connection, analysis_identifier, output_path):
    """
    Export the glycans attached to any identified glycopeptides.

    The files are written in an importable format to be used to create a new glycan hypothesis.
    """
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    glycans = session.query(GlycanComposition).join(
        GlycanCombinationGlycanComposition).join(GlycanCombination).join(
            Glycopeptide,
            Glycopeptide.glycan_combination_id == GlycanCombination.id).join(
                IdentifiedGlycopeptide,
                IdentifiedGlycopeptide.structure_id == Glycopeptide.id).filter(
                    IdentifiedGlycopeptide.analysis_id == analysis.id).all()
    if output_path is None:
        output_stream = ctxstream(click.get_binary_stream('stdout'))
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = ImportableGlycanHypothesisCSVSerializer(output_stream, glycans)
        job.run()


@export.command("annotate-matched-spectra",
                short_help="Exports graphical renderings of individual MS/MS assignments of glycopeptides to PDF")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
def annotate_matched_spectra(database_connection, analysis_identifier, output_path, mzml_path=None):
    """
    Export graphical renderings of individual MS/MS assignments of glycopeptides to PDF.

    The output file will contain one spectrum per PDF page.
    """
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    if output_path is None:
        output_path = os.path.dirname(database_connection._original_connection)

    task = SpectrumAnnotatorExport(
        database_connection._original_connection, analysis.id, output_path,
        mzml_path)
    task.display_header()
    task.start()


@export.command("write-csv-spectrum-library", short_help="Exports individual MS/MS assignments of glycopeptides to CSV")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
@click.option("-t", "--fdr-threshold", type=float, default=0.05)
def write_spectrum_library(database_connection, analysis_identifier, output_path, mzml_path=None, fdr_threshold=0.05):
    """
    Export individual MS/MS assignments of glycopeptides to CSV.

    Each row corresponds to a single product ion match, partitioned by
    scan ID. Unmatched peaks are ignored.
    """
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    if output_path is None:
        output_stream = ctxstream(click.get_binary_stream('stdout'))
    else:
        output_stream = open(output_path, 'wb')

    with output_stream:
        task = CSVSpectrumAnnotatorExport(
            database_connection._original_connection, analysis.id, output_stream,
            mzml_path, fdr_threshold)
        task.run()


@export.command("glycopeptide-chromatogram-records")
@database_connection_arg
@analysis_identifier_arg("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option('-r', '--apex-time-range', type=(float, float), default=(0, float('inf')),
              help='The range of apex times to include')
@click.option("-t", "--fdr-threshold", type=float, default=0.05)
def glycopeptide_chromatogram_records(
        database_connection, analysis_identifier, output_path, apex_time_range=None, fdr_threshold=0.05):
    """
    Export a simplified table of glycopeptide chromatographic features, formatted for retention time
    modeling.
    """
    if apex_time_range is None:
        apex_time_range = (0, float('inf'))
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    if output_path is None:
        fh = click.get_binary_stream('stdout')
    else:
        fh = open(output_path, 'wb')
    idgps = session.query(
        IdentifiedGlycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == analysis.id).all()
    n = len(idgps)
    from glycresoft.scoring.elution_time_grouping import GlycopeptideChromatogramProxy
    cases = []
    analysis_name = analysis.name
    start_time, stop_time = apex_time_range
    for i, idgp in enumerate(idgps):
        if i % 50 == 0:
            click.echo("%d/%d Records Processed" % (i, n), err=True)
        if idgp.chromatogram is None:
            continue
        if idgp.ms1_score < 0:
            continue
        if idgp.q_value > fdr_threshold:
            continue
        obj = GlycopeptideChromatogramProxy.from_chromatogram(
            idgp, ms1_score=idgp.ms1_score, ms2_score=idgp.ms2_score,
            q_value=idgp.q_value, analysis_name=analysis_name,
            mass_shifts=idgp.chromatogram.mass_shifts)
        if obj.apex_time < start_time or obj.apex_time > stop_time:
            continue
        cases.append(obj)
    click.echo("Writing %d Records" % (len(cases), ), err=True)
    with fh:
        GlycopeptideChromatogramProxy.to_csv(cases, csv_stream(fh))
