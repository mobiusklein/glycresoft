import os
import sys

import click

from glycan_profiling.cli.base import cli
from glycan_profiling.cli.validators import get_by_name_or_id, DatabaseConnectionParam

from glycan_profiling.serialize import (
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

from glycan_profiling.output import (
    GlycanHypothesisCSVSerializer, ImportableGlycanHypothesisCSVSerializer,
    GlycopeptideHypothesisCSVSerializer, GlycanLCMSAnalysisCSVSerializer,
    GlycopeptideLCMSMSAnalysisCSVSerializer,
    GlycopeptideSpectrumMatchAnalysisCSVSerializer,
    MzIdentMLSerializer,
    GlycanChromatogramReportCreator,
    GlycopeptideDatabaseSearchReportCreator,
    TrainingMGFExporter)

from glycan_profiling.cli.utils import ctxstream


def database_connection(fn):
    arg = click.argument(
        "database-connection",
        type=DatabaseConnectionParam(exists=True),
        doc_help=(
            "A connection URI for a database, or a path on the file system"))
    return arg(fn)


def analysis_identifier(analysis_type):
    def wrapper(fn):
        arg = click.argument("analysis-identifier", doc_help=(
            "The ID number or name of the %s analysis to use" % (analysis_type,)))
        return arg(fn)
    return wrapper


def hypothesis_identifier(hypothesis_type):
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
@database_connection
@hypothesis_identifier("glycan")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-i", "--importable", is_flag=True,
              help="Make the file importable for later re-use, with less information.")
def glycan_hypothesis(database_connection, hypothesis_identifier, output_path=None, importable=False):
    '''Write each theoretical glycan composition in CSV format
    '''
    database_connection = DatabaseBoundOperation(database_connection)
    hypothesis = get_by_name_or_id(database_connection, GlycanHypothesis, hypothesis_identifier)
    if importable:
        task_type = ImportableGlycanHypothesisCSVSerializer
    else:
        task_type = GlycanHypothesisCSVSerializer
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = task_type(output_stream, hypothesis.glycans)
        job.run()


@export.command("glycopeptide-hypothesis",
                short_help='Export theoretical glycopeptide database to CSV')
@database_connection
@hypothesis_identifier("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def glycopeptide_hypothesis(database_connection, hypothesis_identifier, output_path, multifasta=False):
    '''Write each theoretical glycopeptide in CSV format
    '''
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
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
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = GlycopeptideHypothesisCSVSerializer(output_stream, generate())
        job.run()


@export.command("glycan-identification",
                short_help="Exports assigned LC-MS features of Glycan Compositions to CSV or HTML")
@database_connection
@analysis_identifier("glycan")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-r", "--report", is_flag=True, help="Export an HTML report instead of a CSV")
@click.option("-t", "--threshold", type=float, default=0)
def glycan_composition_identification(database_connection, analysis_identifier, output_path=None,
                                      threshold=0, report=False):
    '''Write each glycan chromatogram in CSV format
    '''
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    analysis_id = analysis.id
    if output_path is None:
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
@database_connection
@analysis_identifier("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-r", "--report", is_flag=True, help="Export an HTML report instead of a CSV")
@click.option("-m", "--mzml-path", type=click.Path(exists=True), default=None, help=(
    "Path to read processed spectra from instead of the path embedded in the analysis metadata"))
@click.option("-t", "--threshold", type=float, default=0)
def glycopeptide_identification(database_connection, analysis_identifier, output_path=None,
                                report=False, mzml_path=None, threshold=0):
    '''Write each distinct identified glycopeptide in CSV format
    '''
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    analysis_id = analysis.id
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
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
            GlycopeptideDatabaseSearchReportCreator(
                database_connection._original_connection, analysis_id,
                stream=output_stream, threshold=threshold,
                mzml_path=mzml_path).run()
    else:
        query = session.query(Protein.id, Protein.name).join(Protein.glycopeptides).join(
            IdentifiedGlycopeptide).filter(
            IdentifiedGlycopeptide.analysis_id == analysis.id)
        protein_index = dict(query)

        def generate():
            i = 0
            interval = 100
            query = session.query(IdentifiedGlycopeptide).filter(
                IdentifiedGlycopeptide.analysis_id == analysis_id)
            while True:
                session.expire_all()
                chunk = query.slice(i, i + interval).all()
                if len(chunk) == 0:
                    break
                for glycopeptide in chunk:
                    yield glycopeptide.convert()
                i += interval
        with output_stream:
            job = GlycopeptideLCMSMSAnalysisCSVSerializer(output_stream, generate(), protein_index)
            job.run()


@export.command("glycopeptide-spectrum-matches",
                short_help="Exports individual MS/MS assignments of Glycopeptides to CSV")
@database_connection
@analysis_identifier("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def glycopeptide_spectrum_matches(database_connection, analysis_identifier, output_path=None):
    '''Write each matched glycopeptide spectrum in CSV format
    '''
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    analysis_id = analysis.id
    query = session.query(Protein.id, Protein.name).join(Protein.glycopeptides).join(
        GlycopeptideSpectrumMatch).filter(
        GlycopeptideSpectrumMatch.analysis_id == analysis.id)
    protein_index = dict(query)

    def generate():
        i = 0
        interval = 100000
        query = session.query(GlycopeptideSpectrumMatch).filter(
            GlycopeptideSpectrumMatch.analysis_id == analysis_id).order_by(
            GlycopeptideSpectrumMatch.scan_id)
        while True:
            session.expire_all()
            chunk = query.slice(i, i + interval).all()
            if len(chunk) == 0:
                break
            for glycopeptide in chunk:
                yield glycopeptide.convert()
            i += interval

    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = GlycopeptideSpectrumMatchAnalysisCSVSerializer(output_stream, generate(), protein_index)
        job.run()


@export.command("mzid", short_help="Export a Glycopeptide Analysis as MzIdentML")
@database_connection
@analysis_identifier("glycopeptide")
@click.argument("output-path")
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
def glycopeptide_mzidentml(database_connection, analysis_identifier, output_path=None,
                           mzml_path=None):
    '''Write identified glycopeptides as mzIdentML file, and associated MSn spectra
    to a paired mzML file if the matched data are available. If an mzML file is written
    it will also contain the extracted ion chromatograms for each glycopeptide with an
    extracted elution profile.
    '''
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    loader = AnalysisDeserializer(
        database_connection._original_connection, analysis_id=analysis.id)
    glycopeptides = loader.load_identified_glycopeptides()
    with open(output_path, 'wb') as outfile:
        writer = MzIdentMLSerializer(
            outfile, glycopeptides, analysis, loader, source_mzml_path=mzml_path)
        writer.run()


@export.command("glycopeptide-training-mgf")
@database_connection
@analysis_identifier("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
@click.option("-t", "--threshold", type=float, default=None)
def glycopeptide_training_mgf(database_connection, analysis_identifier, output_path=None,
                              mzml_path=None, threshold=None):
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        raise click.Abort()
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        TrainingMGFExporter.from_analysis(
            database_connection, analysis.id, output_stream, mzml_path, threshold).run()


@export.command("identified-glycans-from-glycopeptides")
@database_connection
@analysis_identifier("glycopeptide")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def export_identified_glycans_from_glycopeptides(database_connection, analysis_identifier, output_path):
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()
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
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = ImportableGlycanHypothesisCSVSerializer(output_stream, glycans)
        job.run()
