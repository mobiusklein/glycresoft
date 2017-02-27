import sys

import click

from glycan_profiling.cli.base import cli
from glycan_profiling.cli.validators import get_by_name_or_id

from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    Analysis, AnalysisTypeEnum, GlycanCompositionChromatogram,
    Protein, Glycopeptide, IdentifiedGlycopeptide,
    GlycopeptideSpectrumMatch, AnalysisDeserializer, GlycanComposition,
    GlycanCombination, GlycanCombinationGlycanComposition)

from glycan_profiling.output import (
    GlycanHypothesisCSVSerializer, ImportableGlycanHypothesisCSVSerializer,
    GlycopeptideHypothesisCSVSerializer, GlycanLCMSAnalysisCSVSerializer,
    GlycopeptideLCMSMSAnalysisCSVSerializer,
    GlycopeptideSpectrumMatchAnalysisCSVSerializer,
    MzIdentMLSerializer)

from glycan_profiling.serialize import (DatabaseScanDeserializer, SampleRun)
from glycan_profiling.scan_cache import MzMLScanCacheHandler


class ctxstream(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, string):
        self.stream.write(string)

    def flush(self):
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


@cli.group(short_help='Write Data Collections To Text Files')
def export():
    pass


@export.command("glycan-hypothesis",
                short_help="Export theoretical glycan composition database to CSV")
@click.argument("database-connection")
@click.argument("hypothesis-identifier")
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
@click.argument("database-connection")
@click.argument("hypothesis-identifier")
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
                short_help="Exports assigned LC-MS features of Glycan Compositions to CSV")
@click.argument("database-connection")
@click.argument("analysis-identifier")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def glycan_composition_identification(database_connection, analysis_identifier, output_path=None):
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

    def generate():
        i = 0
        interval = 100
        query = session.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == analysis_id)
        while True:
            session.expire_all()
            chunk = query.slice(i, i + interval).all()
            if len(chunk) == 0:
                break
            for gcs in chunk:
                yield gcs.convert()
            i += interval
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = GlycanLCMSAnalysisCSVSerializer(output_stream, generate())
        job.run()


@export.command("glycopeptide-identification",
                short_help="Exports assigned LC-MS/MS features of Glycopeptides to CSV")
@click.argument("database-connection")
@click.argument("analysis-identifier")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
def glycopeptide_identification(database_connection, analysis_identifier, output_path=None):
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

    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'wb')
    with output_stream:
        job = GlycopeptideLCMSMSAnalysisCSVSerializer(output_stream, generate(), protein_index)
        job.run()


@export.command("glycopeptide-spectrum-matches",
                short_help="Exports individual MS/MS assignments of Glycopeptides to CSV")
@click.argument("database-connection")
@click.argument("analysis-identifier")
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
@click.argument("database-connection")
@click.argument("analysis-identifier")
@click.argument("output-path")
@click.option("-m", '--mzml-path', type=click.Path(exists=True), default=None,
              help="Alternative path to find the source mzML file")
def glycopeptide_mzidentml(database_connection, analysis_identifier, output_path=None,
                           mzml_path=None):
    '''Write identified glycopeptides as mzIdentML file, and associated MSn spectra
    to a paired mzML file if the matched data are available. If an mzML file is written
    it will also contain the extracted ion chromatograms for each glycopeptide with an
    extracted trace.
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


@export.command("sample-run", short_help="Export a fully preprocessed sample run as mzML")
@click.argument("database-connection")
@click.argument("sample-run-identifier")
@click.argument("output-path")
def export_sample_run(database_connection, sample_run_identifier, output_path):
    database_connection = DatabaseBoundOperation(database_connection)
    sample_run = get_by_name_or_id(
        database_connection.session, SampleRun, sample_run_identifier)
    n_spectra = sample_run.ms_scans.count()
    writer = MzMLScanCacheHandler.configure_storage(output_path, sample_run.name, None)
    writer.n_spectra = n_spectra
    reader = DatabaseScanDeserializer(
        database_connection._original_connection, sample_run_id=sample_run.id)
    i = 0
    last = 0
    for scan_bunch in reader:
        i += 1 + len(scan_bunch.products)
        if i - 100 > last:
            click.echo("%0.2f%% complete" % (i * 100. / n_spectra))
            last = i
        writer.save_bunch(*scan_bunch)
    writer.complete()


@export.command("identified-glycans-from-glycopeptides")
@click.argument("database-connection")
@click.argument("analysis-identifier")
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
