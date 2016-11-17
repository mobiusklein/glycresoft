import click
from .base import cli

from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    SampleRun)

from glycan_profiling.trace import (
    ChromatogramProcessor, ChromatogramExtractor)

from glycan_profiling.scoring import chromatogram_solution, shape_fitter

from glycan_profiling.profiler import (
    GlycanChromatogramAnalyzer, GlycopeptideLCMSMSAnalyzer)

from glycan_profiling.database.disk_backed_database import (
    GlycanCompositionDiskBackedStructureDatabase)

from glycan_profiling.tandem.glycopeptide.scoring import BinomialSpectrumMatcher, CoverageWeightedBinomialScorer

from glycan_profiling.models import GeneralScorer

from .validators import (
    validate_analysis_name,
    validate_adduct, validate_glycopeptide_tandem_scoring_function,
    glycopeptide_tandem_scoring_functions)


@cli.group(short_help='Identify structures in preprocessed data')
def analyze():
    pass


def get_by_name_or_id(session, model_type, name_or_id):
    try:
        object_id = int(name_or_id)
        inst = session.query(model_type).get(object_id)
        if inst is None:
            raise ValueError("No instance of type %s with id %r" %
                             (model_type, name_or_id))
        return inst
    except ValueError:
        inst = session.query(model_type).filter(
            model_type.name == name_or_id).one()
        return inst


@analyze.command("search-glycopeptide", short_help='Search preprocessed data for glycopeptide sequences')
@click.pass_context
@click.argument("database-connection")
@click.argument("sample-identifier")
@click.argument("hypothesis-identifier")
@click.option("-m", "--mass-error-tolerance", type=float, default=1e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^1 ions.")
@click.option("-mn", "--msn-mass-error-tolerance", type=float, default=2e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^n ions.")
@click.option("-g", "--grouping-error-tolerance", type=float, default=1.5e-5,
              help="Mass accuracy constraint, in parts-per-million error, for grouping chromatograms.")
@click.option("-n", "--analysis-name", default=None, help='Name for analysis to be performed.')
@click.option("-q", "--psm-fdr-threshold", default=0.05, type=float,
              help='Minimum FDR Threshold to use for filtering PSMs when selecting identified glycopeptides')
@click.option("-s", "--tandem-scoring-model", default='binomial', type=click.Choice(
              glycopeptide_tandem_scoring_functions.keys()),
              help="Select a scoring function to use for evaluating glycopeptide-spectrum matches")
def search_glycopeptide(context, database_connection, sample_identifier, hypothesis_identifier,
                        analysis_name, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                        msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                        tandem_scoring_model=None):
    if peak_shape_scoring_model is None:
        peak_shape_scoring_model = GeneralScorer
    if tandem_scoring_model is None:
        tandem_scoring_model = CoverageWeightedBinomialScorer
    database_connection = DatabaseBoundOperation(database_connection)
    try:
        sample_run = get_by_name_or_id(
            database_connection, SampleRun, sample_identifier)
    except:
        click.secho("Could not locate a Sample Run with identifier %r" %
                    sample_identifier, fg='yellow')
        raise click.Abort()
    try:
        hypothesis = get_by_name_or_id(
            database_connection, GlycopeptideHypothesis, hypothesis_identifier)
    except:
        click.secho("Could not locate a Glycan Hypothesis with identifier %r" %
                    sample_identifier, fg='yellow')
        raise click.Abort()

    tandem_scoring_model = validate_glycopeptide_tandem_scoring_function(
        context, tandem_scoring_model)

    if analysis_name is None:
        analysis_name = "%s @ %s" % (sample_run.name, hypothesis.name)
    analysis_name = validate_analysis_name(
        context, database_connection.session, analysis_name)

    click.secho("Preparing analysis of %s by %s" % (sample_run.name, hypothesis.name), fg='cyan')

    try:
        analyzer = GlycopeptideLCMSMSAnalyzer(
            database_connection._original_connection, hypothesis.id, sample_run.id,
            analysis_name, grouping_error_tolerance=grouping_error_tolerance, mass_error_tolerance=mass_error_tolerance,
            msn_mass_error_tolerance=msn_mass_error_tolerance, psm_fdr_threshold=psm_fdr_threshold,
            peak_shape_scoring_model=peak_shape_scoring_model, tandem_scoring_model=tandem_scoring_model)
        analyzer.start()
    except:
        import pdb
        import traceback
        import sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


@analyze.command("search-glycan", short_help='Search preprocessed data for glycan compositions')
@click.pass_context
@click.argument("database-connection")
@click.argument("sample-identifier")
@click.argument("hypothesis-identifier")
@click.option("-m", "--mass-error-tolerance", type=float, default=1e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching.")
@click.option("-g", "--grouping-error-tolerance", type=float, default=1.5e-5,
              help="Mass accuracy constraint, in parts-per-million error, for grouping chromatograms.")
@click.option("-n", "--analysis-name", default=None, help='Name for analysis to be performed.')
@click.option("-a", "--adduct", 'adducts', multiple=True, nargs=2,
              help="Adducts to consider. Specify name or formula, and a multiplicity.")
def search_glycan(context, database_connection, sample_identifier, hypothesis_identifier,
                  analysis_name, adducts, grouping_error_tolerance=1.5e-5,
                  mass_error_tolerance=1e-5, scoring_model=None):
    if scoring_model is None:
        scoring_model = GeneralScorer

    database_connection = DatabaseBoundOperation(database_connection)
    try:
        sample_run = get_by_name_or_id(
            database_connection, SampleRun, sample_identifier)
    except:
        click.secho("Could not locate a Sample Run with identifier %r" %
                    sample_identifier, fg='yellow')
        raise click.Abort()
    try:
        hypothesis = get_by_name_or_id(
            database_connection, GlycanHypothesis, hypothesis_identifier)
    except:
        click.secho("Could not locate a Glycan Hypothesis with identifier %r" %
                    sample_identifier, fg='yellow')
        raise click.Abort()

    if analysis_name is None:
        analysis_name = "%s @ %s" % (sample_run.name, hypothesis.name)
    analysis_name = validate_analysis_name(
        context, database_connection.session, analysis_name)

    adducts = [validate_adduct(adduct, multiplicity)
               for adduct, multiplicity in adducts]
    expanded = []
    for adduct, mult in adducts:
        for i in range(1, mult + 1):
            expanded.append(adduct * i)
    adducts = expanded

    for adduct in adducts:
        click.secho("Adduct: %s" % adduct)

    click.secho("Preparing analysis of %s by %s" %
                (sample_run.name, hypothesis.name), fg='cyan')

    try:
        analyzer = GlycanChromatogramAnalyzer(
            database_connection._original_connection, hypothesis.id,
            sample_run.id, adducts=adducts, mass_error_tolerance=mass_error_tolerance,
            grouping_error_tolerance=grouping_error_tolerance, scoring_model=scoring_model,
            analysis_name=analysis_name)
        proc = analyzer.start()
    except:
        import pdb
        import traceback
        import sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
