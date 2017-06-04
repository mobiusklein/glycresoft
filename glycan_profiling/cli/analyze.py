import os
import multiprocessing
from uuid import uuid4

import click
from .base import cli

from glycan_profiling.serialize import (
    DatabaseBoundOperation,
    GlycanHypothesis,
    GlycopeptideHypothesis)

from glycan_profiling.profiler import (
    MzMLGlycopeptideLCMSMSAnalyzer,
    MzMLGlycanChromatogramAnalyzer,
    ProcessedMzMLDeserializer)

from glycan_profiling.tandem.glycopeptide.scoring import CoverageWeightedBinomialScorer

from glycan_profiling.models import GeneralScorer

from .validators import (
    validate_analysis_name,
    validate_adduct, validate_glycopeptide_tandem_scoring_function,
    glycopeptide_tandem_scoring_functions,
    get_by_name_or_id)


def make_analysis_output_path(prefix):
    output_path_pattern = prefix + "_analysis_%d.db"
    i = 1
    while os.path.exists(output_path_pattern % i) and i < (2 ** 16):
        i += 1
    if i >= (2 ** 16):
        return output_path_pattern % (uuid4().int,)
    else:
        return output_path_pattern % (i,)


@cli.group(short_help='Identify structures in preprocessed data')
def analyze():
    pass


@analyze.command("search-glycopeptide", short_help='Search preprocessed data for glycopeptide sequences')
@click.pass_context
@click.argument("database-connection")
@click.argument("sample-path")
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
@click.option("-s", "--tandem-scoring-model", default='coverage_weighted_binomial', type=click.Choice(
              glycopeptide_tandem_scoring_functions.keys()),
              help="Select a scoring function to use for evaluating glycopeptide-spectrum matches")
@click.option("-x", "--oxonium-threshold", default=0.05, type=float,
              help=('Minimum HexNAc-derived oxonium ion abundance '
                    'ratio to filter MS/MS scans. Defaults to 0.05.'))
@click.option("-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
              default=min(multiprocessing.cpu_count(), 4),
              help=('Number of worker processes to use. Defaults to 4 '
                    'or the number of CPUs, whichever is lower'))
@click.option("-o", "--output-path", default=None, type=click.Path(writable=True), help=(
              "Path to write resulting analysis to."))
@click.option("--save-intermediate-results", default=None, type=click.Path(), required=False,
              help='Save intermediate spectrum matches to a file')
def search_glycopeptide(context, database_connection, sample_path, hypothesis_identifier,
                        analysis_name, output_path=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                        msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                        tandem_scoring_model=None, oxonium_threshold=0.15,
                        save_intermediate_results=None, processes=4):
    """Identify glycopeptide sequences from preprocessed LC-MS/MS data, stored in mzML
    format.
    """
    if output_path is None:
        output_path = make_analysis_output_path("glycopeptide")
    if peak_shape_scoring_model is None:
        peak_shape_scoring_model = GeneralScorer
    if tandem_scoring_model is None:
        tandem_scoring_model = CoverageWeightedBinomialScorer
    database_connection = DatabaseBoundOperation(database_connection)
    ms_data = ProcessedMzMLDeserializer(sample_path, use_index=False)
    sample_run = ms_data.sample_run

    try:
        hypothesis = get_by_name_or_id(
            database_connection, GlycopeptideHypothesis, hypothesis_identifier)
    except Exception:
        click.secho("Could not locate a Glycopeptide Hypothesis with identifier %r" %
                    hypothesis_identifier, fg='yellow')
        raise click.Abort()

    tandem_scoring_model = validate_glycopeptide_tandem_scoring_function(
        context, tandem_scoring_model)

    if analysis_name is None:
        analysis_name = "%s @ %s" % (sample_run.name, hypothesis.name)

    analysis_name = validate_analysis_name(
        context, database_connection.session, analysis_name)

    click.secho("Preparing analysis of %s by %s" % (
        sample_run.name, hypothesis.name), fg='cyan')

    analyzer = MzMLGlycopeptideLCMSMSAnalyzer(
        database_connection._original_connection,
        sample_path=sample_path,
        hypothesis_id=hypothesis.id,
        analysis_name=analysis_name,
        output_path=output_path,
        grouping_error_tolerance=grouping_error_tolerance,
        mass_error_tolerance=mass_error_tolerance,
        msn_mass_error_tolerance=msn_mass_error_tolerance,
        psm_fdr_threshold=psm_fdr_threshold,
        peak_shape_scoring_model=peak_shape_scoring_model,
        tandem_scoring_model=tandem_scoring_model,
        oxonium_threshold=oxonium_threshold,
        n_processes=processes)

    gps, unassigned, target_hits, decoy_hits = analyzer.start()
    if save_intermediate_results is not None:
        import cPickle as pickle
        analyzer.log("Saving Intermediate Results")
        with open(save_intermediate_results, 'wb') as handle:
            pickle.dump((target_hits, decoy_hits, gps), handle)


@analyze.command("search-glycan", short_help=('Search preprocessed data for'
                                              ' glycan compositions'))
@click.pass_context
@click.argument("database-connection")
@click.argument("sample-path")
@click.argument("hypothesis-identifier")
@click.option("-m", "--mass-error-tolerance", type=float, default=1e-5,
              help=("Mass accuracy constraint, in parts-per-million "
                    "error, for matching."))
@click.option("-g", "--grouping-error-tolerance", type=float, default=1.5e-5,
              help=("Mass accuracy constraint, in parts-per-million error, for"
                    " grouping chromatograms."))
@click.option("-n", "--analysis-name", default=None,
              help='Name for analysis to be performed.')
@click.option("-a", "--adduct", 'adducts', multiple=True, nargs=2,
              help=("Adducts to consider. Specify name or formula, and a"
                    " multiplicity."))
@click.option("-d", "--minimum-mass", default=500., type=float,
              help="The minimum mass to consider signal at.")
@click.option("-o", "--output-path", default=None, help=(
              "Path to write resulting analysis to."))
@click.option("-s", "--network-sharing", default=0.2, type=float,
              help="The weight to place on similar compositions' confidence for evaluating"
                   " one composition's confidence.")
def search_glycan(context, database_connection, sample_path,
                  hypothesis_identifier,
                  analysis_name, adducts, grouping_error_tolerance=1.5e-5,
                  mass_error_tolerance=1e-5, minimum_mass=500.,
                  scoring_model=None, network_sharing=0.2,
                  output_path=None):
    """Identify glycan compositions from preprocessed LC-MS data, stored in mzML
    format.
    """
    if output_path is None:
        output_path = make_analysis_output_path("glycan")
    if scoring_model is None:
        scoring_model = GeneralScorer

    database_connection = DatabaseBoundOperation(database_connection)
    ms_data = ProcessedMzMLDeserializer(sample_path, use_index=False)
    sample_run = ms_data.sample_run

    try:
        hypothesis = get_by_name_or_id(
            database_connection, GlycanHypothesis, hypothesis_identifier)
    except Exception:
        click.secho("Could not locate a Glycan Hypothesis with identifier %r" %
                    hypothesis_identifier, fg='yellow')
        raise click.Abort()

    if analysis_name is None:
        analysis_name = "%s @ %s" % (sample_run.name, hypothesis.name)
    analysis_name = validate_analysis_name(
        context, database_connection.session, analysis_name)

    adducts = [validate_adduct(adduct, multiplicity)
               for adduct, multiplicity in adducts]
    expanded = []
    # for adduct, mult in adducts:
    #     for i in range(1, mult + 1):
    #         expanded.append(adduct * i)
    expanded = MzMLGlycanChromatogramAnalyzer.expand_adducts(dict(adducts))
    adducts = expanded

    click.secho("Preparing analysis of %s by %s" %
                (sample_run.name, hypothesis.name), fg='cyan')

    analyzer = MzMLGlycanChromatogramAnalyzer(
        database_connection._original_connection, hypothesis.id,
        sample_path=sample_path, output_path=output_path, adducts=adducts,
        mass_error_tolerance=mass_error_tolerance,
        grouping_error_tolerance=grouping_error_tolerance, scoring_model=scoring_model,
        minimum_mass=minimum_mass, network_sharing=network_sharing,
        analysis_name=analysis_name)
    analyzer.start()
