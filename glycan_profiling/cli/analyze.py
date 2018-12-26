import os
from uuid import uuid4

import dill as pickle

import click
from .base import (
    cli, HiddenOption, processes_option)

from glycan_profiling.serialize import (
    DatabaseBoundOperation,
    GlycanHypothesis,
    GlycopeptideHypothesis)

from glycan_profiling.profiler import (
    MzMLGlycopeptideLCMSMSAnalyzer,
    MzMLComparisonGlycopeptideLCMSMSAnalyzer,
    MzMLGlycanChromatogramAnalyzer,
    LaplacianRegularizedChromatogramProcessor,
    ProcessedMzMLDeserializer)

from glycan_profiling.tandem.glycopeptide.scoring import CoverageWeightedBinomialScorer
from glycan_profiling.composition_distribution_model import GridPointSolution
from glycan_profiling.database.composition_network import GraphReader

from glycan_profiling.models import GeneralScorer
from glycan_profiling.task import fmt_msg

from .validators import (
    validate_analysis_name,
    validate_mass_shift, validate_glycopeptide_tandem_scoring_function,
    glycopeptide_tandem_scoring_functions,
    get_by_name_or_id,
    validate_ms1_feature_name,
    ms1_model_features,
    RelativeMassErrorParam,
    DatabaseConnectionParam,
    GlycopeptideFDRParam)


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


def database_connection(fn):
    arg = click.argument(
        "database-connection",
        type=DatabaseConnectionParam(exists=True),
        doc_help=(
            "A connection URI for a database, or a path on the file system"))
    return arg(fn)


def hypothesis_identifier(hypothesis_type):
    def wrapper(fn):
        arg = click.argument("hypothesis-identifier", doc_help=(
            "The ID number or name of the %s hypothesis to use" % (hypothesis_type,)))
        return arg(fn)
    return wrapper


def sample_path(fn):
    arg = click.argument(
        "sample-path",
        type=click.Path(exists=True, dir_okay=False, file_okay=True),
        doc_help=(
            "The path to the deconvoluted sample file"))
    return arg(fn)


@analyze.command("search-glycopeptide", short_help='Search preprocessed data for glycopeptide sequences')
@click.pass_context
@database_connection
@sample_path
@hypothesis_identifier("glycopeptide")
@click.option("-m", "--mass-error-tolerance", type=RelativeMassErrorParam(), default=1e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^1 ions.")
@click.option("-mn", "--msn-mass-error-tolerance", type=RelativeMassErrorParam(), default=2e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^n ions.")
@click.option("-g", "--grouping-error-tolerance", type=RelativeMassErrorParam(), default=1.5e-5,
              help="Mass accuracy constraint, in parts-per-million error, for grouping chromatograms.")
@click.option("-n", "--analysis-name", default=None, help='Name for analysis to be performed.')
@click.option("-q", "--psm-fdr-threshold", default=0.05, type=float,
              help='Minimum FDR Threshold to use for filtering GPSMs when selecting identified glycopeptides')
@click.option("-s", "--tandem-scoring-model", default='coverage_weighted_binomial', type=click.Choice(
              glycopeptide_tandem_scoring_functions.keys()),
              help="Select a scoring function to use for evaluating glycopeptide-spectrum matches")
@click.option("-x", "--oxonium-threshold", default=0.05, type=float,
              help=('Minimum HexNAc-derived oxonium ion abundance '
                    'ratio to filter MS/MS scans. Defaults to 0.05.'))
@click.option("-a", "--adduct", 'mass_shifts', multiple=True, nargs=2,
              help=("Adducts to consider. Specify name or formula, and a"
                    " multiplicity."))
@click.option("-f", "--use-peptide-mass-filter", is_flag=True, help=(
    "Filter putative spectrum matches by estimating the peptide backbone mass "
    "from the precursor mass and stub glycopeptide signature ions"))
@processes_option
@click.option("--export", type=click.Choice(
              ['csv', 'html', 'psm-csv']), multiple=True,
              help="export command to after search is complete")
@click.option("-o", "--output-path", default=None, type=click.Path(writable=True), help=(
              "Path to write resulting analysis to."))
@click.option("-w", "--workload-size", default=500, type=int, help="Number of spectra to process at once")
@click.option("--save-intermediate-results", default=None, type=click.Path(), required=False,
              help='Save intermediate spectrum matches to a file', cls=HiddenOption)
@click.option("--maximum-mass", default=float('inf'), type=float, cls=HiddenOption)
@click.option("-D", "--decoy-database-connection", default=None, cls=HiddenOption)
@click.option("--fdr-correction", default='auto', type=GlycopeptideFDRParam(),
              help=("Whether to attempt to correct for small sample size for target-decoy analysis."),
              cls=HiddenOption)
@click.option("--isotope-probing-range", type=int, default=3, help=(
    "The maximum number of isotopic peak errors to allow when searching for untrusted precursor masses"))
def search_glycopeptide(context, database_connection, sample_path, hypothesis_identifier,
                        analysis_name, output_path=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                        msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                        tandem_scoring_model=None, oxonium_threshold=0.15, save_intermediate_results=None,
                        processes=4, workload_size=500, mass_shifts=None, export=None,
                        use_peptide_mass_filter=False, maximum_mass=float('inf'),
                        decoy_database_connection=None, fdr_correction='auto',
                        isotope_probing_range=3):
    """Identify glycopeptide sequences from processed LC-MS/MS data
    """
    if output_path is None:
        output_path = make_analysis_output_path("glycopeptide")
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

    mass_shifts = [validate_mass_shift(mass_shift, multiplicity)
                   for mass_shift, multiplicity in mass_shifts]
    expanded = []
    expanded = MzMLGlycanChromatogramAnalyzer.expand_mass_shifts(
        dict(mass_shifts), crossproduct=False)
    mass_shifts = expanded

    if analysis_name is None:
        analysis_name = "%s @ %s" % (sample_run.name, hypothesis.name)

    analysis_name = validate_analysis_name(
        context, database_connection.session, analysis_name)

    click.secho("Preparing analysis of %s by %s" % (
        sample_run.name, hypothesis.name), fg='cyan')

    if decoy_database_connection is None:
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
            n_processes=processes,
            spectra_batch_size=workload_size,
            mass_shifts=mass_shifts,
            use_peptide_mass_filter=use_peptide_mass_filter,
            maximum_mass=maximum_mass,
            probing_range_for_missing_precursors=isotope_probing_range)
    else:
        analyzer = MzMLComparisonGlycopeptideLCMSMSAnalyzer(
            database_connection._original_connection,
            decoy_database_connection,
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
            n_processes=processes,
            spectra_batch_size=workload_size,
            mass_shifts=mass_shifts,
            use_peptide_mass_filter=use_peptide_mass_filter,
            maximum_mass=maximum_mass,
            use_decoy_correction_threshold=fdr_correction,
            probing_range_for_missing_precursors=isotope_probing_range)
    analyzer.display_header()
    gps, unassigned, target_decoy_set = analyzer.start()
    if save_intermediate_results is not None:
        analyzer.log("Saving Intermediate Results")
        with open(save_intermediate_results, 'wb') as handle:
            pickle.dump((target_decoy_set, gps), handle)
    if export:
        for export_type in set(export):
            click.echo(fmt_msg("Handling Export: %s" % (export_type,)))
            if export_type == 'csv':
                from glycan_profiling.cli.export import glycopeptide_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycopeptides.csv" % (base,)
                context.invoke(
                    glycopeptide_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'psm-csv':
                from glycan_profiling.cli.export import glycopeptide_spectrum_matches
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycopeptide-spectrum-matches.csv" % (base,)
                context.invoke(
                    glycopeptide_spectrum_matches,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'html':
                from glycan_profiling.cli.export import glycopeptide_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-report.html" % (base,)
                context.invoke(
                    glycopeptide_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path,
                    report=True)


class RegularizationParameterType(click.ParamType):
    name = "\"grid\" or NUMBER > 0 or [grid|NUMBER]/NUMBER"
    seperator = '/'

    def convert(self, value, param, ctx):
        sep_count = value.count(self.seperator)
        if sep_count > 1:
            self.fail("regularization parameter split \"%s\" cannot be "
                      "used more than once" % self.seperator)
        elif sep_count == 1:
            parts = value.split(self.seperator)
            parts = (self.convert(parts[0], param, ctx),
                     self.convert(parts[1], param, ctx))
            if parts[1] == 'grid':
                self.fail("The second regularization parameter cannot be \"grid\"")
            return parts

        value = value.strip().lower()
        if value == 'grid':
            return LaplacianRegularizedChromatogramProcessor.GRID_SEARCH
        else:
            try:
                value = float(value)
                if value < 0:
                    self.fail("regularization parameter must be either \"grid\" or"
                              " a non-negative number between 0 and 1")
                return value
            except ValueError:
                self.fail("regularization parameter must be either \"grid\" or"
                          " a number between 0 and 1")


@analyze.command("search-glycan", short_help=('Search processed data for'
                                              ' glycan compositions'))
@click.pass_context
@database_connection
@sample_path
@hypothesis_identifier("glycan")
@click.option("-m", "--mass-error-tolerance", type=RelativeMassErrorParam(), default=1e-5,
              help=("Mass accuracy constraint, in parts-per-million "
                    "error, for matching."))
@click.option("-mn", "--msn-mass-error-tolerance", type=RelativeMassErrorParam(), default=2e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^n ions.")
@click.option("-g", "--grouping-error-tolerance", type=RelativeMassErrorParam(), default=1.5e-5,
              help=("Mass accuracy constraint, in parts-per-million error, for"
                    " grouping chromatograms."))
@click.option("-n", "--analysis-name", default=None,
              help='Name for analysis to be performed.')
@click.option("-a", "--mass_shift", 'mass_shifts', multiple=True, nargs=2,
              help=("Adducts to consider. Specify name or formula, and a"
                    " multiplicity."))
@click.option("--mass_shift-combination-limit", type=int, default=8, help=(
    "Maximum number of mass_shift combinations to consider"))
@click.option("-d", "--minimum-mass", default=500., type=float,
              help="The minimum mass to consider signal at.")
@click.option("-o", "--output-path", default=None, help=(
              "Path to write resulting analysis to."))
@click.option("--interact", is_flag=True, cls=HiddenOption)
@click.option("-f", "--ms1-scoring-feature", "scoring_model_features", multiple=True,
              type=click.Choice(ms1_model_features.keys(lambda x: x.replace("_", "-"))),
              help="Additional features to include in evaluating chromatograms")
@click.option("-r", "--regularize", type=RegularizationParameterType(),
              help=("Apply Laplacian regularization with either a"
                    " specified weight or \"grid\" to grid search, or a pair of values"
                    " separated by a / to specify a weight or grid search "
                    " for model fitting and a separate weight for scoring"))
@click.option("-w", "--regularization-model-path", type=click.Path(exists=True),
              default=None,
              help="Path to a file containing a neighborhood model for regularization")
@click.option("-k", "--network-path", type=click.Path(exists=True), default=None,
              help=("Path to a file containing the glycan composition network"
                    " and neighborhood rules"))
@click.option("-t", "--delta-rt", default=0.5, type=float,
              help='The maximum time between observed data points before splitting features')
@click.option("--export", type=click.Choice(
              ['csv', 'glycan-list', 'html', "model"]), multiple=True,
              help="export command to after search is complete")
@click.option('-s', '--require-msms-signature', type=float, default=0.0,
              help="Minimum oxonium ion signature required in MS/MS scans to include.")
@processes_option
def search_glycan(context, database_connection, sample_path,
                  hypothesis_identifier,
                  analysis_name, mass_shifts, grouping_error_tolerance=1.5e-5,
                  mass_error_tolerance=1e-5, minimum_mass=500.,
                  scoring_model=None, regularize=None, regularization_model_path=None,
                  network_path=None,
                  output_path=None, scoring_model_features=None,
                  delta_rt=0.5, export=None, interact=False,
                  require_msms_signature=0.0, msn_mass_error_tolerance=2e-5,
                  mass_shift_combination_limit=None,
                  processes=4):
    """Identify glycan compositions from preprocessed LC-MS data, stored in mzML
    format.
    """
    if output_path is None and not interact:
        output_path = make_analysis_output_path("glycan")
    if scoring_model is None:
        scoring_model = GeneralScorer

    if mass_shift_combination_limit is None:
        mass_shift_combination_limit = 8

    if scoring_model_features:
        for feature in scoring_model_features:
            scoring_model.add_feature(validate_ms1_feature_name(feature))

    if regularization_model_path is not None:
        with open(regularization_model_path, 'r') as mod_file:
            regularization_model = GridPointSolution.load(mod_file)
    else:
        regularization_model = None

    if network_path is not None:
        with open(network_path, 'r') as netfile:
            network = GraphReader(netfile).network
    else:
        network = None

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

    mass_shifts = [validate_mass_shift(mass_shift, multiplicity)
                   for mass_shift, multiplicity in mass_shifts]
    expanded = []
    expanded = MzMLGlycanChromatogramAnalyzer.expand_mass_shifts(dict(mass_shifts), limit=mass_shift_combination_limit)
    mass_shifts = expanded

    click.secho("Preparing analysis of %s by %s" %
                (sample_run.name, hypothesis.name), fg='cyan')

    analyzer = MzMLGlycanChromatogramAnalyzer(
        database_connection._original_connection, hypothesis.id,
        sample_path=sample_path, output_path=output_path, mass_shifts=mass_shifts,
        mass_error_tolerance=mass_error_tolerance,
        msn_mass_error_tolerance=msn_mass_error_tolerance,
        grouping_error_tolerance=grouping_error_tolerance,
        scoring_model=scoring_model,
        minimum_mass=minimum_mass,
        regularize=regularize,
        regularization_model=regularization_model,
        network=network,
        analysis_name=analysis_name,
        delta_rt=delta_rt,
        require_msms_signature=require_msms_signature,
        n_processes=processes)
    analyzer.display_header()
    analyzer.start()
    if interact:
        try:
            import IPython
            click.secho(fmt_msg("Beginning Interactive Session..."), fg='cyan')
            IPython.embed()
        except ImportError:
            click.secho(fmt_msg("Interactive Session Not Supported"), fg='red')
    if export:
        for export_type in set(export):
            click.echo(fmt_msg("Handling Export: %s" % (export_type,)))
            if export_type == 'csv':
                from glycan_profiling.cli.export import glycan_composition_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycan-chromatograms.csv" % (base,)
                context.invoke(
                    glycan_composition_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'html':
                from glycan_profiling.cli.export import glycan_composition_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-report.html" % (base,)
                context.invoke(
                    glycan_composition_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path,
                    report=True)
            elif export_type == 'glycan-list':
                from glycan_profiling.cli.export import glycan_hypothesis
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycan-chromatograms.csv" % (base,)
                context.invoke(
                    glycan_hypothesis,
                    database_connection=output_path,
                    hypothesis_identifier=analyzer.hypothesis_id,
                    output_path=export_path,
                    importable=True)
            elif export_type == "model":
                base = os.path.splitext(output_path)[0]
                export_path = "%s-regularization-parameters.txt" % (base,)
                params = analyzer.analysis.parameters.get("network_parameters")
                if params is None:
                    click.secho("No parameters were fitted, skipping \"model\"", fg='red')
                else:
                    with open(export_path, 'w') as fp:
                        params.dump(fp)
            else:
                click.secho("Unrecognized Export: %s" (export_type,), fg='yellow')
