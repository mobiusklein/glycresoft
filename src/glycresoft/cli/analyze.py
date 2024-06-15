import os
from uuid import uuid4

import dill as pickle

import click
from glycresoft import serialize
from glycresoft.cli.export import analysis_identifier_arg

from glycresoft.serialize import (
    DatabaseBoundOperation,
    GlycanHypothesis,
    GlycopeptideHypothesis,
    Analysis)

from glycresoft.profiler import (
    MzMLGlycopeptideLCMSMSAnalyzer,
    MzMLComparisonGlycopeptideLCMSMSAnalyzer,
    MultipartGlycopeptideLCMSMSAnalyzer,
    GlycopeptideFDREstimationStrategy,
    MzMLGlycanChromatogramAnalyzer,
    LaplacianRegularizedChromatogramProcessor,
    ProcessedMSFileLoader,
    ChromatogramSummarizer)

from glycresoft.composition_distribution_model import site_model
from glycresoft.scoring.elution_time_grouping import GlycopeptideChromatogramProxy, GlycopeptideElutionTimeModeler
from glycresoft.serialize.analysis import AnalysisTypeEnum
from glycresoft.tandem.glycopeptide.scoring import CoverageWeightedBinomialScorer
from glycresoft.composition_distribution_model import GridPointSolution
from glycresoft.database.composition_network import GraphReader

from glycresoft.models import GeneralScorer
from glycresoft.task import fmt_msg

from .base import (
    cli, HiddenOption, processes_option)

from .validators import (
    validate_analysis_name,
    validate_mass_shift, validate_glycopeptide_tandem_scoring_function,
    glycopeptide_tandem_scoring_functions,
    get_by_name_or_id,
    validate_ms1_feature_name,
    ms1_model_features,
    RelativeMassErrorParam,
    DatabaseConnectionParam,
    SmallSampleFDRCorrectionParam, ChoiceOrURI)


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


def database_connection_arg(arg_name='database-connection'):
    def database_connection_arg(fn):
        arg = click.argument(
            arg_name,
            type=DatabaseConnectionParam(exists=True),
            doc_help=(
                "A connection URI for a database, or a path on the file system"))
        return arg(fn)
    return database_connection_arg


def hypothesis_identifier_arg(hypothesis_type, arg_name='hypothesis-identifier'):
    def wrapper(fn):
        arg = click.argument(arg_name, doc_help=(
            "The ID number or name of the %s hypothesis to use" % (hypothesis_type,)))
        return arg(fn)
    return wrapper


def hypothesis_identifier_arg_option(hypothesis_type, *args, **kwargs):
    def wrapper(fn):
        arg = click.option(*args, help=(
            "The ID number or name of the %s hypothesis to use" % (hypothesis_type,)), **kwargs)
        return arg(fn)
    return wrapper


def sample_path_arg(fn):
    arg = click.argument(
        "sample-path",
        type=click.Path(exists=True, dir_okay=False, file_okay=True),
        doc_help=(
            "The path to the deconvoluted sample file"))
    return arg(fn)


@analyze.command("search-glycopeptide", short_help='Search preprocessed data for glycopeptide sequences')
@click.pass_context
@database_connection_arg()
@sample_path_arg
@hypothesis_identifier_arg("glycopeptide")
@click.option("-m", "--mass-error-tolerance", type=RelativeMassErrorParam(), default=1e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^1 ions.")
@click.option("-mn", "--msn-mass-error-tolerance", type=RelativeMassErrorParam(), default=2e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^n ions.")
@click.option("-g", "--grouping-error-tolerance", type=RelativeMassErrorParam(), default=1.5e-5,
              help="Mass accuracy constraint, in parts-per-million error, for grouping chromatograms.")
@click.option("-n", "--analysis-name", default=None, help='Name for analysis to be performed.')
@click.option("-q", "--psm-fdr-threshold", default=0.05, type=float,
              help='Minimum FDR Threshold to use for filtering GPSMs when selecting identified glycopeptides')
@click.option("-s", "--tandem-scoring-model", default='coverage_weighted_binomial',
              type=ChoiceOrURI(glycopeptide_tandem_scoring_functions.keys()),
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
@click.option("--export", type=click.Choice(['csv', 'html', 'psm-csv']), multiple=True,
              help="export command to after search is complete")
@click.option("-o", "--output-path", default=None, type=click.Path(writable=True),
              help=("Path to write resulting analysis to."), required=True)
@click.option("-w", "--workload-size", default=500, type=int, help="Number of spectra to process at once")
@click.option("--save-intermediate-results", default=None, type=click.Path(), required=False,
              help='Save intermediate spectrum matches to a file')
@click.option("--maximum-mass", default=float('inf'), type=float)
@click.option("-D", "--decoy-database-connection", default=None, help=(
    "Provide an alternative hypothesis to draw decoy glycopeptides from instead of the simpler reversed-peptide "
    "decoy. This is especially necessary when the stub peptide+Y ions account for a large fraction of MS2 signal."))
@click.option("-G", "--permute-decoy-glycan-fragments", is_flag=True, default=False, help=(
    "Whether or not to permute decoy glycopeptides' peptide+Y ions. The intact mass, peptide, "
    "and peptide+Y1 ions are unchanged."))
@click.option("--fdr-correction", default='auto', type=SmallSampleFDRCorrectionParam(),
              help=("Whether to attempt to correct for small sample size for target-decoy analysis."),
              cls=HiddenOption)
@click.option("--isotope-probing-range", type=int, default=3, help=(
    "The maximum number of isotopic peak errors to allow when searching for untrusted precursor masses"))
@click.option("-R", "--rare-signatures", is_flag=True, default=False,
              help="Look for rare signature ions when scoring glycan oxonium signature")
@click.option("--retention-time-modeling/--no-retention-time-modeling", is_flag=True, default=True,
              help=("Whether or not to model relative retention time to correct for common glycan"
                    " composition errors."))
def search_glycopeptide(context, database_connection, sample_path, hypothesis_identifier,
                        analysis_name, output_path=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                        msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                        tandem_scoring_model=None, oxonium_threshold=0.15, save_intermediate_results=None,
                        processes=4, workload_size=500, mass_shifts=None, export=None,
                        use_peptide_mass_filter=False, maximum_mass=float('inf'),
                        decoy_database_connection=None, fdr_correction='auto',
                        isotope_probing_range=3,
                        permute_decoy_glycan_fragments=False,
                        rare_signatures=False,
                        retention_time_modeling=True):
    """Identify glycopeptide sequences from processed LC-MS/MS data. This algorithm requires a fully materialized
    cross-product database (the default), and uses a reverse-peptide decoy by default, evaluated on the total score.

    For a search algorithm that applies separate FDR control on the peptide and the glycan, see
    :ref:`search-glycopeptide-multipart`
    """
    if tandem_scoring_model is None:
        tandem_scoring_model = CoverageWeightedBinomialScorer
    if os.path.exists(output_path):
        click.secho("Output path '%s' exists, removing..." % output_path, fg='yellow')
        os.remove(output_path)
    database_connection = DatabaseBoundOperation(database_connection)
    ms_data = ProcessedMSFileLoader(sample_path, use_index=False)
    sample_run = ms_data.sample_run

    try:
        hypothesis = get_by_name_or_id(
            database_connection, GlycopeptideHypothesis, hypothesis_identifier)
    except Exception:
        click.secho("Could not locate a Glycopeptide Hypothesis with identifier %r" %
                    hypothesis_identifier, fg='yellow')
        raise click.Abort()

    tandem_scoring_model, evaluation_kwargs = validate_glycopeptide_tandem_scoring_function(
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
            spectrum_batch_size=workload_size,
            mass_shifts=mass_shifts,
            use_peptide_mass_filter=use_peptide_mass_filter,
            maximum_mass=maximum_mass,
            probing_range_for_missing_precursors=isotope_probing_range,
            permute_decoy_glycans=permute_decoy_glycan_fragments,
            rare_signatures=rare_signatures,
            model_retention_time=retention_time_modeling,
            evaluation_kwargs=evaluation_kwargs
        )
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
            spectrum_batch_size=workload_size,
            mass_shifts=mass_shifts,
            use_peptide_mass_filter=use_peptide_mass_filter,
            maximum_mass=maximum_mass,
            use_decoy_correction_threshold=fdr_correction,
            probing_range_for_missing_precursors=isotope_probing_range,
            permute_decoy_glycans=permute_decoy_glycan_fragments,
            rare_signatures=rare_signatures,
            model_retention_time=retention_time_modeling,
            evaluation_kwargs=evaluation_kwargs
        )
    analyzer.display_header()
    result = analyzer.start()
    gps, unassigned, target_decoy_set = result[:3]
    if save_intermediate_results is not None:
        analyzer.log("Saving Intermediate Results")
        with open(save_intermediate_results, 'wb') as handle:
            pickle.dump((target_decoy_set, gps), handle)
    del gps
    del unassigned
    del target_decoy_set
    if export:
        for export_type in set(export):
            click.echo(fmt_msg("Handling Export: %s" % (export_type,)))
            if export_type == 'csv':
                from glycresoft.cli.export import glycopeptide_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycopeptides.csv" % (base,)
                context.invoke(
                    glycopeptide_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'psm-csv':
                from glycresoft.cli.export import glycopeptide_spectrum_matches
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycopeptide-spectrum-matches.csv" % (base,)
                context.invoke(
                    glycopeptide_spectrum_matches,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'html':
                from glycresoft.cli.export import glycopeptide_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-report.html" % (base,)
                context.invoke(
                    glycopeptide_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path,
                    report=True)


@analyze.command("search-glycopeptide-multipart", short_help=(
    'Search preprocessed data for glycopeptide sequences scored for both peptide and glycan components'))
@click.pass_context
@database_connection_arg("database-connection")
@database_connection_arg("decoy-database-connection")
@sample_path_arg
@hypothesis_identifier_arg_option("glycopeptide", '-T', '--target-hypothesis-identifier', default=1)
@hypothesis_identifier_arg_option("glycopeptide", '-D', "--decoy-hypothesis-identifier", default=1)
@click.option("-M", "--memory-database-index", is_flag=True,
              help=("Whether to load the entire peptide database into memory during spectrum mapping. "
                    "Uses more memory but substantially accelerates the process"), default=False)
@click.option("-m", "--mass-error-tolerance", type=RelativeMassErrorParam(), default=1e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^1 ions.")
@click.option("-mn", "--msn-mass-error-tolerance", type=RelativeMassErrorParam(), default=2e-5,
              help="Mass accuracy constraint, in parts-per-million error, for matching MS^n ions.")
@click.option("-g", "--grouping-error-tolerance", type=RelativeMassErrorParam(), default=1.5e-5,
              help="Mass accuracy constraint, in parts-per-million error, for grouping chromatograms.")
@click.option("-n", "--analysis-name", default=None, help='Name for analysis to be performed.')
@click.option("-q", "--psm-fdr-threshold", default=0.05, type=float,
              help='Minimum FDR Threshold to use for filtering GPSMs when selecting identified glycopeptides')
@click.option("-f", "--fdr-estimation-strategy", type=click.Choice(['joint', 'peptide', 'glycan', 'any']),
              default='joint',
              help=("The FDR estimation strategy to use. The joint estimate uses both peptide and glycan scores, "
                    "peptide uses only peptide scores, glycan uses only glycan scores, and any uses the smallest "
                    "FDR of the joint, peptide, and glycan estiamtes."))
@click.option("-s", "--tandem-scoring-model", default='log_intensity',
              type=ChoiceOrURI(["log_intensity",
                                "simple",
                                "penalized_log_intensty",
                                "log_intensity_v3"]),
              help="Select a scoring function to use for evaluating glycopeptide-spectrum matches")
@click.option("-y", "--glycan-score-threshold", default=1.0, type=float,
              help="The minimum glycan score required to consider a peptide mass")
@click.option("-a", "--adduct", 'mass_shifts', multiple=True, nargs=2,
              help=("Adducts to consider. Specify name or formula, and a"
                    " multiplicity."))
@processes_option
@click.option("--export", type=click.Choice(['csv', 'html', 'psm-csv']), multiple=True,
              help="export command to after search is complete")
@click.option("-o", "--output-path", default=None, type=click.Path(writable=True),
              help=("Path to write resulting analysis to."), required=True)
@click.option("-w", "--workload-size", default=100, type=int, help="Number of spectra to process at once")
@click.option("-R", "--rare-signatures", is_flag=True, default=False,
              help="Look for rare signature ions when scoring glycan oxonium signature")
@click.option("--retention-time-modeling/--no-retention-time-modeling", is_flag=True, default=True,
              help=("Whether or not to model relative retention time to correct for common glycan"
                    " composition errors."))
@click.option("--save-intermediate-results", default=None, type=click.Path(), required=False,
              help='Save intermediate spectrum matches to a file', cls=HiddenOption)
@click.option("--maximum-mass", default=float('inf'), type=float, cls=HiddenOption)
@click.option("--isotope-probing-range", type=int, default=3, help=(
    "The maximum number of isotopic peak errors to allow when searching for untrusted precursor masses"))
@click.option("-S", "--glycoproteome-smoothing-model", type=click.Path(readable=True), help=(
    "Path to a glycoproteome site-specific glycome model"), default=None)
@click.option("-x", "--oxonium-threshold", default=0.05, type=float,
              help=('Minimum HexNAc-derived oxonium ion abundance '
                    'ratio to filter MS/MS scans. Defaults to 0.05.'))
@click.option("-P", "--peptide-masses-per-scan", type=int, default=60,
              help="The maximum number of peptide masses to consider per scan")
def search_glycopeptide_multipart(context, database_connection,
                                  decoy_database_connection,
                                  sample_path,
                                  target_hypothesis_identifier=1,
                                  decoy_hypothesis_identifier=1,
                                  analysis_name=None,
                                  output_path=None,
                                  grouping_error_tolerance=1.5e-5,
                                  mass_error_tolerance=1e-5,
                                  msn_mass_error_tolerance=2e-5,
                                  psm_fdr_threshold=0.05,
                                  peak_shape_scoring_model=None,
                                  tandem_scoring_model=None,
                                  glycan_score_threshold=1.0,
                                  memory_database_index=False,
                                  save_intermediate_results=None,
                                  processes=4,
                                  workload_size=100,
                                  mass_shifts=None,
                                  export=None,
                                  maximum_mass=float('inf'),
                                  isotope_probing_range=3,
                                  fdr_estimation_strategy=None,
                                  glycoproteome_smoothing_model=None,
                                  rare_signatures=False,
                                  retention_time_modeling=True,
                                  oxonium_threshold: float=0.05,
                                  peptide_masses_per_scan: int=60):
    """
    Search preprocessed data for glycopeptide sequences scored for both peptide and glycan components.

    This search strategy requires an explicit decoy database, like one created with the `--reverse`
    flag from `glycopeptide-fa`.
    """
    if fdr_estimation_strategy is None:
        fdr_estimation_strategy = GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture
    else:
        fdr_estimation_strategy = GlycopeptideFDREstimationStrategy[fdr_estimation_strategy]
    if tandem_scoring_model is None:
        tandem_scoring_model = "log_intensity"
    if os.path.exists(output_path):
        click.secho("Output path '%s' exists, removing..." % output_path, fg='yellow')
        os.remove(output_path)
    database_connection = DatabaseBoundOperation(database_connection)
    decoy_database_connection = DatabaseBoundOperation(decoy_database_connection)
    ms_data = ProcessedMSFileLoader(sample_path, use_index=False)
    sample_run = ms_data.sample_run

    try:
        target_hypothesis = get_by_name_or_id(
            database_connection, GlycopeptideHypothesis, target_hypothesis_identifier)
    except Exception:
        click.secho("Could not locate Target Glycopeptide Hypothesis with identifier %r" %
                    target_hypothesis_identifier, fg='yellow')
        raise click.Abort()

    try:
        decoy_hypothesis = get_by_name_or_id(
            decoy_database_connection, GlycopeptideHypothesis, decoy_hypothesis_identifier)
    except Exception:
        click.secho("Could not locate Decoy Glycopeptide Hypothesis with identifier %r" %
                    decoy_hypothesis_identifier, fg='yellow')
        raise click.Abort()

    tandem_scoring_model, evaluation_kwargs = validate_glycopeptide_tandem_scoring_function(
        context, tandem_scoring_model)

    mass_shifts = [validate_mass_shift(mass_shift, multiplicity)
                   for mass_shift, multiplicity in mass_shifts]
    expanded = []
    expanded = MzMLGlycanChromatogramAnalyzer.expand_mass_shifts(
        dict(mass_shifts), crossproduct=False)
    mass_shifts = expanded

    if analysis_name is None:
        analysis_name = "%s @ %s" % (sample_run.name, target_hypothesis.name)

    analysis_name = validate_analysis_name(
        context, database_connection.session, analysis_name)

    click.secho("Preparing analysis of %s by %s" % (
        sample_run.name, target_hypothesis.name), fg='cyan')
    analyzer = MultipartGlycopeptideLCMSMSAnalyzer(
        database_connection._original_connection,
        decoy_database_connection._original_connection,
        target_hypothesis.id,
        decoy_hypothesis.id,
        sample_path,
        output_path,
        analysis_name=analysis_name,
        grouping_error_tolerance=grouping_error_tolerance,
        mass_error_tolerance=mass_error_tolerance,
        msn_mass_error_tolerance=msn_mass_error_tolerance,
        psm_fdr_threshold=psm_fdr_threshold,
        tandem_scoring_model=tandem_scoring_model,
        glycan_score_threshold=glycan_score_threshold,
        mass_shifts=mass_shifts,
        n_processes=processes,
        spectrum_batch_size=workload_size,
        maximum_mass=maximum_mass,
        probing_range_for_missing_precursors=isotope_probing_range,
        use_memory_database=memory_database_index,
        fdr_estimation_strategy=fdr_estimation_strategy,
        glycosylation_site_models_path=glycoproteome_smoothing_model,
        fragile_fucose=False,
        rare_signatures=rare_signatures,
        evaluation_kwargs=evaluation_kwargs,
        model_retention_time=retention_time_modeling,
        oxonium_threshold=oxonium_threshold,
        peptide_masses_per_scan=peptide_masses_per_scan
    )
    analyzer.display_header()
    result = analyzer.start()
    gps, unassigned, target_decoy_set = result[:3]
    if save_intermediate_results is not None:
        analyzer.log("Saving Intermediate Results")
        with open(save_intermediate_results, 'wb') as handle:
            pickle.dump((target_decoy_set, gps), handle)
    del gps
    del unassigned
    del target_decoy_set
    if export:
        for export_type in set(export):
            click.echo(fmt_msg("Handling Export: %s" % (export_type,)))
            if export_type == 'csv':
                from glycresoft.cli.export import glycopeptide_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycopeptides.csv" % (base,)
                context.invoke(
                    glycopeptide_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'psm-csv':
                from glycresoft.cli.export import glycopeptide_spectrum_matches
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycopeptide-spectrum-matches.csv" % (base,)
                context.invoke(
                    glycopeptide_spectrum_matches,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'html':
                from glycresoft.cli.export import glycopeptide_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-report.html" % (base,)
                context.invoke(
                    glycopeptide_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path,
                    report=True)



@analyze.command("fit-glycoproteome-smoothing-model", short_help=(
    "Fit a site-specific glycome network smoothing model for each site in the glycoproteome"))
@click.pass_context
@processes_option
@click.option("-i", "--analysis-path", nargs=2, multiple=True, required=True)
@click.option("-o", "--output-path", type=click.Path(writable=True), required=True)
@click.option('-q', '--fdr-threshold', type=float, default=0.05,
              help="The FDR threshold to apply when selecting identified glycopeptides")
@click.option("-P", "--glycopeptide-hypothesis", type=(DatabaseConnectionParam(exists=True), str))
@click.option("-g", "--glycan-hypothesis", type=(DatabaseConnectionParam(exists=True), str))
@click.option("-u", "--unobserved-penalty-scale", type=float, default=1.0, required=False,
              help="A penalty to scale unobserved-but-suggested glycans by. Defaults to 1.0, no penalty.")
@click.option("-a", "--smoothing-limit", type=float, default=0.2,
              help="An upper bound on the network smoothness to use when estimating the posterior probability.")
@click.option("-r", "--require-multiple-observations/--no-require-multiple-observations", is_flag=True, default=False,
              help=(
                  "Require a glycan/glycosite combination be observed in multiple samples to treat it as real."
                  " Defaults to False."))
@click.option("-w", "--network-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False,
              default=None,
              help=("The path to a text file defining the glycan network and its neighborhoods, as produced by "
                    "`glycresfoft build-hypothesis glycan-network`, otherwise the default human N-glycan network "
                    "will be used with the glycans defined in `-g`."))
@click.option("-D / -ND", "--include-decoys / --exclude-decoys", is_flag=True, default=True,
              help="Include decoy glycans in the network.", cls=HiddenOption)
def fit_glycoproteome_model(context, analysis_path, output_path, glycopeptide_hypothesis, glycan_hypothesis,
                            processes=4, unobserved_penalty_scale=None, smoothing_limit=0.2,
                            require_multiple_observations=True, fdr_threshold=0.05,
                            network_path=None, include_decoys=True):
    analysis_path_set = analysis_path
    analysis_path_set_transformed = []
    if require_multiple_observations and len(analysis_path_set) == 1:
        click.secho("Requested multiple observations required but only one analysis provided"
                    " discarding multiple observation requirement.", fg='yellow')
    click.echo(f"Collecting {len(analysis_path_set)} analyses")
    for analysis_path, analysis_id in analysis_path_set:
        database_connection = DatabaseBoundOperation(analysis_path)
        try:
            click.echo("Checking analysis %s:%s" %
                       (analysis_path, analysis_id))
            _analysis = get_by_name_or_id(database_connection, Analysis, analysis_id)
        except Exception:
            click.secho("Could not locate an Analysis in %r with identifier %r" %
                        (analysis_path, analysis_id), fg='yellow')
            raise click.Abort()
        # analysis_path_set_transformed.append((analysis_path, analysis.id))
        analysis_path_set_transformed.append((analysis_path, int(analysis_id)))
    analysis_path_set = analysis_path_set_transformed

    if network_path is not None:
        click.echo("Loading graph from \"%s\"" % (network_path, ))
        with open(network_path, 'r') as netfile:
            network = GraphReader(netfile).network
    else:
        network = None

    glycopeptide_database_connection_path, hypothesis_identifier = glycopeptide_hypothesis
    glycopeptide_database_connection = DatabaseBoundOperation(glycopeptide_database_connection_path)
    try:
        glycopeptide_hypothesis = get_by_name_or_id(
            glycopeptide_database_connection, GlycopeptideHypothesis, hypothesis_identifier)
    except Exception:
        click.secho("Could not locate a Glycopeptide Hypothesis with identifier %r" %
                    hypothesis_identifier, fg='yellow')
        raise click.Abort()

    glycan_database_connection_path, hypothesis_identifier = glycan_hypothesis
    glycan_database_connection = DatabaseBoundOperation(
        glycan_database_connection_path)
    try:
        glycan_hypothesis = get_by_name_or_id(
            glycan_database_connection, GlycanHypothesis, hypothesis_identifier)
    except Exception:
        click.secho("Could not locate a Glycopeptide Hypothesis with identifier %r" %
                    hypothesis_identifier, fg='yellow')
        raise click.Abort()
    workflow = site_model.GlycoproteinSiteModelBuildingWorkflow.from_paths(
        analysis_path_set, glycopeptide_database_connection_path, glycopeptide_hypothesis.id,
        glycan_database_connection_path, glycan_hypothesis.id, unobserved_penalty_scale, smoothing_limit,
        require_multiple_observations, output_path=output_path, n_threads=processes,
        q_value_threshold=fdr_threshold, network=network, include_decoy_glycans=False)
    workflow.start()


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
@database_connection_arg()
@sample_path_arg
@hypothesis_identifier_arg("glycan")
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
@click.option("-o", "--output-path", default=None, help=("Path to write resulting analysis to."),
              required=True)
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
@click.option("--export", type=click.Choice(['csv', 'glycan-list', 'html', "model"]), multiple=True,
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
    if os.path.exists(output_path):
        click.secho("Output path '%s' exists, removing..." % output_path, fg='yellow')
        os.remove(output_path)

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
    ms_data = ProcessedMSFileLoader(sample_path, use_index=False)
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
                from glycresoft.cli.export import glycan_composition_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-glycan-chromatograms.csv" % (base,)
                context.invoke(
                    glycan_composition_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path)
            elif export_type == 'html':
                from glycresoft.cli.export import glycan_composition_identification
                base = os.path.splitext(output_path)[0]
                export_path = "%s-report.html" % (base,)
                context.invoke(
                    glycan_composition_identification,
                    database_connection=output_path,
                    analysis_identifier=analyzer.analysis_id,
                    output_path=export_path,
                    report=True)
            elif export_type == 'glycan-list':
                from glycresoft.cli.export import glycan_hypothesis
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
                click.secho("Unrecognized Export: %s" % (export_type,), fg='yellow')


@analyze.command("summarize-chromatograms",
                 short_help="Simply summarize coherent chromatograms by mass and signal with time boundaries")
@click.pass_context
@sample_path_arg
@click.option("-o", "--output-path", default=None, type=click.Path(writable=True),
              help=("Path to write resulting analysis to as a CSV"))
@click.option("-e", "--evaluate", is_flag=True,
              help=("Should all chromatograms be evaluated. Can greatly increase runtime."))
def summarize_chromatograms(context, sample_path, output_path, evaluate=False):
    task = ChromatogramSummarizer(sample_path, evaluate=evaluate)
    chromatograms, summary_chromatograms = task.start()
    if output_path is None:
        output_path = os.path.splitext(sample_path)[0] + '.chromatograms.csv'
    from glycresoft.output import (
        SimpleChromatogramCSVSerializer, SimpleScoredChromatogramCSVSerializer)
    click.echo("Writing chromatogram summaries to %s" % (output_path, ))
    with open(output_path, 'wb') as fh:
        if evaluate:
            writer = SimpleScoredChromatogramCSVSerializer(fh, chromatograms)
        else:
            writer = SimpleChromatogramCSVSerializer(fh, chromatograms)
        writer.run()



@analyze.group(short_help="Model and predict the retention time of glycopeptides")
def retention_time():
    pass


@retention_time.command("fit-glycopeptide-retention-time", short_help="Model the retention time of glycopeptides")
@click.pass_context
@click.argument("chromatogram-csv-path", type=click.Path(readable=True))
@click.argument("output-path", type=click.Path(writable=True))
@click.option("-t", "--test-chromatogram-csv-path", type=click.Path(readable=True),
              help=("Path to glycopeptide CSV chromatograms to evaluate with the model, but not fit on"))
@click.option("-p", "--prefer-joint-model", is_flag=True,
              help="Prefer the joint model over the peptide-specific ones")
@click.option('-m', '--minimum-observations-for-specific-model', type=int, default=20,
              help="The minimum number of observations required to fit a peptide sequence-specific model")
@click.option('-r', '--use-retention-time-normalization', is_flag=True)
def glycopeptide_retention_time_fit(context, chromatogram_csv_path, output_path=None,
                                    test_chromatogram_csv_path=None, prefer_joint_model=False,
                                    minimum_observations_for_specific_model=20, use_retention_time_normalization=False):
    '''Fit a retention time model to glycopeptide chromatogram observations, and evaluate the chromatographic apex
    retention of each glycopeptide based upon the model fit. For large enough groups, try to fit a model to just
    that group.
    '''
    with open(chromatogram_csv_path, 'rt') as fh:
        chromatograms = GlycopeptideChromatogramProxy.from_csv(fh)
    if test_chromatogram_csv_path is not None:
        with open(test_chromatogram_csv_path, 'rt') as fh:
            test_chromatograms = GlycopeptideChromatogramProxy.from_csv(fh)
    else:
        test_chromatograms = None
    modeler = GlycopeptideElutionTimeModeler(
        chromatograms, test_chromatograms=test_chromatograms,
        prefer_joint_model=prefer_joint_model,
        use_retention_time_normalization=use_retention_time_normalization,
        minimum_observations_for_specific_model=minimum_observations_for_specific_model)
    modeler.run()
    if output_path:
        modeler.write(output_path)


@retention_time.command(
    "evaluate-glycopeptide-retention-time",
    short_help="Predict and score the retention time of glycopeptides using an existing model")
@click.pass_context
@click.argument("model-path", type=click.Path(readable=True))
@click.argument("chromatogram-csv-path", type=click.Path(readable=True))
@click.argument("output-path", type=click.Path(writable=True))
def glycopeptide_retention_time_predict(context, model_path, chromatogram_csv_path, output_path):
    """Predict and evaluate the retention time of glycopeptides using an existing model.

    The model file should have the extension .pkl and have been produced by the same version of GlycReSoft.
    """
    with open(model_path, 'rb') as fh:
        modeler = pickle.load(fh)
    with open(chromatogram_csv_path, 'rt') as fh:
        chromatograms = GlycopeptideChromatogramProxy.from_csv(fh)
    modeler.evaluate(chromatograms)
    with open(output_path, 'wt') as fh:
        GlycopeptideChromatogramProxy.to_csv(chromatograms, fh)


@analyze.command("summarize-analysis", short_help="Briefly summarize the results of an analysis output file")
@click.pass_context
@database_connection_arg()
@analysis_identifier_arg("glycopeptide")
def summarize_analysis(context: click.Context, database_connection, analysis_identifier, verbose=False):
    database_connection = DatabaseBoundOperation(database_connection)
    session = database_connection.session()  # pylint: disable=not-callable

    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    if not analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        click.secho("Analysis %r is of type %r." % (
            str(analysis.name), str(analysis.analysis_type)), fg='red', err=True)
        context.abort()

    ads = serialize.AnalysisDeserializer(database_connection._original_connection, analysis_id=analysis.id)

    idgps = ads.query(serialize.IdentifiedGlycopeptide).all()
    idgp_05 = 0
    idgp_01 = 0

    for idgp in idgps:
        q = idgp.q_value
        if q <= 0.05:
            idgp_05 += 1
        if q <= 0.01:
            idgp_01 += 1

    gpsms = ads.get_glycopeptide_spectrum_matches(0.05).all()
    gpsm_05 = 0
    gpsm_01 = 0

    seen_spectrum_05 = set()
    seen_spectrum_01 = set()
    for gpsm in gpsms:
        scan_id = gpsm.scan.scan_id
        q = gpsm.q_value
        if q <= 0.05:
            if scan_id not in seen_spectrum_05:
                seen_spectrum_05.add(scan_id)
                gpsm_05 += 1
        if q <= 0.01:
            if scan_id not in seen_spectrum_01:
                seen_spectrum_01.add(scan_id)
                gpsm_01 += 1

    click.echo(f"Name: {ads.analysis.name}")
    click.echo(f"Identified Glycopeptides @ 5% FDR: {idgp_05}")
    click.echo(f"Identified Glycopeptides @ 1% FDR: {idgp_01}")

    click.echo(f"Identified GPSMs @ 5% FDR: {gpsm_05}")
    click.echo(f"Identified GPSMs @ 1% FDR: {gpsm_01}")
