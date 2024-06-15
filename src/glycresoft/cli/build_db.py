import sys
import multiprocessing
import click
import textwrap

from glycresoft.cli.base import cli, HiddenOption

from glycresoft.cli.validators import (
    glycan_source_validators,
    validate_modifications,
    validate_glycan_source,
    validate_glycopeptide_hypothesis_name,
    validate_glycan_hypothesis_name,
    get_by_name_or_id,
    validate_reduction,
    validate_derivatization,
    validate_mzid_proteins,
    GlycanHypothesisCopier,
    DatabaseConnectionParam,
    SubstituentParamType)

from glycresoft.cli.utils import ctxstream

from glycresoft.serialize import (
    DatabaseBoundOperation,
    GlycanHypothesis,
    Analysis,
    AnalysisTypeEnum)

from glycresoft.database.builder.glycopeptide.naive_glycopeptide import (
    MultipleProcessFastaGlycopeptideHypothesisSerializer,
    ReversingMultipleProcessFastaGlycopeptideHypothesisSerializer,
    NonSavingMultipleProcessFastaGlycopeptideHypothesisSerializer)

from glycresoft.database.builder.glycopeptide.informed_glycopeptide import (
    MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer,
    ReversingMultipleProcessMzIdentMLGlycopeptideHypothesisSerializer)

from glycresoft.database.builder.glycan import (
    TextFileGlycanHypothesisSerializer,
    CombinatorialGlycanHypothesisSerializer,
    GlycanCompositionHypothesisMerger,
    NGlycanGlyspaceHypothesisSerializer,
    OGlycanGlyspaceHypothesisSerializer,
    TaxonomyFilter,
    GlycanAnalysisHypothesisSerializer,
    GlycopeptideAnalysisGlycanCompositionExtractionHypothesisSerializer,
    SynthesisGlycanHypothesisSerializer)

from glycresoft.database.prebuilt import hypothesis_register as prebuilt_hypothesis_register

from glycresoft.database.disk_backed_database import GlycanCompositionDiskBackedStructureDatabase
from glycresoft.database.composition_network import (
    CompositionGraph, GraphWriter, GraphReader,
    CompositionRangeRule,
    CompositionRatioRule,
    CompositionRuleClassifier,
    CompositionExpressionRule,
    make_n_glycan_neighborhoods,
    make_adjacency_neighborhoods,
    make_mammalian_n_glycan_neighborhoods)

from glycresoft.database.builder.glycan.synthesis import synthesis_register

from glycopeptidepy.enzyme import enzyme_rules
from glycopeptidepy.utils.collectiontools import decoratordict
from glycopeptidepy.structure.modification import RestrictedModificationTable


@cli.group('build-hypothesis', short_help='Build search spaces for glycans and glycopeptides')
def build_hypothesis():
    pass


command_group = build_hypothesis


def database_connection(fn):
    arg = click.argument("database-connection", doc_help=(
        "A connection URI for a database, or a path on the file system"))
    return arg(fn)


_glycan_hypothesis_builders = decoratordict()


@_glycan_hypothesis_builders('text')
def _build_glycan_composition_hypothesis_from_text(database_connection, text_file, hypothesis_name,
                                                   identifier=None):
    if hypothesis_name is not None:
        hypothesis_name = hypothesis_name + '-Glycans'
    builder = TextFileGlycanHypothesisSerializer(
        text_file, database_connection, hypothesis_name=hypothesis_name)
    builder.run()
    return builder.hypothesis_id


@_glycan_hypothesis_builders("combinatorial")
def _build_glycan_composition_hypothesis_from_combinatorial(database_connection, text_file, hypothesis_name,
                                                            identifier=None):
    if hypothesis_name is not None:
        hypothesis_name = hypothesis_name + '-Glycans'
    builder = CombinatorialGlycanHypothesisSerializer(
        text_file, database_connection, hypothesis_name=hypothesis_name)
    builder.run()
    return builder.hypothesis_id


@_glycan_hypothesis_builders("hypothesis")
def _copy_hypothesis_across_file_boundaries(database_connection, source, hypothesis_name,
                                            identifier=None):
    source_handle = DatabaseBoundOperation(source)
    source_hypothesis_id = None
    source_hypothesis_name = None

    try:
        hypothesis_id = int(identifier)
        inst = source_handle.query(GlycanHypothesis).get(hypothesis_id)
        if inst is not None:
            source_hypothesis_id = hypothesis_id
            source_hypothesis_name = inst.name

    except TypeError:
        hypothesis_name = identifier
        inst = source_handle.query(GlycanHypothesis).filter(
            GlycanHypothesis.name == hypothesis_name).first()
        if inst is not None:
            source_hypothesis_id = inst.id
            source_hypothesis_name = inst.name

    if source == database_connection:
        return source_hypothesis_id

    mover = GlycanHypothesisCopier(
        database_connection, [(source, source_hypothesis_id)],
        hypothesis_name=source_hypothesis_name)
    mover.run()
    return mover.hypothesis_id


@_glycan_hypothesis_builders("analysis")
def _copy_analysis_across_file_boundaries(database_connection, source, hypothesis_name,
                                          identifier=None):
    source_handle = DatabaseBoundOperation(source)
    source_analysis_id = None
    source_analysis_name = None
    try:
        hypothesis_id = int(identifier)
        inst = source_handle.query(Analysis).get(hypothesis_id)
        if inst is not None:
            source_analysis_id = hypothesis_id
            source_analysis_name = inst.name

    except TypeError:
        hypothesis_name = identifier
        inst = source_handle.query(Analysis).filter(
            Analysis.name == hypothesis_name).first()
        if inst is not None:
            source_analysis_id = inst.id
            source_analysis_name = inst.name
    if hypothesis_name is None:
        hypothesis_name = source_analysis_name
    mover = GlycanAnalysisHypothesisSerializer(
        source, source_analysis_id, hypothesis_name,
        database_connection)
    mover.run()
    return mover.hypothesis_id


def _validate_glycan_source_identifier(ctx, param, value):
    try:
        if ctx.params['glycan_source_type'] not in ("hypothesis", "analysis") and value:
            width = click.get_terminal_size()[0]
            click.secho('\n'.join(
                textwrap.wrap(
                    "Warning: --glycan-source-identifier specified when "
                    "--glycan-source is neither \"hypothesis\" nor \"analysis\"."
                    " Its value will be ignored.", width=int(width * 0.6))), fg='yellow')
            return None
        else:
            return value
    except KeyError:
        click.secho("Specify --glycan-source before --glycan-source-identifier.", fg='yellow')


def glycopeptide_hypothesis_common_options(cmd):
    options = [
        click.option("-u", "--occupied-glycosites", type=int, default=1,
                     help=("The number of occupied glycosylation sites permitted.")),
        click.option("-n", "--name", default=None, help="The name for the hypothesis to be created"),
        click.option("-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
                     default=min(multiprocessing.cpu_count(), 4),
                     help=('Number of worker processes to use. Defaults to 4 '
                           'or the number of CPUs, whichever is lower')),
        click.option("-G", "--glycan-source-identifier", required=False, default=None,
                     help=("The name or id number of the hypothesis or analysis to"
                           " be used when using those glycan source types."),
                     callback=_validate_glycan_source_identifier),
        click.option("-s", "--glycan-source-type", default='text', type=click.Choice(
                     list(glycan_source_validators.keys())),
                     help="The type of glycan information source to use"),
        click.option("-g", "--glycan-source", required=True,
                     help="The path, identity, or other specifier for the glycan source"),
        click.option("-z", "--peptide-length-range", type=(int, int), default=(5, 60),
                     help="The minimum and maximum peptide length to consider")
    ]
    for opt in options:
        cmd = opt(cmd)
    return cmd


@build_hypothesis.command("glycopeptide-fa",
                          short_help="Build glycopeptide search spaces with a FASTA file of proteins")
@click.pass_context
@glycopeptide_hypothesis_common_options
@click.argument("fasta-file", type=click.Path(exists=True), doc_help=(
    "A file containing protein sequences in FASTA format"))
@database_connection
@click.option("-e", "--enzyme", default=['trypsin'], multiple=True,
              help=(
                  'The proteolytic enzyme to use during digestion. May be specified multiple'
                  ' times, generating a co-digestion. May specify an enzyme name or a regular'
                  ' expression describing the cleavage pattern. Recognized enzyme names are: ' +
                  ', '.join(sorted(enzyme_rules))))
@click.option("-m", "--missed-cleavages", type=int, default=1,
              help="The number of missed proteolytic cleavage sites permitted")
@click.option("-c", "--constant-modification", multiple=True,
              help='Peptide modification rule which will be applied constantly')
@click.option("-v", "--variable-modification", multiple=True,
              help='Peptide modification rule which will be applied variablely')
@click.option("-V", "--max-variable-modifications", type=int, default=4, required=False,
              help=("The maximum number of variable modifications that can be applied to a single peptide"))
@click.option("-y", "--semispecific-digest", is_flag=True, help=(
    "Apply a semispecific enzyme digest permitting one peptide terminal to be non-specific"))
@click.option("-R", "--reverse", default=False, is_flag=True, help='Reverse protein sequences')
@click.option("--dry-run", default=False, is_flag=True, help="Do not save glycopeptides", cls=HiddenOption)
@click.option("-F", "--not-full-crossproduct", is_flag=True, help=(
    "Do not produce full crossproduct. For when the search space is too large to enumerate, store, and load."))
@click.option("--retain-all-peptides", is_flag=True, default=False,
              help=("Do not require a glycosylation site when saving base peptides"))
@click.option("-C", "--include-n-x-c-glycosylation", "include_cysteine_n_glycosylation", default=False, is_flag=True,
              help="Whether to support N-x-C in the N-glycosylation sequon")
@click.option("-a", "--annotation-path", type=click.Path(exists=True, allow_dash=True), required=False,
              default=None, help="The path to an annotation XML file from UniProt, or '-' to suppress annotations.")
def glycopeptide_fa(context, fasta_file, database_connection, enzyme, missed_cleavages, occupied_glycosites, name,
                    constant_modification, variable_modification, processes, glycan_source, glycan_source_type,
                    glycan_source_identifier=None, semispecific_digest=False, reverse=False, dry_run=False,
                    peptide_length_range=(5, 60), not_full_crossproduct=False, max_variable_modifications=4,
                    retain_all_peptides=False, include_cysteine_n_glycosylation: bool=False, annotation_path=None):
    '''Constructs a glycopeptide hypothesis from a FASTA file of proteins and a
    collection of glycans.
    '''
    if reverse:
        task_type = ReversingMultipleProcessFastaGlycopeptideHypothesisSerializer
        click.secho("Using ReversingMultipleProcessFastaGlycopeptideHypothesisSerializer", fg='yellow')
    elif dry_run:
        task_type = NonSavingMultipleProcessFastaGlycopeptideHypothesisSerializer
        click.secho("Using NonSavingMultipleProcessFastaGlycopeptideHypothesisSerializer", fg='yellow')
    else:
        task_type = MultipleProcessFastaGlycopeptideHypothesisSerializer

    validate_modifications(
        context, constant_modification + variable_modification)
    validate_glycan_source(context, database_connection,
                           glycan_source, glycan_source_type,
                           glycan_source_identifier)

    processes = min(multiprocessing.cpu_count(), processes)

    if name is not None:
        name = validate_glycopeptide_hypothesis_name(
            context, database_connection, name)
        click.secho("Building Glycopeptide Hypothesis %s" % name, fg='cyan')
    mt = RestrictedModificationTable(
        None, constant_modification, variable_modification)
    constant_modification = [mt[c] for c in constant_modification]
    variable_modification = [mt[c] for c in variable_modification]

    glycan_hypothesis_id = _glycan_hypothesis_builders[
        glycan_source_type](database_connection, glycan_source, name, glycan_source_identifier)

    builder = task_type(
        fasta_file, database_connection,
        glycan_hypothesis_id=glycan_hypothesis_id,
        protease=enzyme,
        constant_modifications=constant_modification,
        variable_modifications=variable_modification,
        max_missed_cleavages=missed_cleavages,
        max_glycosylation_events=occupied_glycosites,
        hypothesis_name=name,
        semispecific=semispecific_digest,
        n_processes=processes,
        full_cross_product=not not_full_crossproduct,
        max_variable_modifications=max_variable_modifications,
        peptide_length_range=peptide_length_range,
        require_glycosylation_sites=not retain_all_peptides,
        include_cysteine_n_glycosylation=include_cysteine_n_glycosylation,
        uniprot_source_file=annotation_path)
    builder.display_header()
    builder.start()
    return builder.hypothesis_id


@build_hypothesis.command("glycopeptide-mzid",
                          short_help="Build a glycopeptide search space with an mzIdentML file")
@click.pass_context
@click.argument("mzid-file", type=click.Path(exists=True))
@database_connection
@glycopeptide_hypothesis_common_options
@click.option("-t", "--target-protein", multiple=True,
              help='Specifies the name of a protein to include in the hypothesis, or a file containing such'
                   ', one per line. May be used many times.')
@click.option('-r', '--target-protein-re', multiple=True,
              help=('Specifies a regular expression to select proteins'
                    ' to be included by name. May be used many times.'))
@click.option("--reference-fasta", default=None, required=False,
              help=("When the full sequence for each protein is not embedded in the mzIdentML file and "
                    "the FASTA file used is not local."))
@click.option("-a", "--annotation-path", type=click.Path(exists=True, allow_dash=True), required=False,
              default=None, help="The path to an annotation XML file from UniProt or '-' to suppress loading annotations")
@click.option("-R", "--reverse", default=False, is_flag=True, help='Reverse protein sequences')
@click.option("-F", "--not-full-crossproduct", is_flag=True, help=(
    "Do not produce full crossproduct. For when the search space is too large to enumerate, store, and load."))
def glycopeptide_mzid(context, mzid_file, database_connection, name, occupied_glycosites, target_protein,
                      target_protein_re, processes, glycan_source, glycan_source_type, glycan_source_identifier,
                      reference_fasta, peptide_length_range=(5, 60), annotation_path=None, reverse=False,
                      not_full_crossproduct=False):
    """
    Constructs a glycopeptide hypothesis from a MzIdentML file of proteins and a
    collection of glycans.
    """
    proteins = validate_mzid_proteins(
        context, mzid_file, target_protein, target_protein_re)
    validate_glycan_source(context, database_connection,
                           glycan_source, glycan_source_type,
                           glycan_source_identifier)

    processes = min(multiprocessing.cpu_count(), processes)

    if name is not None:
        name = validate_glycopeptide_hypothesis_name(
            context, database_connection, name)
        click.secho("Building Glycopeptide Hypothesis %s" % name, fg='cyan')

    glycan_hypothesis_id = _glycan_hypothesis_builders[
        glycan_source_type](database_connection, glycan_source, name, glycan_source_identifier)

    job_cls = MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer
    if reverse:
        job_cls = ReversingMultipleProcessMzIdentMLGlycopeptideHypothesisSerializer

    builder = job_cls(
        mzid_file, database_connection,
        glycan_hypothesis_id=glycan_hypothesis_id,
        hypothesis_name=name,
        target_proteins=proteins,
        max_glycosylation_events=occupied_glycosites,
        reference_fasta=reference_fasta,
        n_processes=processes,
        peptide_length_range=peptide_length_range,
        uniprot_source_file=annotation_path,
        full_cross_product=not not_full_crossproduct
    )
    builder.display_header()
    builder.start()
    return builder.hypothesis_id


@build_hypothesis.command("glycan-text",
                          short_help='Build a glycan search space with a text file of glycan compositions')
@click.pass_context
@click.argument("text-file", type=click.Path(exists=True))
@database_connection
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
def glycan_text(context, text_file, database_connection, reduction, derivatization, name):
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    reduction = validate_reduction(context, reduction)
    derivatization = validate_derivatization(context, derivatization)
    builder = TextFileGlycanHypothesisSerializer(
        text_file, database_connection, reduction=reduction, derivatization=derivatization,
        hypothesis_name=name)
    builder.display_header()
    builder.start()


@build_hypothesis.command("glycan-combinatorial", short_help=('Build a glycan search space with a text file'
                                                              ' containing algebraic combination rules'))
@click.pass_context
@click.argument("rule-file", type=click.Path(exists=True))
@database_connection
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-c", "--glycan-type", default=None, required=False,
              type=click.Choice(['n-glycan', 'o-glycan', 'gag-linker'], case_sensitive=False),
              help="The glycan classification to apply to all generated glycan compositions."
                   " If not set, a heuristic will be used to guess if a composition in an N-glycan."
                   " This is only relevant when building a glycopeptide hypothesis.")
def glycan_combinatorial(context, rule_file, database_connection, reduction, derivatization, name,
                         glycan_type=None):
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    reduction = validate_reduction(context, reduction)
    derivatization = validate_derivatization(context, derivatization)
    builder = CombinatorialGlycanHypothesisSerializer(
        rule_file, database_connection, reduction=reduction, derivatization=derivatization,
        hypothesis_name=name, glycan_type=glycan_type)
    builder.display_header()
    builder.start()


@build_hypothesis.command("merge-glycan", short_help=("Combine two or more glycan search spaces to create a "
                                                      "new one containing unique entries from all constituents"))
@click.pass_context
@database_connection
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option(
    "-i", "--hypothesis-specification", multiple=True,
    nargs=2, help=("The location and identity for the hypothesis to"
                   " include. May be specified many times"))
def merge_glycan_hypotheses(context, database_connection, hypothesis_specification, name):
    database_connection = DatabaseBoundOperation(database_connection)
    hypothesis_ids = []
    for connection, ident in hypothesis_specification:
        hypothesis = get_by_name_or_id(DatabaseBoundOperation(connection), GlycanHypothesis, ident)
        hypothesis_ids.append((connection, hypothesis.id))

    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection._original_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')

    task = GlycanCompositionHypothesisMerger(
        database_connection._original_connection, hypothesis_ids, name)
    task.display_header()
    task.start()


@build_hypothesis.command("glycan-glyspace", short_help=("Construct a glycan hypothesis from GlySpace"))
@click.pass_context
@database_connection
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-m", "--motif-class", type=click.Choice(["n-linked", "o-linked"]), default=None,
              help="Specify a glycan structure family to search for")
@click.option("-t", "--target-taxon", default=None, help="Only select structures annotated with this taxonomy")
@click.option("-i", "--include-children", default=False, is_flag=True,
              help="Include child taxa of --target-taxon. No effect otherwise.")
@click.option("-s", "--detatch-substituent", multiple=True, type=SubstituentParamType(),
              help='Substituent type to detatch from all monosaccharides')
def glyspace_glycan_hypothesis(context, database_connection, motif_class, reduction, derivatization, name,
                               target_taxon=None, include_children=False, detatch_substituent=None):
    database_connection = DatabaseBoundOperation(database_connection)
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection._original_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    filter_funcs = []

    if target_taxon is not None:
        filter_funcs.append(TaxonomyFilter(target_taxon, include_children))

    serializer_type = None
    if motif_class == "n-linked":
        serializer_type = NGlycanGlyspaceHypothesisSerializer
    elif motif_class == "o-linked":
        serializer_type = OGlycanGlyspaceHypothesisSerializer

    reduction = validate_reduction(context, reduction)
    derivatization = validate_derivatization(context, derivatization)
    job = serializer_type(
        database_connection._original_connection, name, reduction, derivatization, filter_funcs,
        simplify=True, substituents_to_detatch=detatch_substituent)
    job.display_header()
    job.start()


@build_hypothesis.command("glycan-from-analysis", short_help=("Construct a glycan hypothesis from a matched analysis"))
@click.pass_context
@database_connection
@click.argument("analysis-connection", type=DatabaseConnectionParam(exists=True))
@click.argument("analysis-identifier")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
def from_analysis(context, database_connection, analysis_connection, analysis_identifier,
                  reduction, derivatization, name):
    database_connection = DatabaseBoundOperation(database_connection)
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection._original_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    reduction = validate_reduction(context, reduction)
    derivatization = validate_derivatization(context, derivatization)

    analysis_connection = DatabaseBoundOperation(analysis_connection)
    analysis = get_by_name_or_id(analysis_connection.session, Analysis, analysis_identifier)
    if analysis.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
        job = GlycanAnalysisHypothesisSerializer(
            analysis_connection._original_connection, analysis.id, name,
            output_connection=database_connection._original_connection)
        job.display_header()
        job.start()
    elif analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        job = GlycopeptideAnalysisGlycanCompositionExtractionHypothesisSerializer(
            analysis_connection._original_connection, analysis.id, name,
            output_connection=database_connection._original_connection)
        job.display_header()
        job.start()
    else:
        click.secho("Analysis Type %r could not be converted" % (
            analysis.analysis_type.name,), fg='red')


@build_hypothesis.command("prebuilt-glycan", short_help=(
    'Construct a glycan hypothesis from a list of pre-made recipes'))
@click.pass_context
@database_connection
@click.option("-p", "--recipe-name", type=click.Choice(prebuilt_hypothesis_register.keys()), required=True)
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
def prebuilt_glycan(context, database_connection, recipe_name, name, reduction, derivatization):
    database_connection = DatabaseBoundOperation(database_connection)
    reduction = validate_reduction(context, reduction)
    derivatization = validate_derivatization(context, derivatization)
    if name is not None:
        name = validate_glycan_hypothesis_name(
            context, database_connection._original_connection, name)
    recipe = prebuilt_hypothesis_register[recipe_name]()
    recipe(database_connection._original_connection,
           hypothesis_name=name, reduction=reduction,
           derivatization=derivatization)


@build_hypothesis.command(
    "glycan-synthesis",
    short_help='Construct a glycan hypothesis from a set of biosynthetic pathways')
@click.pass_context
@database_connection
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-w", "--pathway-type", default='human-n-glycan', type=click.Choice([
    'mammalian-n-glycan', 'human-n-glycan',
    'human-o-glycan', 'mammalian-o-glycan']), help=(
    'The default pathway template to use, and the glycan class annotation type.'))
@click.option('-m', '--maximum-size', default=26, type=int,
              help='The maximum number of residues to allow in a single glycan')
@click.option('-e', "--remove-enzyme", "removed_enzymes", multiple=True)
@click.option('-a', "--add-enzyme", "added_enzymes", multiple=True)
@click.option("-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
              default=min(multiprocessing.cpu_count(), 4),
              help=('Number of worker processes to use. Defaults to 4 '
                    'or the number of CPUs, whichever is lower'))
def glycan_synthesis(context, database_connection, name, pathway_type, removed_enzymes, added_enzymes,
                     maximum_size, reduction, derivatization, processes):
    pathway = synthesis_register[pathway_type](n_processes=processes)
    pathway.add_limit(pathway.size_limiter_type(maximum_size))
    for enzyme_to_remove in removed_enzymes:
        pathway.remove_enzyme(enzyme_to_remove)
    for enzyme_to_add in added_enzymes:
        click.echo("Adding enzymes is unsupported at this time")
    task = SynthesisGlycanHypothesisSerializer(
        pathway, database_connection,
        hypothesis_name=name, reduction=reduction, derivatization=derivatization)
    task.start()


@build_hypothesis.group("glycan-network", short_help='Build and manipulate glycan network definitions')
def glycan_network_tools():
    pass


@glycan_network_tools.command("build-network", short_help=(
    "Build a glycan network for an existing glycan hypothesis"))
@click.pass_context
@database_connection
@click.argument("hypothesis-identifier", doc_help=(
                "The ID number or name of the glycan hypothesis to use"))
@click.option("-o", "--output-path", type=click.Path(
    file_okay=True, dir_okay=False, writable=True), default=None,
    help='Path to write to instead of stdout')
@click.option("-e", "--edge-strategy", type=click.Choice(["manhattan", ]), default='manhattan',
              help="Strategy to use to decide when two nodes are connected by an edge")
def glycan_network(context, database_connection, hypothesis_identifier, edge_strategy, output_path):
    conn = DatabaseBoundOperation(database_connection)
    hypothesis = get_by_name_or_id(conn, GlycanHypothesis, hypothesis_identifier)
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'w')
    with output_stream:
        db = GlycanCompositionDiskBackedStructureDatabase(
            database_connection, hypothesis.id)
        glycans = list(db)
        graph = CompositionGraph(glycans)
        if edge_strategy == 'manhattan':
            graph.create_edges(1)
        else:
            raise click.ClickException(
                "Could not find edge strategy %r" % (edge_strategy,))
        GraphWriter(graph, output_stream)


@glycan_network_tools.command("add-prebuilt-neighborhoods", short_help=(
    "Add a set of neighborhoods with built-in rules"))
@click.pass_context
@click.option("-i", "--input-path", type=click.Path(dir_okay=False), default=None)
@click.option("-o", "--output-path", type=click.Path(dir_okay=False, writable=True),
              default=None)
@click.option("-n", "--name", help='Set the neighborhood name', type=click.Choice(
    ["n-glycan", "mammalian-n-glycan", ]), required=True)
def add_prebuild_neighborhoods_to_network(context, input_path, output_path, name):
    if input_path is None:
        input_stream = ctxstream(sys.stdin)
    else:
        input_stream = open(input_path, 'r')
    with input_stream:
        graph = GraphReader(input_stream).network
    if name == 'n-glycan':
        neighborhoods = make_n_glycan_neighborhoods()
    elif name == 'mammalian-n-glycan':
        neighborhoods = make_mammalian_n_glycan_neighborhoods()
    else:
        raise LookupError(name)
    for n in neighborhoods:
        graph.neighborhoods.add(n)
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'w')
    with output_stream:
        GraphWriter(graph, output_stream)


@glycan_network_tools.command("add-neighborhood", short_help=(
    'Add a neighborhood definition to a glycan network'))
@click.pass_context
@click.option("-i", "--input-path", type=click.Path(dir_okay=False), default=None)
@click.option("-o", "--output-path", type=click.Path(dir_okay=False, writable=True),
              default=None)
@click.option("-n", "--name", help='Set the neighborhood name', required=True)
@click.option("-r", "--range-rule", nargs=4, multiple=True, help=(
    "Format: <expression:str> <low:int> <high:int> <required:bool>"
))
@click.option("-e", "--expression-rule", nargs=2, multiple=True, help=(
    "Format: <bool-expression:str> <required:bool>"
))
@click.option("-a", "--ratio-rule", nargs=4, multiple=True, help=(
    "Format: <numerator-expr:str> <denominator-expr:str> <threshold:float> <required:bool>"
))
def add_neighborhood_to_network(context, input_path, output_path, name, range_rule, expression_rule, ratio_rule):
    if input_path is None:
        input_stream = ctxstream(sys.stdin)
    else:
        input_stream = open(input_path, 'r')
    with input_stream:
        graph = GraphReader(input_stream).network
    if name in graph.neighborhoods:
        click.secho(
            "This network already has a neighborhood called %s, overwriting" % name,
            fg='yellow', err=True)
    rules = []
    for rule in range_rule:
        expr, lo, hi, req = rule
        lo = int(lo)
        hi = int(hi)
        req = req.lower().strip() in ('true', 'yes', '1')
        rules.append(CompositionRangeRule(expr, lo, hi, req))
    for rule in expression_rule:
        expr, req = rule
        req = req.lower().strip() in ('true', 'yes', '1')
        rules.append(CompositionExpressionRule(expr, req))
    for rule in ratio_rule:
        numer, denom, threshold, req = rule
        threshold = float(threshold)
        req = req.lower().strip() in ('true', 'yes', '1')
        rules.append(CompositionRatioRule(numer, denom, threshold, req))
    graph.neighborhoods.add(CompositionRuleClassifier(name, rules))
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'w')
    with output_stream:
        GraphWriter(graph, output_stream)


@glycan_network_tools.command("remove-neighborhood", short_help=(
    'Remove a neighborhood definition to a glycan network'))
@click.pass_context
@click.option("-i", "--input-path", type=click.Path(dir_okay=False), default=None)
@click.option("-o", "--output-path", type=click.Path(dir_okay=False, writable=True),
              default=None)
@click.option("-n", "--name", help='Set the neighborhood name', required=True)
def remove_neighborhood(context, input_path, output_path, name):
    if input_path is None:
        input_stream = ctxstream(sys.stdin)
    else:
        input_stream = open(input_path, 'r')
    with input_stream:
        graph = GraphReader(input_stream).network
    try:
        graph.neighborhoods.remove(name)
    except KeyError:
        click.secho(
            "No neighborhood with name %r was found" % name, err=True, fg='yellow')
    if output_path is None:
        output_stream = ctxstream(sys.stdout)
    else:
        output_stream = open(output_path, 'w')
    with output_stream:
        GraphWriter(graph, output_stream)


if __name__ == '__main__':
    cli.main()
