import multiprocessing
import click

from glycan_profiling.cli.base import cli
from glycan_profiling.cli.validators import (
    glycan_source_validators, validate_modifications,
    validate_glycan_source, validate_glycopeptide_hypothesis_name,
    validate_glycan_hypothesis_name, get_by_name_or_id,
    validate_reduction, validate_derivatization, validate_mzid_proteins)

from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, Analysis,
    AnalysisTypeEnum)

from glycan_profiling.database.builder.glycopeptide.naive_glycopeptide import (
    MultipleProcessFastaGlycopeptideHypothesisSerializer,
    ReversingMultipleProcessFastaGlycopeptideHypothesisSerializer,
    NonSavingMultipleProcessFastaGlycopeptideHypothesisSerializer)

from glycan_profiling.database.builder.glycopeptide.informed_glycopeptide import (
    MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer)

from glycan_profiling.database.builder.glycan import (
    TextFileGlycanHypothesisSerializer,
    CombinatorialGlycanHypothesisSerializer,
    GlycanCompositionHypothesisMerger,
    NGlycanGlyspaceHypothesisSerializer,
    OGlycanGlyspaceHypothesisSerializer,
    TaxonomyFilter,
    GlycanAnalysisHypothesisSerializer,
    GlycopeptideAnalysisGlycanCompositionExtractionHypothesisSerializer)

from glycopeptidepy.utils.collectiontools import decoratordict
from glycopeptidepy.structure.modification import RestrictedModificationTable


@cli.group('build-hypothesis', short_help='Build search spaces for glycans and glycopeptides')
def build_hypothesis():
    pass


_glycan_hypothesis_builders = decoratordict()


@_glycan_hypothesis_builders('text')
def _build_glycan_composition_hypothesis_from_text(database_connection, text_file, hypothesis_name):
    if hypothesis_name is not None:
        hypothesis_name = hypothesis_name + '-Glycans'
    builder = TextFileGlycanHypothesisSerializer(
        text_file, database_connection, hypothesis_name=hypothesis_name)
    builder.start()
    click.echo(builder.summarize())
    return builder.hypothesis_id


@_glycan_hypothesis_builders("combinatorial")
def _build_glycan_composition_hypothesis_from_combinatorial(database_connection, text_file, hypothesis_name):
    if hypothesis_name is not None:
        hypothesis_name = hypothesis_name + '-Glycans'
    builder = CombinatorialGlycanHypothesisSerializer(
        text_file, database_connection, hypothesis_name=hypothesis_name)
    builder.start()
    click.echo(builder.summarize())
    return builder.hypothesis_id


@_glycan_hypothesis_builders("hypothesis")
def _get_hypothesis_id_for_glycan_composition_hypothesis(database_connection, source, hypothesis_name):
    handle = DatabaseBoundOperation(database_connection)
    try:
        hypothesis_id = int(source)
        inst = handle.query(GlycanHypothesis).get(hypothesis_id)
        if inst is not None:
            return hypothesis_id
    except TypeError:
        hypothesis_name = source
        inst = handle.query(GlycanHypothesis).filter(
            GlycanHypothesis.name == hypothesis_name).first()
        if inst is not None:
            return inst.id


@build_hypothesis.command("glycopeptide-fa",
                          short_help="Build glycopeptide search spaces with a fasta file of proteins")
@click.pass_context
@click.argument("fasta-file", type=click.Path(exists=True))
@click.argument("database-connection")
@click.option("-e", "--enzyme", default='trypsin', help='The proteolytic enzyme to use during digestion')
@click.option("-m", "--missed-cleavages", type=int, default=1,
              help="The number of missed proteolytic cleavage sites permitted")
@click.option("-u", "--occupied-glycosites", type=int, default=1,
              help=("The number of occupied glycosylation sites permitted. "
                    "Warning: Increasing this number exponentially increases the complexity of this process "
                    "as the number of glycan compositions increases."))
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-c", "--constant-modification", multiple=True,
              help='Peptide modification rule which will be applied constantly')
@click.option("-v", "--variable-modification", multiple=True,
              help='Peptide modification rule which will be applied variablely')
@click.option("-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
              default=min(multiprocessing.cpu_count(), 4), help=('Number of worker processes to use. Defaults to 4 '
                                                                 'or the number of CPUs, whichever is lower'))
@click.option("-s", "--glycan-source-type", default='text', type=click.Choice(
              list(glycan_source_validators.keys())),
              help="The type of glycan information source to use")
@click.option("-g", "--glycan-source", required=True,
              help="The path, identity, or other specifier for the glycan source")
@click.option("--reverse", default=False, is_flag=True, help='Reverse protein sequences')
@click.option("--dry-run", default=False, is_flag=True, help="Do not save glycopeptides")
def glycopeptide_fa(context, fasta_file, database_connection, enzyme, missed_cleavages, occupied_glycosites, name,
                    constant_modification, variable_modification, processes, glycan_source, glycan_source_type,
                    reverse=False, dry_run=False):
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
                           glycan_source, glycan_source_type)

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
        glycan_source_type](database_connection, glycan_source, name)

    builder = task_type(
        fasta_file, database_connection,
        glycan_hypothesis_id=glycan_hypothesis_id,
        protease=enzyme,
        constant_modifications=constant_modification,
        variable_modifications=variable_modification,
        max_missed_cleavages=missed_cleavages,
        max_glycosylation_events=occupied_glycosites,
        hypothesis_name=name,
        n_processes=processes)

    builder.start()
    return builder.hypothesis_id


@build_hypothesis.command("glycopeptide-mzid",
                          short_help="Build a glycopeptide search space with an mzIdentML file")
@click.pass_context
@click.argument("mzid-file", type=click.Path(exists=True))
@click.argument("database-connection")
@click.option("-u", "--occupied-glycosites", type=int, default=1,
              help=("The number of occupied glycosylation sites permitted. "
                    "Warning: Increasing this number exponentially increases the complexity of this process "
                    "as the number of glycan compositions increases."))
@click.option("-t", "--target-protein", multiple=True,
              help='Specifies the name of a protein to include in the hypothesis. May be used many times.')
@click.option('-r', '--target-protein-re', multiple=True,
              help=('Specifies a regular expression to select proteins'
                    ' to be included by name. May be used many times.'))
@click.option("-n", "--name", default=None, help="Name for the hypothesis to be created")
@click.option("-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
              default=min(multiprocessing.cpu_count(), 4), help=('Number of worker processes to use. Defaults to 4 '
                                                                 'or the number of CPUs, whichever is lower'))
@click.option("-s", "--glycan-source-type", default='text', type=click.Choice(
              list(glycan_source_validators.keys())),
              help="The type of glycan information source to use")
@click.option("-g", "--glycan-source", required=True,
              help="The path, identity, or other specifier for the glycan source")
@click.option("-r", "--reference-fasta", default=None, required=False,
              help=("When the full sequence for each protein is not embedded in the mzIdentML file and "
                    "the FASTA file used by the search engine that created the mzIdentML file is not "
                    "at the path specified in the file, you must provide a FASTA file to retrieve "
                    "protein sequences from."))
def glycopeptide_mzid(context, mzid_file, database_connection, name, occupied_glycosites, target_protein,
                      target_protein_re, processes, glycan_source, glycan_source_type, reference_fasta):
    proteins = validate_mzid_proteins(
        context, mzid_file, target_protein, target_protein_re)
    validate_glycan_source(context, database_connection,
                           glycan_source, glycan_source_type)

    processes = min(multiprocessing.cpu_count(), processes)

    if name is not None:
        name = validate_glycopeptide_hypothesis_name(
            context, database_connection, name)
        click.secho("Building Glycopeptide Hypothesis %s" % name, fg='cyan')

    glycan_hypothesis_id = _glycan_hypothesis_builders[
        glycan_source_type](database_connection, glycan_source, name)

    builder = MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer(
        mzid_file, database_connection,
        glycan_hypothesis_id=glycan_hypothesis_id,
        hypothesis_name=name,
        target_proteins=proteins,
        max_glycosylation_events=occupied_glycosites,
        reference_fasta=reference_fasta,
        n_processes=processes)
    builder.start()
    return builder.hypothesis_id


@build_hypothesis.command("glycan-text",
                          short_help='Build a glycan search space with a text file of glycan compositions')
@click.pass_context
@click.argument("text-file", type=click.Path(exists=True))
@click.argument("database-connection")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
def glycan_text(context, text_file, database_connection, reduction, derivatization, name):
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    validate_reduction(context, reduction)
    validate_derivatization(context, derivatization)
    builder = TextFileGlycanHypothesisSerializer(
        text_file, database_connection, reduction=reduction, derivatization=derivatization,
        hypothesis_name=name)
    builder.start()


@build_hypothesis.command("glycan-combinatorial", short_help=('Build a glycan search space with a text file'
                                                              ' containing algebraic combination rules'))
@click.pass_context
@click.argument("rule-file", type=click.Path(exists=True))
@click.argument("database-connection")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
def glycan_combinatorial(context, rule_file, database_connection, reduction, derivatization, name):
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    validate_reduction(context, reduction)
    validate_derivatization(context, derivatization)
    builder = CombinatorialGlycanHypothesisSerializer(
        rule_file, database_connection, reduction=reduction, derivatization=derivatization,
        hypothesis_name=name)
    builder.start()


@build_hypothesis.command("merge-glycan", short_help=("Combine two or more glycan search spaces to create a "
                                                      "new one containing unique entries from all constituents"))
@click.pass_context
@click.argument("database-connection")
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-i", "--hypothesis-identifier", multiple=True, help="A hypothesis to include. May be used many times")
def merge_glycan_hypotheses(context, database_connection, hypothesis_identifier, name):
    database_connection = DatabaseBoundOperation(database_connection)
    hypothesis_ids = []
    for ident in hypothesis_identifier:
        hypothesis = get_by_name_or_id(database_connection, GlycanHypothesis, ident)
        hypothesis_ids.append(hypothesis.id)

    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection._original_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')

    task = GlycanCompositionHypothesisMerger(
        database_connection._original_connection, hypothesis_ids, name)
    task.start()


@build_hypothesis.command("glyspace-glycan", short_help=("Construct a glycan hypothesis from GlySpace"))
@click.pass_context
@click.argument("database-connection")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
@click.option("-m", "--motif-class", type=click.Choice(["n-linked", "o-linked"]), default=None,
              help="Specify a glycan structure family to search for")
@click.option("-t", "--target-taxon", default=None, help="Only select structures annotated with this taxonomy")
@click.option("-i", "--include-children", default=False, is_flag=True,
              help="Include child taxa of --target-taxon. No effect otherwise.")
def glyspace_glycan_hypothesis(context, database_connection, motif_class, reduction, derivatization, name,
                               target_taxon=None, include_children=False):
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
    job = serializer_type(
        database_connection._original_connection, name, reduction, derivatization, filter_funcs,
        simplify=True)
    job.start()


@build_hypothesis.command("glycan-from-analysis", short_help=("Construct a glycan hypothesis from a matched analysis"))
@click.pass_context
@click.argument("database-connection")
@click.argument("analysis-identifier")
@click.option("-r", "--reduction", default=None, help='Reducing end modification')
@click.option("-d", "--derivatization", default=None, help='Chemical derivatization to apply')
@click.option("-n", "--name", default=None, help="The name for the hypothesis to be created")
def from_analysis(context, database_connection, analysis_identifier, reduction, derivatization, name):
    database_connection = DatabaseBoundOperation(database_connection)
    if name is not None:
        name = validate_glycan_hypothesis_name(context, database_connection._original_connection, name)
        click.secho("Building Glycan Hypothesis %s" % name, fg='cyan')
    validate_reduction(context, reduction)
    validate_derivatization(context, derivatization)
    analysis = get_by_name_or_id(database_connection.session, Analysis, analysis_identifier)
    if analysis.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
        job = GlycanAnalysisHypothesisSerializer(database_connection._original_connection, analysis.id, name)
        job.start()
    elif analysis.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
        job = GlycopeptideAnalysisGlycanCompositionExtractionHypothesisSerializer(
            database_connection._original_connection, analysis.id, name)
        job.start()
    else:
        click.secho("Analysis Type %r could not be converted" % (
            analysis.analysis_type.name,), fg='red')


if __name__ == '__main__':
    cli.main()
