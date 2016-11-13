import os
import re

import click

from glycan_profiling.cli.base import cli
from glycan_profiling.cli.validators import (
    glycan_source_validators, validate_modifications,
    validate_glycan_source, validate_glycopeptide_hypothesis_name,
    validate_glycan_hypothesis_name,
    validate_reduction, validate_derivatization, validate_mzid_proteins)

from glycan_profiling.serialize import DatabaseBoundOperation, GlycanHypothesis

from glycan_profiling.database.builder.glycopeptide.naive_glycopeptide import (
    MultipleProcessFastaGlycopeptideHypothesisSerializer)

from glycan_profiling.database.builder.glycopeptide.informed_glycopeptide import (
    MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer)

from glycan_profiling.database.builder.glycan import (
    TextFileGlycanHypothesisSerializer,
    CombinatorialGlycanHypothesisSerializer)

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


@build_hypothesis.command("glycopeptide-fa")
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
@click.option("-p", "--processes", type=int, default=4, help='Number of worker processes to use')
@click.option("-s", "--glycan-source-type", default='text', type=click.Choice(
              list(glycan_source_validators.keys())),
              help="The type of glycan information source to use")
@click.option("-g", "--glycan-source", required=True,
              help="The path, identity, or other specifier for the glycan source")
def glycopeptide_fa(context, fasta_file, database_connection, enzyme, missed_cleavages, occupied_glycosites, name,
                    constant_modification, variable_modification, processes, glycan_source, glycan_source_type):
    validate_modifications(
        context, constant_modification + variable_modification)
    validate_glycan_source(context, database_connection,
                           glycan_source, glycan_source_type)

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

    builder = MultipleProcessFastaGlycopeptideHypothesisSerializer(
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


@build_hypothesis.command("glycopeptide-mzid")
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
@click.option("-p", "--processes", type=int, default=4, help='Number of worker processes to use')
@click.option("-s", "--glycan-source-type", default='text', type=click.Choice(
              list(glycan_source_validators.keys())),
              help="The type of glycan information source to use")
@click.option("-g", "--glycan-source", required=True,
              help="The path, identity, or other specifier for the glycan source")
def glycopeptide_mzid(context, mzid_file, database_connection, name, occupied_glycosites, target_protein,
                      target_protein_re, processes, glycan_source, glycan_source_type):
    proteins = validate_mzid_proteins(
        context, mzid_file, target_protein, target_protein_re)
    validate_glycan_source(context, database_connection,
                           glycan_source, glycan_source_type)

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
        n_processes=processes)
    builder.start()
    return builder.hypothesis_id


@build_hypothesis.command("glycan-text")
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


@build_hypothesis.command("glycan-combinatorial")
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


if __name__ == '__main__':
    cli.main()
