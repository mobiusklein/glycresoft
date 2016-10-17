import click
from .base import cli
from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    SampleRun, Analysis)

from glycan_profiling.database.builder.glycopeptide.proteomics import mzid_proteome


@cli.command("list", short_help='List names and ids of collections in the database')
@click.pass_context
@click.argument("database-connection", type=DatabaseBoundOperation)
def list_contents(context, database_connection):
    click.echo("Glycan Hypothesis")
    for hypothesis in database_connection.query(GlycanHypothesis):
        click.echo("\t%d\t%s\t%s" % (hypothesis.id, hypothesis.name, hypothesis.uuid))

    click.echo("\nGlycopeptide Hypothesis")
    for hypothesis in database_connection.query(GlycopeptideHypothesis):
        click.echo("\t%d\t%s\t%s\t%d" % (hypothesis.id, hypothesis.name, hypothesis.uuid,
                                         hypothesis.glycan_hypothesis.id))

    click.echo("\nSample Run")
    for run in database_connection.query(SampleRun):
        click.echo("\t%d\t%s\t%s" % (run.id, run.name, run.uuid))

    click.echo("\nAnalysis")
    for analysis in database_connection.query(Analysis):
        click.echo("\t%d\t%s\t%s" % (analysis.id, analysis.name, analysis.sample_run.name))


@cli.command('mzid-proteins', short_help='Extract proteins from mzIdentML files')
@click.argument("mzid-file")
def list_protein_names(mzid_file):
    for name in mzid_proteome.protein_names(mzid_file):
        click.echo(name)
