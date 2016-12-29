import click

from glycan_profiling.cli.base import cli
from glycan_profiling.cli.validators import get_by_name_or_id

from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    Analysis, SampleRun)

from glycan_profiling.output import (
    GlycanHypothesisCSVSerializer, ImportableGlycanHypothesisCSVSerializer)


@cli.group(short_help='Write Data Collections To Text Files')
def export():
    pass


@export.command("glycan-hypothesis")
@click.argument("database-connection")
@click.argument("hypothesis-identifier")
@click.option("-o", "--output-path", type=click.Path(), default=None, help='Path to write to instead of stdout')
@click.option("-i", "--importable", is_flag=True,
              help="Make the file importable for later re-use, using the tab-separator instead of comma.")
def glycan_hypothesis(database_connection, hypothesis_identifier, output_path, importable=False):
    database_connection = DatabaseBoundOperation(database_connection)
    hypothesis = get_by_name_or_id(database_connection, GlycanHypothesis, hypothesis_identifier)
    if importable:
        task_type = ImportableGlycanHypothesisCSVSerializer
    else:
        task_type = GlycanHypothesisCSVSerializer
    output_path
    with open(output_path, 'wb') as output_stream:
        job = task_type(output_stream, hypothesis.glycans)
        job.start()
