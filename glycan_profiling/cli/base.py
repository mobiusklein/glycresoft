import multiprocessing

import click

from glycan_profiling import version


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version.version)
def cli():
    pass


class HiddenOption(click.Option):
    """
    HiddenOption -- absent from Help text.

    Supported in latest and greatest version of Click, but not old versions, so
    use generic 'cls=HiddenOption' to get the desired behavior.
    """
    def get_help_record(self, ctx):
        """
        Has "None" as its help record. All that is needed.
        """
        return


processes_option = click.option(
    "-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
    default=min(multiprocessing.cpu_count(), 4), help=('Number of worker processes to use. Defaults to 4 '
                                                       'or the number of CPUs, whichever is lower'))
