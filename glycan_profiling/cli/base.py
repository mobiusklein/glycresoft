import multiprocessing
import functools

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

    @property
    def hidden(self):
        return True

    @hidden.setter
    def hidden(self, value):
        return

    def get_help_record(self, ctx):
        """
        Has "None" as its help record. All that is needed.
        """
        return


class DocumentableArgument(click.Argument):
    def __init__(self, *args, **kwargs):
        doc_help = kwargs.pop("doc_help", "")
        super(DocumentableArgument, self).__init__(*args, **kwargs)
        self.doc_help = doc_help


_option = click.option
_argument = click.argument


def option(*args, **kwargs):
    kwargs.setdefault("show_default", True)
    return _option(*args, **kwargs)


def argument(*args, **kwargs):
    a = _argument(*args, cls=DocumentableArgument, **kwargs)
    return a


click.option = option
click.argument = argument


processes_option = click.option(
    "-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
    default=min(multiprocessing.cpu_count(), 4), help=('Number of worker processes to use. Defaults to 4 '
                                                       'or the number of CPUs, whichever is lower'),
    # show_default=True
)
