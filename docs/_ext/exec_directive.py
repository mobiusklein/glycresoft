# https://stackoverflow.com/questions/7250659/python-code-to-generate-part-of-sphinx-documentation-is-it-possible

import sys
from os.path import basename
import traceback

from io import StringIO

from docutils.parsers.rst import Directive
from docutils import nodes, statemachine


class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            exc = traceback.format_exc()
            return [nodes.error(
                None,
                nodes.paragraph(text="Unable to execute python code at %s:%d:" % (
                    basename(source), self.lineno)),
                nodes.paragraph(text=str(sys.exc_info()[1]))),
                nodes.paragraph(text=exc)
            ]

        finally:
            sys.stdout = oldStdout


def setup(app):
    app.add_directive('exec', ExecDirective)
