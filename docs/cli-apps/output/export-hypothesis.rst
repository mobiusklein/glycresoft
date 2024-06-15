Export Hypotheses
-----------------

:title-reference:`glycresoft` stores its search spaces
in a SQLite3 database like the one it writes hypotheses
into. This lets it search for entries quickly, but the
format is not as easy to read through as a text file, and
adds extra layers of complexity to find information requiring
complex SQL queries. It's possible to export hypotheses in
several other formats.



.. click:: glycresoft.cli.export:glycan_hypothesis
    :prog: glycresoft export glycan-hypothesis


.. click:: glycresoft.cli.export:glycopeptide_hypothesis
    :prog: glycresoft export glycopeptide-hypothesis
