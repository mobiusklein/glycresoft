Export Glycan Search Results
-----------------------------

:title-reference:`glycresoft` stores its search results
in a SQLite3 database like the one it writes hypotheses
into. This lets it search for results quickly, but the
format is not as easy to read through as a text file, and
adds extra layers of complexity to find information requiring
complex SQL queries. It's possible to export search results in
several other formats which can be read by other tools.


.. click:: glycresoft.cli.export:glycan_composition_identification
    :prog: glycresoft export glycan-identification
