Export Glycopeptide Search Results
-------------------------------------

:title-reference:`glycresoft` stores its search results
in a SQLite3 database like the one it writes hypotheses
into. This lets it search for results quickly, but the
format is not as easy to read through as a text file, and
adds extra layers of complexity to find information requiring
complex SQL queries. It's possible to export search results in
several other formats which can be read by other tools.


.. click:: glycresoft.cli.export:glycopeptide_identification
    :prog: glycresoft export glycopeptide-identification


.. click:: glycresoft.cli.export:glycopeptide_spectrum_matches
    :prog: glycresoft export glycopeptide-spectrum-matches


.. click:: glycresoft.cli.export:glycopeptide_training_mgf
    :prog: glycresoft export glycopeptide-training-mgf


.. click:: glycresoft.cli.export:export_identified_glycans_from_glycopeptides
    :prog: glycresoft export identified-glycans-from-glycopeptides


.. click:: glycresoft.cli.export:annotate_matched_spectra
    :prog: glycresoft export annotate-matched-spectra


.. click:: glycresoft.cli.export:write_spectrum_library
    :prog: glycresoft export write-csv-spectrum-library

.. .. click:: glycresoft.cli.export:glycopeptide_mzidentml
..     :prog: glycresoft export mzid


