Retention Time Modeling for Glycopeptide Identifications
--------------------------------------------------------

GlycReSoft can use chromatographic retention time to refine glycan composition assignments
to glycopeptides using related glycoforms as points of reference.

Preparing Input Data
=====================

The model fitting process uses a simplified representation of glycopeptide chromatograms
which can be exported as a CSV file from a result database.

.. click:: glycan_profiling.cli.export:glycopeptide_chromatogram_records
    :prog: glycresoft export glycopeptide-chromatogram-records


Fitting The Model
=================

.. click:: glycan_profiling.cli.analyze:glycopeptide_retention_time_fit
    :prog: glycresoft analyze retention-time fit-glycopeptide-retention-time


Re-using A Fitted Model
=======================

.. click:: glycan_profiling.cli.analyze:glycopeptide_retention_time_predict
    :prog: glycresoft analyze retention-time evaluate-glycopeptide-retention-time

