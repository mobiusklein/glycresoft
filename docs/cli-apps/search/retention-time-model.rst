Retention Time Modeling for Glycopeptide Identifications
--------------------------------------------------------


Preparing Input Data
=====================

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

