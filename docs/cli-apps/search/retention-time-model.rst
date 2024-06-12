Retention Time Modeling for Glycopeptide Identifications
--------------------------------------------------------

.. note::

    This topic covers the stand-alone RT model described in
    Klein, J., & Zaia, J. (2020). Relative Retention Time Estimation Improves N-Glycopeptide Identifications by LC–MS/MS. Journal of Proteome Research, 19(5), 2113–2121.
    `https://doi.org/10.1021/acs.jproteome.0c00051 <https://doi.org/10.1021/acs.jproteome.0c00051>`_.

    Another retention time modeling strategy was added to the glycopeptide search tools that is more robust and has more features. This is retained
    purely for reference to that publication.


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

