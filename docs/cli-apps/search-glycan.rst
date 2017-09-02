.. _glycan-search:

Searching a Processed Sample with a Glycan Database
===================================================

Match features from a deconvoluted LC-MS or LC-MS/MS
data file with released glycan compositions from a
glycan hypothesis (see :ref:`Combinatorial <combinatorial-glycan-hypothesis>`,
:ref:`Text <text-glycan-hypothesis>`, and :ref:`glySpace <glyspace-glycan-hypothesis>`
glycan database construction methods).

.. click:: glycan_profiling.cli.analyze:search_glycan
    :prog: glycresoft analyze search-glycan


Usage Example
-------------

.. code-block:: bash

    $ glycresoft analyze search-glycan -a Formate 1 -o agp-native-results.db\
        ../hypothesis/native-n-glycans.db path/to/sample.preprocessed.mzML 1\
        --export csv


Adducts
-------

Adducts are mass shifts that may represent alternative charge carriers
such as formate or sodium, or chemical defects such as water loss or
incomplete permethylation.

Adducts are considered combinatorially, so if you were to pass ``-a Ammonium 3``
and ``-a "C-1H-2" 1`` to indicate up to three ammonium adducts and up to one
incomplete permethylation, the program would search for

+----------------------+----------------------+----------------------+----------------------+
| 0 Ammonium, 0 C-1H-2 | 1 Ammonium, 0 C-1H-2 | 2 Ammonium, 0 C-1H-2 | 3 Ammonium, 0 C-1H-2 | 
+----------------------+----------------------+----------------------+----------------------+
| 0 Ammonium, 1 C-1H-2 | 1 Ammonium, 1 C-1H-2 | 2 Ammonium, 1 C-1H-2 | 3 Ammonium, 1 C-1H-2 | 
+----------------------+----------------------+----------------------+----------------------+

At this time, adduction models do not have any interaction with charge state.


Network Regularization
----------------------

This feature is experimental. Do not use it yet.


MS/MS Signatures
----------------

Though this tool is designed to annotate putative glycan compositions from
LC-MS, this can lead to lots of strange matches. If your data contain MS/MS
scans, passing a non-zero value to ``--require-msms-signature`` causes the
program to include only features which contain MS/MS scans which look
"glycan-like". Here "glycan-like" means containing abundant peaks which have
masses derived from mono-, di-, or tri-saccharide losses. The value of this
parameter sets the minimum ratio score.


