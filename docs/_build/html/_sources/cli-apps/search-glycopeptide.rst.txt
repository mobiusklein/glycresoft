Searching a Processed Sample with a Glycopeptide Database
============================================================

The end-goal of all of these tools is to be able to identify glycopeptides
from experimental data. After you've constructed a glycopeptide database
and deconvoluted an LC-MS/MS data file, you're ready to do just that.


.. click:: glycan_profiling.cli.analyze:search_glycopeptide
    :prog: glycresoft analyze search-glycopeptide


Usage Example
-------------

.. code-block:: bash

    $ glycresoft analyze search-glycopeptide -m 5e-6 -mn 1e-5 fasta-glycopeptides.db path/to/processed/sample.mzML 1\
         -o "agp-glycopepitdes-in-sample.db"


Memory Consumption and Workload Size
------------------------------------
Extensive use of caching and work-sharing has been done to make searching enormous
databases still tractable. If you find you are running out of memory during a search
consider shrinking the ``-w`` parameter.
