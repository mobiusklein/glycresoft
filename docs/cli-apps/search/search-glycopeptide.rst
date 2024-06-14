Searching a Processed Sample with a Glycopeptide Database
============================================================

The end-goal of all of these tools is to be able to identify glycopeptides
from experimental data. After you've constructed a glycopeptide database
and deconvoluted an LC-MS/MS data file, you're ready to do just that.

.. _search-glycopeptide:

Traditional Database Search
----------------------------

.. click:: glycan_profiling.cli.analyze:search_glycopeptide
    :prog: glycresoft analyze search-glycopeptide


Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

    $ glycresoft analyze search-glycopeptide -m 5e-6 -mn 1e-5 fasta-glycopeptides.db path/to/processed/sample.mzML 1\
         -o "agp-glycopepitdes-in-sample.db"


.. _search-glycopeptide-multipart:

Multi-component Database Search
-------------------------------

.. click:: glycan_profiling.cli.analyze:search_glycopeptide_multipart
    :prog: glycresoft analyze search-glycopeptide-multipart

Usage Example
~~~~~~~~~~~~~

Please see the :ref:`SCE Tutorial for example usage <sce_glycopeptide_tutorial>`


Memory Consumption and Workload Size
------------------------------------
Extensive use of caching and work-sharing has been done to make searching enormous
databases still tractable. If you find you are running out of memory during a search
consider shrinking the ``-w`` parameter.


.. _build-glycosite-model:

Build a Glycosite Network Smoothing Model
------------------------------------------

.. click:: glycan_profiling.cli.analyze:fit_glycoproteome_model
    :prog: glycresoft analyze fit-glycoproteome-smoothing-model


Adducts
-------

Unlike the glycan search tool, the glycopeptide search tool does not apply combinatorial expansion of adducts.
It will not mix mass shifts of different types together, so if both ``Ammonium 2`` and ``Na1H-1 1`` are specified,
the algorithm will only search for 0, 1, or 2 ``Ammonium`` shifts and 0 or 1 ``Na1H-1`` shifts. This is in order to
keep the search space tractable, but also in tested datasets, most multiply adducted ion species are low in abundance.