Database Building Tools
-----------------------

:mod:`glycresoft` requires that a search space, called a "hypothesis", be pre-specified prior to searching it. This is done explicitly so that
the database construction process is not repeated needlessly for multiple searches of the same space, and to allow slower
data organization and indexing steps to run independent of the main search program.

There are many different ways to build glycan and glycopeptide databases.

.. toctree::
   :maxdepth: 1

   Building a Combinatorial Glycan Hypothesis <combinatorial-glycan>
   Building a Glycan Hypothesis from a Text File <text-glycan>
   Building a Glycan Hypothesis from glySpace <glyspace-glycan>

   Building a Glycan Composition Graph <glycan-networks>

   Building a Glycopeptide Hypothesis from a FASTA File <fasta-glycopeptide>
