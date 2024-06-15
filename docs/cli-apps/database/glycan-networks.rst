Glycome Network Definitions
===========================

The glycome network smoothing method presented in "Klein, J., Carvalho, L., & Zaia, J. (2018). Application of network smoothing to glycan LC-MS profiling.
Bioinformatics, 34(20), 3511-3518. https://doi.org/10.1093/bioinformatics/bty397" requires a network definition.

.. _build-glycan-graph:

Building a Network
------------------

The first step is to build a network for a glycan list, embodied in a :class:`~.GlycanHypothesis` in a database file.
This produces a new text file defining the nodes of the network and builds edges between them.

.. click:: glycresoft.cli.build_db:glycan_network
    :prog: glycresoft build-hypothesis glycan-network build-network


.. _add-predefined-neighborhood-glycan-graph:

Adding Pre-Defined Network Neighborhoods
----------------------------------------

Add pre-programmed neighborhood rules to a network, writing them out to a new text file.

.. click:: glycresoft.cli.build_db:add_prebuild_neighborhoods_to_network
    :prog: glycresoft build-hypothesis glycan-network add-prebuilt-neighborhoods


.. _add-custom-neighborhood-rule-glycan-graph:

Adding a Custom Network Neighborhood
------------------------------------

Add a custom neighborhood to a network, writing them out to a new text file. A neighborhood is
composed of one or more rule expressions.

.. click:: glycresoft.cli.build_db:add_neighborhood_to_network
    :prog: glycresoft build-hypothesis glycan-network add-neighborhood
