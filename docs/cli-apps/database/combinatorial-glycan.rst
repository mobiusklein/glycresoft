
.. _combinatorial-glycan-hypothesis:

Building a Combinatorial Glycan Hypothesis
==========================================

When you do not have a complete list of glycan compositions you
know are reasonable and present, a first pass can be done with a
combinatorial glycan composition hypothesis.

A combinatorial glycan hypothesis is defined by a set of ``component
ranges`` and a set of ``algebraic constraints``.

A ``component range`` is defined by a ``component`` residue such as a
monossaccharide (or any entity which can be encoded with `IUPAClite <todo>`_)
with a ``minimum`` count and a ``maximum`` count. Candidate glycan compositions will
contain the ``component`` with every possible value in the range between ``minimum``
and ``maximum``

An ``algebraic constraint`` is defined by two expressions containing numbers
binary arithmetic operators and variables corresponding to ``components`` and
a comparator operator relating the two expressions. No candidate glycan composition
will be included in the hypothesis will fail to satisfy all applied constraints.

.. code-block:: text

    HexNAc > (NeuAc + NeuGc + 1)

This constraint requires that the total number of sialylic acids always be one less
than the number of HexNAc, which is common for canonical N-glycans in humans.

To pass these instructions to the build tool, they must be written in a text file
following the format

.. code-block:: text

    ; comments are allowed on lines starting with a semi-colon
    <component>  <minimum>  <maximum>
    <component>  <minimum>  <maximum>
    ...
    ; a blank line separates the component range rules from the constrains

    <expression> <operator> <expression>
    <expression> <operator> <expression>
    ...


.. warning::

    Due to the average size of a combinatorial search space and limitations of composition
    motifs, glycan compositions produced by this program are only checked for being N-glycans,
    and as such you cannot tag compositions O-glycans directly with this tool at this time. This
    restriction is wholly artificial and may be removed in the future. For more information
    on glycan composition classifiers, please see `Glycan Composition Classes <todo>`_


.. click:: glycan_profiling.cli.build_db:glycan_combinatorial
    :prog: glycresoft build-hypothesis glycan-combinatorial

For more information on reductions and derivatizations, please see `Glycan Modifications <todo>`_

Example Usage
-------------

.. code-block:: bash

    $ cat rules-file.txt
    Hex 3 10
    HexNAc 2 9
    Fuc 0 5
    Neu5Ac 0 4

    Fuc < HexNAc
    HexNAc > NeuAc + 1

    $ glycresoft build-hypothesis glycan-combinatorial rules-file.txt combinatorial-database -n "Combinatorial Human N-Glycans"
    Building Glycan Hypothesis Combinatorial Human N-Glycans
    13:12:06 - glycresoft:log:175 - INFO - Begin Combinatorial Glycan Hypothesis Serializer
    {'derivatization': None,
     'engine': Engine(sqlite:///combinatorial-database),
     'glycan_file': u'rules-file.txt',
     'loader': None,
     'reduction': None,
     'start_time': datetime.datetime(2017, 8, 31, 13, 12, 6, 105000),
     'status': 'started',
     'transformer': None,
     'uuid': '459bffe3d9fd42019eb202c7afb3f72f'}
    13:12:06 - glycresoft:log:175 - INFO - Generating Glycan Compositions from Symbolic Rules for GlycanHypothesis(id=1, name=Combinatorial Human N-Glycans)
    13:12:08 - glycresoft:log:175 - INFO - 1000 glycan compositions created
    13:12:09 - glycresoft:log:175 - INFO - Generated 1280 glycan compositions
    13:12:09 - glycresoft:log:175 - INFO - Hypothesis Completed
    13:12:09 - glycresoft:log:175 - INFO - End Combinatorial Glycan Hypothesis Serializer
    13:12:09 - glycresoft:log:175 - INFO - Started at 2017-08-31 13:12:06.105000.
    Ended at 2017-08-31 13:12:09.047000.
    Total time elapsed: 0:00:02.942000
    CombinatorialGlycanHypothesisSerializer completed successfully.

