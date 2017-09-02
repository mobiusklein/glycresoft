.. _text-glycan-hypothesis:

Building a Glycan Hypothesis from a Text File
=============================================

When a comprehensive list of glycan compositions is available,
you can translate them into a list encoded in `IUPAClite <todo>`_
with one composition per line. Lines may have the form

.. code-block:: Text

    <composition>  [<classifier>[  <classifier>...]]
    <composition>  [<classifier>[  <classifier>...]]
    ...

where classifier is one of the recognized glycan composition classifiers
such as ``N-glycan``, ``O-glycan``, or ``GAG-linker``. For more information
on glycan composition classifiers, please see `Glycan Composition Classes <todo>`_.


.. click:: glycan_profiling.cli.build_db:glycan_text
    :prog: glycresoft build-hypothesis glycan-text

For more information on reductions and derivatizations, please see `Glycan Modifications <todo>`_


Usage Example
-------------

.. code-block:: bash
    
    $ head glycan-compositions.txt
    {Hex:5; HexNAc:4; Neu5Ac:1}  N-Glycan
    {Hex:5; HexNAc:4; Neu5Ac:2}  N-Glycan
    {Fuc:1; Hex:5; HexNAc:4; Neu5Ac:2}  N-Glycan
    {Fuc:2; Hex:6; HexNAc:5; Neu5Ac:1}  N-Glycan
    {Fuc:1; Hex:6; HexNAc:5; Neu5Ac:2}  N-Glycan

    $ wc -l glycan-compositions.txt
    41 glycan-compositions.txt

    $ glycresoft build-hypothesis glycan-text glycan-compositions.txt glycan-list.db -n "N-Glycan short list"
    Building Glycan Hypothesis N-Glycan short list
    13:51:54 - glycresoft:log:175 - INFO - Begin Text File Glycan Hypothesis Serializer
    {'derivatization': None,
     'engine': Engine(sqlite:///glycan-list.db),
     'glycan_file': u'glycan-compositions.txt',
     'loader': None,
     'reduction': None,
     'start_time': datetime.datetime(2017, 8, 31, 13, 51, 54, 636000),
     'status': 'started',
     'transformer': None,
     'uuid': '10e06326097c41f4beeeddd8e17bbc0e'}
    13:51:54 - glycresoft:log:175 - INFO - Loading Glycan Compositions from Stream for GlycanHypothesis(id=1, name=N-Glycan short list)
    13:51:54 - glycresoft:log:175 - INFO - Generated 41 glycan compositions
    13:51:54 - glycresoft:log:175 - INFO - Hypothesis Completed
    13:51:54 - glycresoft:log:175 - INFO - End Text File Glycan Hypothesis Serializer
    13:51:54 - glycresoft:log:175 - INFO - Started at 2017-08-31 13:51:54.636000.
    Ended at 2017-08-31 13:51:54.761000.
    Total time elapsed: 0:00:00.125000
    TextFileGlycanHypothesisSerializer completed successfully.
