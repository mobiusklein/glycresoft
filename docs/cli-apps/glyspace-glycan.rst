.. _glyspace-glycan-hypothesis:

Building a Glycan Hypothesis from glySpace
==========================================

The glycoinformatics community has developed a federation of
databases called :title-reference:`glySpace`, which composes
the "namespace of all glycan structures". It uses semantic web
technologies to describe and relate these entities, and gateways
into :title-reference:`glySpace` such as [GlyTouCan]_
provide query-able access to this data.

You can build a glycan composition hypothesis from the set of annotated
glycan structures in :title-reference:`glySpace` for N-glycans and O-glycans,
with or without taxonomic filters.


.. click:: glycan_profiling.cli.build_db:glyspace_glycan_hypothesis
    :prog: glycresoft build-hypothesis glyspace-glycan

For more information on reductions and derivatizations, please see `Glycan Modifications <todo>`_


Usage Example
-------------

.. code-block:: bash

    # Get all human N-glycans
    $ glycresoft build-hypothesis glyspace-glycan -m n-linked -t 9606 glyspace-glycans.db -n "Human N-Linked Glycans"
    Building Glycan Hypothesis Human N-Linked Glycans (2)
    14:35:34 - glycresoft:log:175 - INFO - Begin N Glycan Glyspace Hypothesis Serializer
    {'composition_cache': None,
     'derivatization': None,
     'engine': Engine(sqlite:///glyspace-glycans.db),
     'filter_functions': [TaxonomyFilter({'9606'})],
     'loader': None,
     'reduction': None,
     'seen': {},
     'simplify': True,
     'start_time': datetime.datetime(2017, 8, 31, 14, 35, 34, 164000),
     'status': 'started',
     'transformer': None,
     'uuid': '471041e33f23485c9a570a5b4ea6e0d2'}
    14:35:34 - glycresoft:log:175 - INFO - Querying GlySpace
    14:35:54 - glycresoft:log:175 - INFO - Translating Response
    14:36:19 - glycresoft:log:175 - INFO - Stored 976 glycan structures and 195 glycan compositions
    14:38:24 - glycresoft:log:175 - INFO - Hypothesis Completed
    14:38:24 - glycresoft:log:175 - INFO - End N Glycan Glyspace Hypothesis Serializer
    14:38:24 - glycresoft:log:175 - INFO - Started at 2017-08-31 14:35:34.164000.
    Ended at 2017-08-31 14:38:24.535000.
    Total time elapsed: 0:02:50.371000
    NGlycanGlyspaceHypothesisSerializer completed successfully.

    # Get all human O-glycans
    $ glycresoft build-hypothesis glyspace-glycan -m o-linked -t 9606 glyspace-glycans.db -n "Human O-Linked Glycans"
    Building Glycan Hypothesis Human O-Linked Glycans
    15:33:49 - glycresoft:log:175 - INFO - Begin O Glycan Glyspace Hypothesis Serializer
    {'composition_cache': None,
     'derivatization': None,
     'engine': Engine(sqlite:///glyspace-glycans.db),
     'filter_functions': [TaxonomyFilter({'9606'})],
     'loader': None,
     'reduction': None,
     'seen': {},
     'simplify': True,
     'start_time': datetime.datetime(2017, 8, 31, 15, 33, 49, 601000),
     'status': 'started',
     'transformer': None,
     'uuid': '714d52c2525c499286f673e294672b9e'}
    15:33:49 - glycresoft:log:175 - INFO - Querying GlySpace
    15:33:55 - glycresoft:log:175 - INFO - Translating Response
    15:34:01 - glycresoft:log:175 - INFO - Stored 315 glycan structures and 95 glycan compositions
    15:34:01 - glycresoft:log:175 - INFO - Hypothesis Completed
    15:34:01 - glycresoft:log:175 - INFO - End O Glycan Glyspace Hypothesis Serializer
    15:34:01 - glycresoft:log:175 - INFO - Started at 2017-08-31 15:33:49.601000.
    Ended at 2017-08-31 15:34:01.737000.
    Total time elapsed: 0:00:12.136000
    OGlycanGlyspaceHypothesisSerializer completed successfully.

Bibliography
------------

.. [GlyTouCan] Aoki-Kinoshita, K., Agravat, S., Aoki, N. P., Arpinar, S., Cummings, R. D., Fujita, A., … Narimatsu, H. (2015).
               GlyTouCan 1.0 – The international glycan structure repository. Nucleic Acids Research, gkv1041.
               https://doi.org/10.1093/nar/gkv1041