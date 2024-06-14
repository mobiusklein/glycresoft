.. _peptide_modifications:

Peptide Modifications
---------------------

.. toctree::
    :maxdepth: 3

:mod:`glycresoft` supports the full range of `UNIMOD <http://www.unimod.org/modifications_list.php?>`_
modification rules as well as some common alternative namings. Targeting rules listed in this table are the known
specificities from the database, but arbitrary targets may be specified.

Additionally, custom modifications may be specified using the syntax ``@<name>-<mass>``, e.g. ``@custom modification-410.5``
for a modification "custom modification" with a mass delta of +410.5 Da, or ``@Antimatter--99999`` for a modification with the
name "Antimatter" with a mass delta of -99999.0 Da.

UNIMOD
======

.. warning:: If you wish to specify a glycan modification, please use GlycReSoft's system for glycan compositions, not UNIMOD's.

.. exec::

    from unimod_table import rendered_table
    print(rendered_table)


Targeting Rules
===============

When specifying modification target restrictions, following a modification rule's name, write ``(<target>)`` where ``<target>``
may be a series of amino acids or peptide terminals ``N-term`` or ``C-term`` or ``<AA> @ <terminal>``, e.g. ``Deamidated (N)``
to specify the Deamidated modification rule on asparagine.
