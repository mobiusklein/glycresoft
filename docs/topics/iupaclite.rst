IUPAClite Monosaccharide and Glycan Notation
--------------------------------------------

GlycReSoft uses :mod:`glypy`'s :term:`IUPAClite` notation for monosaccharides and glycan compositions.

A monosaccharide is denoted using IUPAC notation, omitting ring shape, anomeric state, chirality,
and modification positions (optionally). For example, ``a-D-Manp`` would be written ``Man``, or
``b-D-Glcp2NAc`` would be written ``GlcNAc``.

You can also use generic base types like ``Hex`` or ``Pen`` for example, to denote a six or five
carbon monosaccharide. The notation is composable, so you can specify an arbitrarily modified
monosaccharide, like ``HexNAc(S)`` to specify a sulfated HexNAc, using the parenthesized convention
that separates substituent groups, or ``dHexN`` for a deoxy-Hexosamine.

You can also define "floating" substituent groups by prefixing their full lowercase
names with an ``@``-sign, like ``@sulfate`` for sulfate or ``@acetyl`` for an acetyl group.

A glycan composition is written as one or more ``<monosaccharide>:<count>`` occurrences separated by
a "; " (semi-colon + space), enclosed in "{ }". A few examples are shown below:

.. code-block:: python
    :caption: IUPAClite glycan compositions

    {Hex:5; HexNAc:4; Neu5Ac:1}
    {Hex:5; HexNAc:4; Neu5Ac:2}
    {Fuc:1; Hex:5; HexNAc:4; Neu5Ac:2}
    {Fuc:2; Hex:6; HexNAc:5; Neu5Ac:1}
    {Fuc:1; Hex:6; HexNAc:5; Neu5Ac:2}
