.. _glycan_classes:

Glycan Composition Classes
--------------------------

GlycReSoft treats glycan compositions as N-glycans, O-glycans, or GAG-linkers based upon
user assertion. These do not directly impact their use in glycan database searches, but
they are required for glycopeptide database construction.


Class Names
============

When referred to in text, these are the expected names for these classes. They are parsed case-insensitively.

- ``N-glycan``: *N*-linked glycans. Assumed to have a chitobiose core.
- ``O-glycan``: *O*-linked glycans. The default assumption is mucin-type *O*-glycan but not strictly required.
- ``GAG-linker``: Glycosaminoglycan linker saccharide. Assumed to contain the core trisaccharide.