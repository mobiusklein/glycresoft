import json
from collections import namedtuple

import numpy as np

from glypy.structure.glycan_composition import HashableGlycanComposition

MINIMUM = 1e-4


glycan_composition_cache = dict()
decoy_glycan_cache = dict()

def parse_glycan_composition(string):
    try:
        return glycan_composition_cache[string]
    except KeyError:
        gc = HashableGlycanComposition.parse(string)
        glycan_composition_cache[string] = gc
        return gc


def to_decoy_glycan(string):
    try:
        return decoy_glycan_cache[string]
    except KeyError:
        gc = HashableGlycanComposition.parse(string)
        gc["#decoy#"] = 2
        decoy_glycan_cache[string] = gc
        return gc


def is_decoy_glycan(string):
    return "#decoy#" in string


GlycanPriorRecord = namedtuple("GlycanPriorRecord", ("score", "matched"))

try:
    from glycan_profiling._c.composition_distribution_model.utils import GlycanPriorRecord
except ImportError:
    pass


class GlycosylationSiteModel(object):

    def __init__(self, protein_name, position, site_distribution, lmbda, glycan_map):
        self.protein_name = protein_name
        self.position = position
        self.site_distribution = site_distribution
        self.lmbda = lmbda
        self.glycan_map = glycan_map

    def __getitem__(self, key):
        return self.glycan_map[key][0]

    def get_record(self, key):
        try:
            return self.glycan_map[key]
        except KeyError:
            return GlycanPriorRecord(MINIMUM, False)

    def __eq__(self, other):
        if self.protein_name != other.protein_name:
            return False
        if self.position != other.position:
            return False
        if not np.isclose(self.lmbda, other.lmbda):
            return False
        if self.site_distribution != other.site_distribution:
            return False
        if self.glycan_map != other.glycan_map:
            return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def to_dict(self):
        d = {}
        d['protein_name'] = self.protein_name
        d['position'] = self.position
        d['lmbda'] = self.lmbda
        d['site_distribution'] = dict(**self.site_distribution)
        d['glycan_map'] = {
            str(k): (v.score, v.matched) for k, v in self.glycan_map.items()
        }
        return d

    @classmethod
    def from_dict(cls, d):
        name = d['protein_name']
        position = d['position']
        lmbda = d['lmbda']
        try:
            site_distribution = d['site_distribution']
        except KeyError:
            site_distribution = d['tau']
        glycan_map = d['glycan_map']
        glycan_map = {
            parse_glycan_composition(k): GlycanPriorRecord(v[0], v[1])
            for k, v in glycan_map.items()
        }
        inst = cls(name, position, site_distribution, lmbda, glycan_map)
        return inst

    def pack(self, inplace=True):
        if inplace:
            self._pack()
            return self
        return self.copy()._pack()

    def _pack(self):
        new_map = {}
        for key, value in self.glycan_map.items():
            if value.score > MINIMUM:
                new_map[key] = value
        self.glycan_map = new_map
        return self

    def __repr__(self):
        template = ('{self.__class__.__name__}({self.protein_name!r}, {self.position}, '
                    '{site_distribution}, {self.lmbda}, <{glycan_map_size} Glycans>)')
        glycan_map_size = len(self.glycan_map)
        site_distribution = {k: v for k,
                             v in self.site_distribution.items() if v > 0.0}
        return template.format(self=self, glycan_map_size=glycan_map_size, site_distribution=site_distribution)

    def copy(self, deep=False):
        dup = self.__class__(
            self.protein_name, self.position, self.site_distribution, self.lmbda, self.glycan_map)
        if deep:
            dup.site_distribution = dup.site_distribution.copy()
            dup.glycan_map = dup.glycan_map.copy()
        return dup

    def clone(self, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def observed_glycans(self, threshold=0):
        return {k: v.score for k, v in self.glycan_map.items() if v.matched and v.score > threshold}

    @classmethod
    def load(cls, fh):
        site_dicts = json.load(fh)
        site_models = [cls.from_dict(d) for d in site_dicts]
        return site_models

    @classmethod
    def dump(cls, instances, fh):
        site_dicts = [d.to_dict() for d in instances]
        json.dump(site_dicts, fh)

    def equalize_decoys(self):
        for glycan in self.glycan_map.keys():
            if not is_decoy_glycan(glycan):
                record = self.get_record(glycan)
                new_key = to_decoy_glycan(glycan)
                self.glycan_map[new_key] = record
        return self
