from collections import defaultdict, Counter
import itertools

import glypy
from glypy.composition.glycan_composition import FrozenGlycanComposition
from glypy.composition import formula
from glypy import ReducedEnd

from .glycan_source import GlycanHypothesisSerializerBase
from glycan_profiling.serialize import (
    DatabaseBoundOperation,
    GlycanComposition as DBGlycanComposition,
    GlycanCompositionToClass)


def get_glycan_composition(glycan_composition):
    try:
        return glycan_composition.convert()
    except AttributeError:
        return glycan_composition


def linearize_composition(glycan_composition):
    return [mr for mr, count in glycan_composition.items() for i in range(count)]


class GlycanCompositionTransformerRule(object):
    def __init__(self, target, modifier):
        self.target = target
        self.modifier = modifier

    def find_valid_sites(self, linear):
        return [i for i, mr in enumerate(linear) if mr == self.target]

    def __repr__(self):
        return "GlycanCompositionTransformerRule(%r, %r)" % (
            self.target, self.modifier)

    def apply(self, glycan_composition, i):
        glycan_composition[self.modifier] = i
        return glycan_composition

    def __eq__(self, other):
        return self.target == other.target and self.modifier == other.modifier

    def __hash__(self):
        return hash(self.target)


class GlycanReductionTransformerRule(object):
    def __init__(self, reduction):
        if isinstance(reduction, ReducedEnd):
            reduction = reduction.clone()
        elif isinstance(reduction, glypy.Composition):
            reduction = ReducedEnd(reduction.clone())
        elif isinstance(reduction, basestring):
            reduction = ReducedEnd(glypy.Composition(reduction))
        self.reduction = reduction

    def apply(self, glycan_composition):
        glycan_composition.reducing_end = self.reduction.clone()
        return glycan_composition


def split_reduction_and_modification_rules(rule_list):
    modification_rules = []
    reduction_rules = []
    for rule in rule_list:
        if isinstance(rule, GlycanReductionTransformerRule):
            reduction_rules.append(rule)
        else:
            modification_rules.append(rule)
    return modification_rules, reduction_rules


def modification_series(variable_sites):
    """Given a dictionary mapping between modification names and
    an iterable of valid sites, create a dictionary mapping between
    modification names and a list of valid sites plus the constant `None`

    Parameters
    ----------
    variable_sites : dict
        Description

    Returns
    -------
    dict
        Description
    """
    sites = defaultdict(list)
    for mod, varsites in variable_sites.items():
        for site in varsites:
            sites[site].append(mod)
    for site in list(sites):
        sites[site].append(None)
    return sites


def site_modification_assigner(modification_sites_dict):
    sites = modification_sites_dict.keys()
    choices = modification_sites_dict.values()
    for selected in itertools.product(*choices):
        yield zip(sites, selected)


def simplify_assignments(assignments_generator):
    distinct = set()
    for assignments in assignments_generator:
        distinct.add(frozenset(Counter(mod for site, mod in assignments if mod is not None).items()))
    return distinct


def glycan_composition_permutations(glycan_composition, constant_modifications=None, variable_modifications=None):
    if constant_modifications is None:
        constant_modifications = []
    if variable_modifications is None:
        variable_modifications = []

    glycan_composition = FrozenGlycanComposition(
        get_glycan_composition(glycan_composition))
    final_composition = FrozenGlycanComposition()
    temporary_composition = FrozenGlycanComposition()
    sequence = linearize_composition(glycan_composition)

    if not constant_modifications:
        temporary_composition = glycan_composition.clone()
        has_constant_reduction = False
    else:
        constant_modifications, constant_reduction = split_reduction_and_modification_rules(
            constant_modifications)
        for rule in constant_modifications:
            extracted = rule.find_valid_sites(sequence)
            for i, mr in enumerate(sequence):
                if i in extracted:
                    final_composition[mr] += 1
                    final_composition[rule.modifier] += 1
                else:
                    temporary_composition[mr] += 1
            sequence = linearize_composition(temporary_composition)
        if constant_reduction:
            has_constant_reduction = True
            constant_reduction[0].apply(temporary_composition)

    variable_modifications, variable_reduction = split_reduction_and_modification_rules(
        variable_modifications)

    mod_site_map = {
        rule: rule.find_valid_sites(sequence) for rule in variable_modifications
    }
    assignments_generator = site_modification_assigner(
        modification_series(mod_site_map))
    for assignments in simplify_assignments(assignments_generator):
        result = temporary_composition.clone()
        for mod, count in assignments:
            mod.apply(result, count)
        result += final_composition
        if glycan_composition.reducing_end is not None:
            result.reducing_end = glycan_composition.reducing_end.clone()
        yield result
        if not has_constant_reduction:
            for rule in variable_reduction:
                yield rule.apply(result.clone())


class GlycanCompositionPermutationHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, source_hypothesis_id, database_connection, constant_modifications=None,
                 variable_modifications=None, hypothesis_name=None):
        super(GlycanCompositionPermutationHypothesisSerializer, self).__init__(database_connection, hypothesis_name)
        self.source_hypothesis_id = source_hypothesis_id
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications

        self.loader = None
        self.transformer = None

    def make_pipeline(self):
        self.loader = iter(DatabaseBoundOperation(
            self._original_connection).query(
            DBGlycanComposition).filter(
            DBGlycanComposition.hypothesis_id == self.source_hypothesis_id).all())
        self.transformer = self.permuter(self.loader)

    def permuter(self, source_iterator):
        for glycan_composition in source_iterator:
            structure_classes = [sc.name for sc in glycan_composition.structure_classes]
            for permutation in glycan_composition_permutations(
                    glycan_composition, self.constant_modifications, self.variable_modifications):
                yield permutation, structure_classes

    def run(self):
        self.make_pipeline()
        structure_class_lookup = self.structure_class_loader
        self.log("Loading Glycan Compositions from Stream for %r" % self.hypothesis)

        acc = []
        for composition, structure_classes in self.transformer:
            mass = composition.mass()
            composition_string = composition.serialize()
            formula_string = formula(composition.total_composition())
            inst = DBGlycanComposition(
                calculated_mass=mass, formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id)
            self.session.add(inst)
            self.session.flush()
            for structure_class in structure_classes:
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if len(acc) % 100 == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
        if acc:
            self.session.execute(GlycanCompositionToClass.insert(), acc)
            acc = []
        self.session.commit()
