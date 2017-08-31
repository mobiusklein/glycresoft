import re
from itertools import product
from collections import deque

from glypy import GlycanComposition as MemoryGlycanComposition
from glypy.composition.glycan_composition import FrozenMonosaccharideResidue
from glypy.io.iupac import IUPACError
from glycan_profiling.serialize.hypothesis.glycan import GlycanComposition as DBGlycanComposition

from glypy.composition import formula

from .glycan_source import (
    GlycanTransformer, GlycanHypothesisSerializerBase, GlycanCompositionToClass)
from glycan_profiling.symbolic_expression import (
    ConstraintExpression, SymbolContext, SymbolNode, ExpressionNode,
    ensuretext)


def descending_combination_counter(counter):
    keys = counter.keys()
    count_ranges = map(lambda lo_hi: range(
        lo_hi[0], lo_hi[1] + 1), counter.values())
    for combination in product(*count_ranges):
        yield dict(zip(keys, combination))


def normalize_iupac_lite(string):
    return str(FrozenMonosaccharideResidue.from_iupac_lite(string))


class CombinatoricCompositionGenerator(object):
    """
    Summary

    Attributes
    ----------
    constraints : list
        Description
    lower_bound : list
        Description
    residue_list : list
        Description
    rules_table : dict
        Description
    upper_bound : list
        Description
    """
    @staticmethod
    def build_rules_table(residue_list, lower_bound, upper_bound):
        rules_table = {}
        for i, residue in enumerate(residue_list):
            lower = lower_bound[i]
            upper = upper_bound[i]
            rules_table[residue] = (lower, upper)
        return rules_table

    def __init__(self, residue_list=None, lower_bound=None, upper_bound=None, constraints=None, rules_table=None,
                 structure_classifiers=None):
        if structure_classifiers is None:
            structure_classifiers = {"N-Glycan": is_n_glycan_classifier}
        self.residue_list = list(map(normalize_iupac_lite, (residue_list or [])))
        self.lower_bound = lower_bound or []
        self.upper_bound = upper_bound or []
        self.constraints = constraints or []
        self.rules_table = rules_table
        self.structure_classifiers = structure_classifiers

        if len(self.constraints) > 0 and not isinstance(self.constraints[0], ConstraintExpression):
            self.constraints = list(
                map(ConstraintExpression.from_list, self.constraints))

        if rules_table is None:
            self._build_rules_table()

        self.normalize_rules_table()
        for constraint in self.constraints:
            self.normalize_symbols(constraint.expression)

        self._iter = None

    def normalize_rules_table(self):
        rules_table = {}
        for key, bounds in self.rules_table.items():
            rules_table[normalize_iupac_lite(key)] = bounds
        self.rules_table = rules_table

    def normalize_symbols(self, expression):
        symbol_stack = deque([expression])
        while symbol_stack:
            node = symbol_stack.popleft()
            if isinstance(node, ExpressionNode):
                symbol_stack.append(node.left)
                symbol_stack.append(node.right)
            elif isinstance(node, SymbolNode):
                node.symbol = normalize_iupac_lite(node.symbol)

    def _build_rules_table(self):
        rules_table = {}
        for i, residue in enumerate(self.residue_list):
            lower = self.lower_bound[i]
            upper = self.upper_bound[i]
            rules_table[residue] = (lower, upper)
        self.rules_table = rules_table
        return rules_table

    def generate(self):
        for combin in descending_combination_counter(self.rules_table):
            passed = True
            combin = SymbolContext(combin)
            structure_classes = []
            if sum(combin.values()) == 0:
                continue
            for constraint in self.constraints:
                if not constraint(combin):
                    passed = False
                    break
            if passed:
                for name, classifier in self.structure_classifiers.items():
                    if classifier(combin):
                        structure_classes.append(name)
                try:
                    yield MemoryGlycanComposition(**combin.context), structure_classes
                except IUPACError:
                    raise

    __iter__ = generate

    def __next__(self):
        if self._iter is None:
            self._iter = self.generate()
        return next(self._iter)

    next = __next__

    def __repr__(self):
        return "CombinatoricCompositionGenerator\n" + repr(self.rules_table) + '\n' + repr(self.constraints)


class CombinatorialGlycanHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, glycan_text_file, database_connection, reduction=None, derivatization=None,
                 hypothesis_name=None):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)

        self.glycan_file = glycan_text_file
        self.reduction = reduction
        self.derivatization = derivatization

        self.loader = None
        self.transformer = None

    def make_pipeline(self):
        rules, constraints = parse_rules_from_file(self.glycan_file)
        self.loader = CombinatoricCompositionGenerator(rules_table=rules, constraints=constraints)
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def run(self):
        self.make_pipeline()
        structure_class_lookup = self.structure_class_loader
        acc = []
        self.log("Generating Glycan Compositions from Symbolic Rules for %r" % self.hypothesis)
        counter = 0
        for composition, structure_classes in self.transformer:
            mass = composition.mass()
            composition_string = composition.serialize()
            formula_string = formula(composition.total_composition())
            inst = DBGlycanComposition(
                calculated_mass=mass, formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id)
            counter += 1
            self.session.add(inst)
            self.session.flush()
            for structure_class in structure_classes:
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if len(acc) % 100 == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
            if counter % 1000 == 0:
                self.log("%d glycan compositions created" % (counter,))
        if acc:
            self.session.execute(GlycanCompositionToClass.insert(), acc)
            acc = []
        self.session.commit()
        self.log("Generated %d glycan compositions" % counter)


def tryopen(obj):
    if hasattr(obj, "read"):
        return obj
    else:
        return open(obj)


def parse_rules_from_file(path):
    """
    Summary

    Parameters
    ----------
    path : TYPE
        Description

    Raises
    ------
    Exception
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    ranges = []
    constraints = []
    stream = tryopen(path)

    i = 0

    def cast(parts):
        return parts[0], int(parts[1]), int(parts[2])

    for line in stream:
        i += 1
        line = ensuretext(line)

        if line.startswith(";"):
            continue
        parts = re.split(r"\s+", line.replace("\n", ""))
        try:
            if len(parts) == 3:
                ranges.append(cast(parts))
            elif len(parts) == 1:
                if parts[0] in ["\n", "\r", ""]:
                    break
                else:
                    raise Exception("Could not interpret line '%r' at line %d" % (parts, i))
        except Exception as e:
            raise Exception("Failed to interpret line %r at line %d (%r)" % (parts, i, e))

    for line in stream:
        i += 1
        line = ensuretext(line)
        if line.startswith(";"):
            continue
        line = line.replace("\n", "")
        if line in ["\n", "\r", ""]:
            break
        try:
            constraints.append(ConstraintExpression.parse(line))
        except Exception as e:
            raise Exception("Failed to interpret line %r at line %d (%r)" % (line, i, e))

    rules_table = CombinatoricCompositionGenerator.build_rules_table(
        *zip(*ranges))
    try:
        stream.close()
    except Exception:
        pass
    return rules_table, constraints


def write_rules(rules, writer):
    for key, bounds in rules.items():
        lo, hi = bounds
        writer.write("%s %d %d\n" % (key, lo, hi))
    return writer


def write_constraints(constraints, writer):
    writer.write("\n")
    for constraint in constraints:
        writer.write("%s\n" % constraint)
    return writer


class ClassificationConstraint(object):
    """
    Express a classification (label) of a solution by satisfaction
    of a boolean Expression

    Attributes
    ----------
    classification : str
        The label to grant on satisfaction of `constraints`
    constraint : ConstraintExpression
        The constraint that must be satisfied.
    """

    def __init__(self, classification, constraint):
        self.classification = classification
        self.constraint = constraint

    def __call__(self, context):
        """
        Test for satisfaction.

        Parameters
        ----------
        context : SymbolContext

        Returns
        -------
        bool
            Whether the constraint is satisfied
        """
        return self.constraint(context)

    def __repr__(self):
        return "{}\n{}".format(self.classification, self.constraint)


is_n_glycan_classifier = ClassificationConstraint(
    "N-Glycan", (
        ConstraintExpression.from_list(
            ["HexNAc", ">=", "2"]) & ConstraintExpression.from_list(
            ["Hex", ">=", "3"]))
)

is_o_glycan_classifier = ClassificationConstraint(
    "O-Glycan", ((ConstraintExpression.from_list(["HexNAc", ">=", "1"]) &
                  ConstraintExpression.from_list(["Hex", ">=", "1"])) |
                 (ConstraintExpression.from_list(["HexNAc", ">=", "2"])))
)
