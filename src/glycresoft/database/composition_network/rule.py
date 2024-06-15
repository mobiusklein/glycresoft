from io import StringIO
from typing import Union, Protocol

from glycopeptidepy import HashableGlycanComposition
from glycopeptidepy.utils import simple_repr

from glycresoft import symbolic_expression


class HasGlycanComposition(Protocol):
    glycan_composition: HashableGlycanComposition


AsGlycanComposition = Union[HashableGlycanComposition, str, HasGlycanComposition]


class CompositionRuleBase(object):

    __repr__ = simple_repr

    def __call__(self, obj: AsGlycanComposition) -> bool:
        raise NotImplementedError()

    def get_composition(self, obj: AsGlycanComposition) -> symbolic_expression.GlycanSymbolContext:
        try:
            composition = obj.glycan_composition
        except AttributeError:
            composition = HashableGlycanComposition.parse(obj)
        composition = symbolic_expression.GlycanSymbolContext(composition)
        return composition

    def __and__(self, other):
        if isinstance(other, CompositionRuleClassifier):
            other = other.copy()
            other.rules.append(self)
            return other
        else:
            return CompositionRuleClassifier(None, [self, other])

    def get_symbols(self):
        raise NotImplementedError()

    @property
    def symbols(self):
        return self.get_symbols()

    def is_univariate(self):
        return len(self.get_symbols()) == 1

    def serialize(self):
        raise NotImplementedError()

    @classmethod
    def parse(cls, line, handle=None):
        raise NotImplementedError()


def int_or_none(x):
    try:
        return int(x)
    except ValueError:
        return None


class CompositionExpressionRule(CompositionRuleBase):
    def __init__(self, expression, required=True):
        self.expression = symbolic_expression.ExpressionNode.parse(str(expression))
        self.required = required

    def get_symbols(self):
        return self.expression.get_symbols()

    def __call__(self, obj):
        composition = self.get_composition(obj)
        if composition.partially_defined(self.expression):
            return composition[self.expression]
        else:
            if self.required:
                return False
            else:
                return True

    def serialize(self):
        tokens = ["CompositionExpressionRule", str(self.expression),
                  str(self.required)]
        return '\t'.join(tokens)

    @classmethod
    def parse(cls, line, handle=None):
        tokens = line.strip().split("\t")
        n = len(tokens)
        i = 0
        while tokens[i] != "CompositionExpressionRule" and i < n:
            i += 1
        i += 1
        if i >= n:
            raise ValueError("Coult not parse %r with %s" % (line, cls))
        expr = symbolic_expression.parse_expression(tokens[i])
        required = tokens[i + 1].lower() in ('true', 'yes', '1')
        return cls(expr, required)

    def __repr__(self):
        template = "{self.__class__.__name__}(expression={self.expression}, required={self.required})"
        return template.format(self=self)


class CompositionRangeRule(CompositionRuleBase):

    def __init__(self, expression, low=None, high=None, required=True):
        self.expression = symbolic_expression.ExpressionNode.parse(str(expression))
        self.low = low
        self.high = high
        self.required = required

    def __repr__(self):
        template = \
            ("{self.__class__.__name__}(expression={self.expression}, "
             "low={self.low}, high={self.high}, required={self.required})")
        return template.format(self=self)

    def get_symbols(self):
        return self.expression.get_symbols()

    def __call__(self, obj: AsGlycanComposition) -> bool:
        composition = self.get_composition(obj)
        if composition.partially_defined(self.expression):
            if self.low is None:
                return composition[self.expression] <= self.high
            elif self.high is None:
                return self.low <= composition[self.expression]
            return self.low <= composition[self.expression] <= self.high
        else:
            if self.required and self.low > 0:
                return False
            else:
                return True

    def serialize(self):
        tokens = ["CompositionRangeRule", str(self.expression), str(self.low),
                  str(self.high), str(self.required)]
        return '\t'.join(tokens)

    @classmethod
    def parse(cls, line, handle=None):
        tokens = line.strip().split("\t")
        n = len(tokens)
        i = 0
        while tokens[i] != "CompositionRangeRule" and i < n:
            i += 1
        i += 1
        if i >= n:
            raise ValueError("Coult not parse %r with %s" % (line, cls))
        expr = symbolic_expression.parse_expression(tokens[i])
        low = int_or_none(tokens[i + 1])
        high = int_or_none(tokens[i + 2])
        required = tokens[i + 3].lower() in ('true', 'yes', '1')
        return cls(expr, low, high, required)


class CompositionRatioRule(CompositionRuleBase):
    def __init__(self, numerator, denominator, ratio_threshold, required=True):
        self.numerator = numerator
        self.denominator = denominator
        self.ratio_threshold = ratio_threshold
        self.required = required

    def __repr__(self):
        template = \
            ("{self.__class__.__name__}(numerator={self.numerator}, "
             "denominator={self.denominator}, ratio_threshold={self.ratio_threshold}, "
             "required={self.required})")
        return template.format(self=self)

    def _test(self, x):
        if isinstance(self.ratio_threshold, (tuple, list)):
            return self.ratio_threshold[0] <= x < self.ratio_threshold[1]
        else:
            return x >= self.ratio_threshold

    def get_symbols(self):
        return (self.numerator, self.denominator)

    def __call__(self, obj: AsGlycanComposition) -> bool:
        composition = self.get_composition(obj)
        val = composition[self.numerator]
        ref = composition[self.denominator]

        if ref == 0 and self.required:
            return False
        else:
            ratio = val / float(ref)
            return self._test(ratio)

    def serialize(self):
        tokens = ["CompositionRatioRule", str(self.numerator), str(self.denominator),
                  str(self.ratio_threshold), str(self.required)]
        return '\t'.join(tokens)

    @classmethod
    def parse(cls, line, handle=None):
        tokens = line.strip().split("\t")
        n = len(tokens)
        i = 0
        while tokens[i] != "CompositionRatioRule" and i < n:
            i += 1
        i += 1
        numerator = symbolic_expression.parse_expression(tokens[i])
        denominator = symbolic_expression.parse_expression(tokens[i + 1])
        ratio_threshold = float(tokens[i + 2])
        required = tokens[i + 3].lower() in ('true', 'yes', '1')
        return cls(numerator, denominator, ratio_threshold, required)


class CompositionRuleClassifier(object):

    def __init__(self, name, rules):
        self.name = name
        self.rules = rules

    def __iter__(self):
        return iter(self.rules)

    def __call__(self, obj: AsGlycanComposition) -> bool:
        for rule in self:
            if not rule(obj):
                return False
        return True

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            return self.name == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    __repr__ = simple_repr

    def copy(self):
        return CompositionRuleClassifier(self.name, list(self.rules))

    def __and__(self, other):
        if isinstance(other, CompositionRuleClassifier):
            other = other.copy()
            other.rules.extend(self.rules)
            return other
        else:
            self = self.copy()
            self.rules.append(other)
            return self

    def get_symbols(self):
        symbols = set()
        for rule in self:
            symbols.update(rule.symbols)
        return symbols

    @property
    def symbols(self):
        return self.get_symbols()

    def serialize(self):
        text_buffer = StringIO()
        text_buffer.write("CompositionRuleClassifier\t%s\n" % (self.name,))
        for rule in self:
            text = rule.serialize()
            text_buffer.write("\t%s\n" % (text,))
        text_buffer.seek(0)
        return text_buffer.read()

    @classmethod
    def parse(cls, lines):
        line = lines[0]
        name = line.strip().split("\t")[1]
        rules = []
        for line in lines[1:]:
            if line == "":
                continue
            rule_type = line.split("\t")[0]
            if rule_type == "CompositionRangeRule":
                rule = CompositionRangeRule.parse(line)
            elif rule_type == "CompositionRatioRule":
                rule = CompositionRatioRule.parse(line)
            elif rule_type == "CompositionExpressionRule":
                rule = CompositionExpressionRule.parse(line)
            else:
                raise ValueError("Unrecognized Rule Type: %r" % (line,))
            rules.append(rule)
        return cls(name, rules)
