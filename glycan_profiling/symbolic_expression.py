'''
Provides a simple symbolic algebra engine for expressing relationships between
symbols representing counts in terms of the arithmetic operators addition,
subtraction, multiplication, and division, and conditional operators greater
than, less than equality, as well as compound & and | relationships.
'''
import re
from numbers import Number

from six import string_types as basestring, PY3

from glypy.composition import composition_transform
from glypy.structure import glycan_composition


_FMR = glycan_composition.FrozenMonosaccharideResidue
_MR = glycan_composition.MonosaccharideResidue


def ensuretext(x):
    if isinstance(x, str):
        return x
    elif PY3:
        try:
            return x.decode("utf8")
        except AttributeError:
            return str(x)
    else:
        return str(x)


class ExpressionBase(object):
    @staticmethod
    def _coerce(v):
        if isinstance(v, ExpressionBase):
            return v
        elif isinstance(v, basestring):
            return SymbolNode.parse(ensuretext(v))
        elif isinstance(v, Number):
            return ValueNode(v)

    def _as_compound(self, op, other):
        return ExpressionNode(self, op, self._coerce(other))

    def __add__(self, other):
        return self._as_compound(Operator.get("+"), other)

    def __sub__(self, other):
        return self._as_compound(Operator.get("-"), other)

    def __mul__(self, other):
        return self._as_compound(Operator.get("*"), other)

    def __div__(self, other):
        return self._as_compound(Operator.get("/"), other)

    def __gt__(self, other):
        return self._as_compound(Operator.get(">"), other)

    def __ge__(self, other):
        return self._as_compound(Operator.get(">="), other)

    def __lt__(self, other):
        return self._as_compound(Operator.get("<"), other)

    def __le__(self, other):
        return self._as_compound(Operator.get("<="), other)

    def __eq__(self, other):
        return self._as_compound(Operator.get("=="), other)

    def __ne__(self, other):
        return self._as_compound(Operator.get("!="), other)

    def __and__(self, other):
        return self._as_compound(Operator.get("and"), other)

    def __or__(self, other):
        return self._as_compound(Operator.get("or"), other)

    def __call__(self, other):
        return self._as_compound(Operator.get("call"), other)

    def get_symbols(self):
        return []


class ConstraintExpression(ExpressionBase):
    """
    A wrapper around an ExpressionNode that is callable
    to test for satisfaction.

    Attributes
    ----------
    expression : ExpressionNode
        The symbolic expression that a composition must
        satisfy in order to be retained.
    """
    @classmethod
    def parse(cls, string):
        return ConstraintExpression(ExpressionNode.parse(string))

    @classmethod
    def from_list(cls, sym_list):
        return cls.parse(' '.join(sym_list))

    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return "{}".format(self.expression)

    def __call__(self, context):
        """
        Test for satisfaction of :attr:`expression`

        Parameters
        ----------
        context : Solution
            Context to evaluate :attr:`expression` in

        Returns
        -------
        bool
        """
        return context[self.expression]

    def __and__(self, other):
        """
        Construct an :class:`AndCompoundConstraint` from
        `self` and `other`

        Parameters
        ----------
        other : ConstraintExpression

        Returns
        -------
        AndCompoundConstraint
        """
        return AndCompoundConstraint(self, other)

    def __or__(self, other):
        """
        Construct an :class:`OrCompoundConstraint` from
        `self` and `other`

        Parameters
        ----------
        other : ConstraintExpression

        Returns
        -------
        OrCompoundConstraint
        """
        return OrCompoundConstraint(self, other)

    def __eq__(self, other):
        return self.expression == other.expression

    def __ne__(self, other):
        return self.expression != other.expression

    def get_symbols(self):
        return self.expression.get_symbols()


class SymbolNode(ExpressionBase):
    """
    A node representing a single symbol or variable with a scalar multiplication
    coefficient. When evaluated in the context of a :class:`Solution`, it's value
    will be substituted for the value hashed there.

    A SymbolNode is equal to its symbol string and hashes the same way.

    Attributes
    ----------
    coefficient : int
        Multiplicative coefficient to scale the value by
    symbol : str
        A name
    """
    @classmethod
    def parse(cls, string):
        coef = []
        i = 0
        while i < len(string) and (string[i].isdigit() or string[i] in '-+'):
            coef.append(string[i])
            i += 1
        coef_val = int(''.join(coef) or 1)
        residue_sym = string[i:]
        if residue_sym == "":
            residue_sym = None
        elif string[i] == "(" and string[-1] == ")":
            residue_sym = residue_sym[1:-1]
        return cls(residue_sym, coef_val)

    def __init__(self, symbol, coefficient=1):
        self.symbol = symbol
        self.coefficient = coefficient

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        if isinstance(other, basestring):
            return self.symbol == ensuretext(other)
        else:
            return self.symbol == other.symbol and self.coefficient == other.coefficient

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.symbol is not None:
            if self.coefficient != 1:
                return "{} * ({})".format(self.coefficient, self.symbol)
            else:
                return "({})".format(self.symbol)
        else:
            return "{}".format(self.coefficient)

    def get_symbols(self):
        return [self.symbol]


class ValueNode(ExpressionBase):
    """
    Represent a single numeric value in an ExpressionNode

    Attributes
    ----------
    value : int
        The value of this node
    """
    coefficient = 1

    @classmethod
    def parse(cls, string):
        try:
            value = int(string.replace(" ", ""))
        except ValueError:
            value = float(string.replace(" ", ""))
        return cls(value)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, basestring):
            return str(self.value) == ensuretext(other)
        else:
            return self.value == other.value

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self.value)


def typify(node):
    """Convert the input into the appropriate
    type of ExpressionBase-derived Node. Will
    apply pattern matching to determine if a
    passed string looks like a numerical constant
    or if the input is a numerical constant, return
    the appropriate node type.

    Parameters
    ----------
    node : object
        The object to be converted.

    Returns
    -------
    ExpressionBase
    """
    if isinstance(node, basestring):
        if re.match(r"^(\d+(.\d+)?)$", ensuretext(node).strip()):
            return ValueNode.parse(node)
        else:
            return SymbolNode.parse(node)
    elif isinstance(node, Number):
        return ValueNode(node)
    else:
        return node


def parse_expression(string):
    """Parses a str argument into a :class:`ExpressionNode` instance
    containing arbitrarily complex expressions.

    Parameters
    ----------
    string : str
        The text to be parsed

    Returns
    -------
    ExpressionNode

    Raises
    ------
    SyntaxError
        If there are unresolved parentheses leftover after resolving the parse
        stack
    """
    i = 0
    size = len(string)
    paren_level = 0
    current_symbol = ''
    expression_stack = []
    resolver_stack = []
    current_function = ''
    function_stack = []

    while i < size:
        c = string[i]
        if c == " ":
            if current_symbol != "":
                if current_symbol.endswith(","):
                    expression_stack.append(current_symbol[:-1])
                    expression_stack.append(",")
                else:
                    expression_stack.append(current_symbol)
            current_symbol = ""
        elif c == "(":
            paren_level += 1
            if current_symbol != "":
                if current_function != '':
                    function_stack.append(current_function)
                current_function = current_symbol
            current_symbol = ""
            resolver_stack.append(expression_stack)
            expression_stack = []
        elif c == ")":
            paren_level -= 1
            if current_symbol != "":
                if current_symbol.endswith(","):
                    current_symbol = current_symbol[:-1]
                expression_stack.append(current_symbol)
            current_symbol = ""
            term = collapse_expression_sequence(expression_stack)
            expression_stack = resolver_stack.pop()
            if current_function != "":
                fn = SymbolNode(current_function)
                term = FunctionCallNode(fn, Operator.get("call"), term)
                if function_stack:
                    current_function = function_stack.pop()
                else:
                    current_function = ''
            expression_stack.append(term)
        else:
            current_symbol += c
        i += 1

    if current_symbol != "":
        expression_stack.append(current_symbol)

    if len(resolver_stack) > 0:
        raise SyntaxError("Unpaired parenthesis")
    return collapse_expression_sequence(expression_stack)


def collapse_expression_sequence(expression_sequence):
    stack = []
    i = 0
    size = len(expression_sequence)
    if size == 1:
        return typify(expression_sequence[0])

    while i < size:
        next_term = expression_sequence[i]
        stack.append(next_term)
        if len(stack) == 3:
            node = ExpressionNode(*stack)
            if hasattr(node.left, 'op'):
                if node.left.op.precedence < node.op.precedence:
                    node = ExpressionNode(
                        left=node.left.left, op=node.left.op, right=ExpressionNode(
                            left=node.left.right, op=node.op, right=node.right))
            stack = [node]
        i += 1
    if len(stack) != 1:
        raise SyntaxError("Incomplete Expression: %s" %
                          ' '.join(expression_sequence))
    return stack[0]


class ExpressionNode(ExpressionBase):
    """
    A compound node representing two terms joined by a binary operator

    Attributes
    ----------
    left : ExpressionBase
        The term to the left of the binary operator
    op : Operator or str
        The binary operator being applied. If passed as a str
        the Operator type will be looked up using the registered
        operator map.
    right : ExpressionBase
        The term to the right of the binary operator
    """

    #: Expressions have an implicit constant multiplier of 1
    coefficient = 1

    #: Parses a string-like argument into an ExpressionNode instance
    parse = staticmethod(parse_expression)

    def __init__(self, left, op, right):
        self.left = typify(left)
        try:
            self.op = operator_map[op]
        except KeyError:
            if isinstance(op, Operator):
                self.op = op
            else:
                raise TypeError("Could not determine operator type from %r" % op)
        self.right = typify(right)

    def __repr__(self):
        return "{} {} {}".format(self.left, self.op, self.right)

    def evaluate(self, context):
        """
        Execute :attr:`op` on :attr:`left` and :attr:`right` in `context`

        Parameters
        ----------
        context : Solution

        Returns
        -------
        bool or int
        """
        if not isinstance(context, SymbolContext):
            context = SymbolContext(context)
        try:
            return self.op(self.left, self.right, context)
        except KeyError:
            if isinstance(self.left, ValueNode) and self.left.value == 0:
                return True
            elif isinstance(self.right, ValueNode) and self.right.value == 0:
                return True
            else:
                return False

    def get_symbols(self):
        return self.left.get_symbols() + self.right.get_symbols()


class FunctionCallNode(ExpressionNode):
    def __repr__(self):
        return "{}({})".format(re.sub(r"\)|\(", '', str(self.left)), self.right)


class SymbolSpace(object):
    """Represent an abstract collection of symbols. Provides a
    notion of definedness for Symbols, without associating an
    explicit value for them.

    Attributes
    ----------
    context : dict
        Internal storage of defined symbols
    """
    def __init__(self, context=None):
        if context is None:
            context = dict()
        self.context = self._format_iterable(context)

    @staticmethod
    def _format_iterable(iterable):
        space = dict()
        for key in iterable:
            if isinstance(key, SymbolNode):
                key = key.symbol
            else:
                key = str(key)
            space[key] = 1
        return space

    def _test_symbols_defined(self, symbols, partial=False):
        if not partial:
            for symbol in symbols:
                if not self.defined(symbol):
                    return False
            return True
        else:
            for symbol in symbols:
                if self.defined(symbol):
                    return True
            return False

    def defined(self, expr, partial=False):
        if isinstance(expr, (list, tuple, set, frozenset)):
            return self._test_symbols_defined(expr, partial)
        if not isinstance(expr, ExpressionBase):
            expr = str(expr)
        if isinstance(expr, basestring):
            expr = parse_expression(expr)
        if isinstance(expr, SymbolNode):
            return expr.symbol in self.context
        if isinstance(expr, ValueNode):
            return True
        if isinstance(expr, ExpressionNode):
            symbols = expr.get_symbols()
            return self._test_symbols_defined(symbols, partial)

    def partially_defined(self, expr):
        return self.defined(expr, True)

    def __contains__(self, expr):
        return self.defined(expr)

    def __repr__(self):
        return "{%s}" % ', '.join("%s" % c for c in self.context)

    def __iter__(self):
        return iter(self.context)

    def keys(self):
        return self.context.keys()


class SymbolContext(SymbolSpace):
    """
    Expands the notion of :class:`SymbolSpace` to bind values
    to Symbols

    Attributes
    ----------
    context : dict
        Mapping from Symbol to Value
    """

    def __init__(self, context=None):
        if context is None:
            context = dict()
        self.context = self._format_map(context)

    @staticmethod
    def _format_map(mapping):
        store = dict()
        for key, value in mapping.items():
            if isinstance(key, SymbolNode):
                key = key.symbol
            else:
                key = str(key)
            store[key] = value
        return store

    def __getitem__(self, node):
        """
        Summary

        Parameters
        ----------
        node : ExpressionBase

        Returns
        -------
        name : Number
        """
        if not isinstance(node, ExpressionBase):
            node = str(node)

        if isinstance(node, basestring):
            node = parse_expression(node)

        if isinstance(node, SymbolNode):
            if node.symbol is None:
                return 1
            else:
                try:
                    return self.context[node] * node.coefficient
                except KeyError:
                    return 0
        elif isinstance(node, ValueNode):
            return node.value
        elif isinstance(node, ExpressionNode):
            return node.evaluate(self)

    def __contains__(self, expr):
        if not isinstance(expr, ExpressionBase):
            expr = str(expr)
        if isinstance(expr, basestring):
            expr = parse_expression(expr)
        if isinstance(expr, SymbolNode):
            return expr.symbol in self.context
        elif isinstance(expr, ValueNode):
            return True
        elif isinstance(expr, ExpressionNode):
            return expr.evaluate(self) > 0

    def __repr__(self):
        return "SymbolContext(%r)" % self.context

    def items(self):
        return self.context.items()

    def keys(self):
        return self.context.keys()

    def values(self):
        return self.context.values()


class GlycanSymbolContext(SymbolContext):
    @staticmethod
    def _format_map(mapping):
        store = dict()
        for key, value in dict(mapping).items():
            if isinstance(key, SymbolNode):
                key = key.symbol
            elif isinstance(key, (str, _MR)):
                text_key = str(key)
                key = _FMR.from_iupac_lite(text_key)
                is_derivatized = composition_transform.has_derivatization(key)
                if is_derivatized:
                    key = str(glycan_composition.from_iupac_lite.strip_derivatization(text_key))
                else:
                    key = text_key
            store[key] = value
        return store

    def serialize(self):
        form = "{%s}" % '; '.join("{}:{}".format(str(k), v) for k, v in sorted(
            self.items(), key=lambda x: (
                _FMR.from_iupac_lite(str(x[0])).mass(), str(x[0]))) if v != 0)
        return form

    @staticmethod
    def normalize_key(composition):
        key_obj = composition.clone()
        key_obj = composition_transform.strip_derivatization(key_obj)
        return str(key_obj)


Solution = SymbolContext

operator_map = {}


def register_operator(cls):
    """
    Summary

    Returns
    -------
    name : TYPE
        Description
    """
    inst = cls()
    operator_map[cls.symbol] = inst
    for alias in getattr(cls, "aliases", []):
        operator_map[alias] = inst
    return cls


@register_operator
class Operator(object):
    """
    A base class for defining binary arithmetic
    operators over expressions. Subclasses should
    provide a :meth:`__call__` method to handle
    a left and right :class:`ExpressionBase` objects
    and a :class:`SymbolContext` instance in which
    to resolve them.

    Attributes
    ----------
    precedence : int
        The operator's precedence, which influences
        the order in which they are applied
    symbol : str
        The textual encoding of the operator
    aliases : list
        A list of alternative symbols for the operator
    """
    symbol = "NoOp"
    precedence = 0

    def __init__(self, symbol=None):
        """
        Summary

        Parameters
        ----------
        symbol : TYPE, optional
            Description
        """
        if symbol is not None:
            self.symbol = symbol

    def __call__(self, left, right, context):
        """
        Summary

        Parameters
        ----------
        left : TYPE
            Description
        right : TYPE
            Description
        context : TYPE
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        return NotImplemented

    def __repr__(self):
        return self.symbol

    def __eq__(self, other):
        """
        Summary

        Parameters
        ----------
        other : TYPE
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        try:
            return self.symbol == other.symbol
        except AttributeError:
            return self.symbol == other

    def __hash__(self):
        return hash(self.symbol)

    operator_map = operator_map

    @classmethod
    def get(cls, symbol):
        return cls.operator_map[symbol]


@register_operator
class LessThan(Operator):
    symbol = "<"

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val < right_val


@register_operator
class LessThanOrEqual(Operator):
    symbol = "<="

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val <= right_val


@register_operator
class GreaterThan(Operator):
    symbol = ">"

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val > right_val


@register_operator
class GreaterThanOrEqual(Operator):
    symbol = ">="

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val >= right_val


@register_operator
class Equal(Operator):
    symbol = "="
    precedence = 10
    aliases = ["=="]

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val == right_val


@register_operator
class Unequal(Operator):
    symbol = "!="
    precedence = 10

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val != right_val


@register_operator
class Subtraction(Operator):
    """
    Summary

    Attributes
    ----------
    precedence : int
        Description
    symbol : str
        Description
    """
    symbol = "-"
    precedence = 1

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val - right_val


@register_operator
class Addition(Operator):
    symbol = "+"
    precedence = 1

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val + right_val


@register_operator
class Multplication(Operator):
    symbol = '*'
    precedence = 2

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val * right_val


@register_operator
class Division(Operator):
    symbol = '/'
    precedence = 2

    def __call__(self, left, right, context):
        left_val = context[left] * left.coefficient
        right_val = context[right] * right.coefficient
        return left_val / right_val


@register_operator
class Or(Operator):
    symbol = 'or'
    aliases = ['|', "||"]

    def __call__(self, left, right, context):
        return context[left] or context[right]


@register_operator
class And(Operator):
    symbol = "and"
    aliases = ["&", "&&"]

    def __call__(self, left, right, context):
        return context[left] and context[right]


@register_operator
class Append(Operator):
    symbol = ','

    def __call__(self, left, right, context):
        lval = context[left]
        rval = context[right]
        if not isinstance(lval, tuple):
            lval = (lval,)
        if not isinstance(rval, tuple):
            rval = (rval,)
        return lval + rval


@register_operator
class Call(Operator):
    symbol = "call"

    def __call__(self, left, right, context):
        try:
            fn = function_table[left]
            return fn(right, context)
        except KeyError:
            raise UnknownFunctionError(left)


class UnknownFunctionError(KeyError):
    pass


def symbolic_sum(terms, context):
    value = 0
    try:
        for term in context[terms]:
            value += context[term]
        return value
    except TypeError:
        return context[terms]


def symbolic_abs(terms, context):
    return abs(context[terms])


function_table = {
    "sum": symbolic_sum,
    "abs": symbolic_abs
}


class OrCompoundConstraint(ConstraintExpression):
    """
    A combination of ConstraintExpression instances which
    is satisfied if either of its components are satisfied.

    Attributes
    ----------
    left : ConstraintExpression
    right : ConstraintExpression
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __call__(self, context):
        return self.left(context) or self.right(context)

    def __repr__(self):
        return "(({}) or ({}))".format(self.left, self.right)

    def get_symbols(self):
        return self.left.get_symbols() + self.right.get_symbols()


class AndCompoundConstraint(ConstraintExpression):
    """
    A combination of ConstraintExpression instances which
    is only satisfied if both of its components are satisfied.

    Attributes
    ----------
    left : ConstraintExpression
    right : ConstraintExpression
    """

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def __call__(self, context):
        return self.left(context) and self.right(context)

    def __repr__(self):
        return "(({}) and ({}))".format(self.left, self.right)

    def get_symbols(self):
        return self.left.get_symbols() + self.right.get_symbols()
