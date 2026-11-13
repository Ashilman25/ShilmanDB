from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "ON", "AND", "OR", "NOT", "GROUP", "BY", "ORDER",
    "ASC", "DESC", "LIMIT", "AS", "HAVING", "BETWEEN", "IN", "LIKE", "COUNT", "SUM", "AVG", "MIN", "MAX",
    "DATE", "CASE", "WHEN", "THEN", "ELSE", "END",
}
AGGREGATES = ("COUNT", "SUM", "AVG", "MIN", "MAX")
COMPARISONS = ("=", "<>", "<", ">", "<=", ">=")
ARITHMETIC = ("+", "-", "*", "/")
INT_TYPES = ("INTEGER", "BIGINT")
TEXT_TYPES = ("VARCHAR", "DATE")
RANK = {"INTEGER": 1, "BIGINT": 2, "DECIMAL": 3}
INT32_MAX, INT64_MAX = 2**31 - 1, 2**63 - 1

_DIGITS = frozenset("0123456789")
_WORD_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_WORD = _WORD_START | _DIGITS
_SINGLE = frozenset("=<>+-*/,.();")


@dataclass
class Agg:
    func: str
    columns: set[str]
    star: bool


@dataclass
class SqlFacts:
    ok: bool
    error: str | None = None
    tables: list[str] = field(default_factory=list)
    columns: set[str] = field(default_factory=set)
    column_table: dict[str, str] = field(default_factory=dict)
    aggregates: list[Agg] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    equalities: list[tuple[str, str]] = field(default_factory=list)
    has_left_join: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    constant_conjuncts: list[str] = field(default_factory=list)
    text_arithmetic: bool = False
    int_divisions: list[str] = field(default_factory=list)
    select_columns: list[str | None] = field(default_factory=list)
    count_star_items: list[int] = field(default_factory=list)


class SqlError(Exception):
    pass


class Token(NamedTuple):
    kind: str   # a keyword, an operator, IDENT, INT, FLOAT, STRING, INVALID or EOF
    value: str
    pos: int


@dataclass(eq=False)
class Col:
    table: str | None
    name: str


@dataclass(eq=False)
class Lit:
    kind: str   # INT, FLOAT, STRING or DATE
    text: str


@dataclass(eq=False)
class Star:
    pass


@dataclass(eq=False)
class Bin:
    op: str
    left: Node
    right: Node


@dataclass(eq=False)
class Unary:
    op: str     # NOT or -
    operand: Node


@dataclass(eq=False)
class AggCall:
    func: str
    arg: Node | None  # None for COUNT(*)


@dataclass(eq=False)
class Case:
    whens: list[tuple[Node, Node]]
    otherwise: Node | None


Node = Col | Lit | Star | Bin | Unary | AggCall | Case


@dataclass
class Join:
    table: str
    left: bool
    on: Node


@dataclass
class Select:
    items: list[tuple[Node, str | None]]
    tables: list[str]
    joins: list[Join]
    where: Node | None
    group_by: list[Node]
    having: Node | None
    order_by: list[Node]
    limit: int | None


# Tokens one at a time, as the engine's lexer reads them; then EOF forever.
def tokenize(sql: str) -> Iterator[Token]:
    i, n = 0, len(sql)
    while True:
        while i < n:
            if sql[i] in " \t\r\n":
                i += 1
            elif sql.startswith("--", i):
                end = sql.find("\n", i)
                i = n if end < 0 else end
            else:
                break
            
        if i >= n:
            break

        start, c = i, sql[i]
        if c in _DIGITS:
            while i < n and sql[i] in _DIGITS:
                i += 1
            kind = "INT"
            if i + 1 < n and sql[i] == "." and sql[i + 1] in _DIGITS:
                i += 1
                while i < n and sql[i] in _DIGITS:
                    i += 1
                kind = "FLOAT"
            yield Token(kind, sql[start:i], start)
            
        elif c == "'":
            text, i = _read_string(sql, i)
            yield Token("STRING", text, start)
            
        elif c in _WORD_START:
            while i < n and sql[i] in _WORD:
                i += 1
            word = sql[start:i]
            upper = word.upper()
            yield Token(upper, upper, start) if upper in KEYWORDS else Token("IDENT", word, start)
            
        elif sql.startswith(("!=", "<>"), i):
            i += 2
            yield Token("<>", sql[start:i], start)
            
        elif sql.startswith(("<=", ">="), i):
            i += 2
            yield Token(sql[start:i], sql[start:i], start)
            
        else:
            i += 1
            yield Token(c if c in _SINGLE else "INVALID", c, start)

    while True:
        yield Token("EOF", "", n)


# A quoted string starting at i, with '' as a quote.
# Returns the text and the index after it.
def _read_string(sql: str, i: int) -> tuple[str, int]:
    start, parts = i, []
    i += 1
    while i < len(sql):
        if sql[i] != "'":
            parts.append(sql[i])
        elif sql.startswith("''", i):
            parts.append("'")
            i += 1
        else:
            return "".join(parts), i + 1
        i += 1
    raise SqlError(f"unterminated string literal at offset {start}")


# An all-digit literal no bigger than limit, as the engine's stoi / stoll check it.
def _in_range(digits: str, limit: int) -> bool:
    digits = digits.lstrip("0")
    return len(digits) <= len(str(limit)) and int(digits or "0") <= limit


# Recursive descent over the engine's grammar, one method per parser.cpp function.
class Parser:
    def __init__(self, sql: str):
        self._tokens = tokenize(sql)
        self._peeked: Token | None = None
        self.tok = Token("EOF", "", 0)

    def advance(self) -> None:
        if self._peeked is not None:
            self.tok, self._peeked = self._peeked, None
        else:
            self.tok = next(self._tokens)

    def peek(self) -> Token:
        if self._peeked is None:
            self._peeked = next(self._tokens)
        return self._peeked

    def fail(self, message: str) -> SqlError:
        return SqlError(f"{message} at offset {self.tok.pos}")

    # The current token as an error message shows it.
    def shown(self) -> str:
        return "the end" if self.tok.kind == "EOF" else repr(self.tok.value)

    def expect(self, kind: str) -> str:
        if self.tok.kind != kind:
            raise self.fail(f"expected {kind} but got {self.shown()}")
        value = self.tok.value
        self.advance()
        return value

    def match(self, kind: str) -> bool:
        if self.tok.kind == kind:
            self.advance()
            return True
        return False

    # One statement; text after a ';' is never read, as in the engine.
    def statement(self) -> Select:
        self.advance()
        stmt = self.select()
        if self.tok.kind != "EOF":
            self.expect(";")
        return stmt

    def select(self) -> Select:
        self.expect("SELECT")
        items = self.select_list()
        self.expect("FROM")
        tables = self.from_list()
        joins = self.joins()

        where = self.expression() if self.match("WHERE") else None
        group_by = self.expr_list("GROUP") if self.tok.kind == "GROUP" else []
        having = self.expression() if self.match("HAVING") else None
        order_by = self.expr_list("ORDER") if self.tok.kind == "ORDER" else []
        limit = self.limit() if self.match("LIMIT") else None
        return Select(items, tables + [j.table for j in joins], joins, where, group_by, having, order_by, limit)

    # After AS any token is the alias; without AS only an identifier is.
    def alias(self) -> str | None:
        if self.match("AS") or self.tok.kind == "IDENT":
            name = self.tok.value
            self.advance()
            return name
        return None

    def select_list(self) -> list[tuple[Node, str | None]]:
        items = []
        while True:
            if self.match("*"):
                expr = Star()
            else:
                expr = self.expression()
            items.append((expr, self.alias()))
            if not self.match(","):
                return items

    def from_list(self) -> list[str]:
        tables = []
        while True:
            tables.append(self.expect("IDENT"))
            self.alias()
            if not self.match(","):
                return tables

    def joins(self) -> list[Join]:
        joins = []
        while self.tok.kind in ("JOIN", "INNER", "LEFT"):
            left = self.tok.kind == "LEFT"
            if self.tok.kind != "JOIN":
                self.advance()
            self.expect("JOIN")
            table = self.expect("IDENT")
            self.alias()
            self.expect("ON")
            joins.append(Join(table, left, self.expression()))
        return joins

    # GROUP BY or ORDER BY items; ORDER BY's ASC / DESC are skipped.
    def expr_list(self, clause: str) -> list[Node]:
        self.expect(clause)
        self.expect("BY")
        exprs = []
        while True:
            exprs.append(self.expression())
            if clause == "ORDER" and self.tok.kind in ("ASC", "DESC"):
                self.advance()
            if not self.match(","):
                return exprs

    def limit(self) -> int:
        if self.tok.kind != "INT":
            raise self.fail("expected integer after LIMIT")
        if not _in_range(self.tok.value, INT64_MAX):
            raise self.fail(f"LIMIT value out of range: {self.tok.value}")
        value = int(self.tok.value)
        self.advance()
        return value

    def expression(self) -> Node:
        left = self.and_expr()
        while self.match("OR"):
            left = Bin("OR", left, self.and_expr())
        return left

    def and_expr(self) -> Node:
        left = self.comparison()
        while self.match("AND"):
            left = Bin("AND", left, self.comparison())
        return left

    # BETWEEN becomes (x >= a) AND (x <= b).
    # NOT BETWEEN and NOT LIKE wrap theirs in NOT.
    def comparison(self) -> Node:
        left = self.add_sub()
        negated = self.tok.kind == "NOT" and self.peek().kind in ("BETWEEN", "IN", "LIKE")
        if negated:
            self.advance()
            
        if self.match("BETWEEN"):
            low = self.add_sub()
            self.expect("AND")
            node = Bin("AND", Bin(">=", left, low), Bin("<=", left, self.add_sub()))
            
        elif self.match("IN"):
            return self.in_list(left, negated)
        
        elif self.match("LIKE"):
            node = Bin("LIKE", left, self.add_sub())
            
        else:
            while self.tok.kind in COMPARISONS:
                op = self.tok.kind
                self.advance()
                left = Bin(op, left, self.add_sub())
            return left
        return Unary("NOT", node) if negated else node

    # x IN (a, b) is (x = a) OR (x = b); NOT IN is (x <> a) AND (x <> b).
    def in_list(self, lhs: Node, negated: bool) -> Node:
        self.expect("(")
        values = [self.add_sub()]
        while self.match(","):
            values.append(self.add_sub())
        self.expect(")")

        leaf, glue = ("<>", "AND") if negated else ("=", "OR")
        result = Bin(leaf, lhs, values[-1])
        for value in reversed(values[:-1]):
            result = Bin(glue, Bin(leaf, lhs, value), result)
        return result

    def add_sub(self) -> Node:
        left = self.mul_div()
        while self.tok.kind in ("+", "-"):
            op = self.tok.kind
            self.advance()
            left = Bin(op, left, self.mul_div())
        return left

    def mul_div(self) -> Node:
        left = self.unary()
        while self.tok.kind in ("*", "/"):
            op = self.tok.kind
            self.advance()
            left = Bin(op, left, self.unary())
        return left

    def unary(self) -> Node:
        if self.tok.kind in ("NOT", "-"):
            op = self.tok.kind
            self.advance()
            return Unary(op, self.unary())
        return self.primary()

    # Literals, parentheses, CASE, aggregates, columns and a bare *.
    def primary(self) -> Node:
        tok = self.tok
        if tok.kind == "INT" and not _in_range(tok.value, INT32_MAX):
            raise self.fail(f"integer literal out of range: {tok.value}")
        if tok.kind == "FLOAT" and math.isinf(float(tok.value)):
            raise self.fail(f"float literal out of range: {tok.value}")

        if tok.kind in ("INT", "FLOAT", "STRING"):
            self.advance()
            return Lit(tok.kind, tok.value)
        if self.match("DATE"):
            if self.tok.kind != "STRING":
                raise self.fail("expected date string after DATE")
            return Lit("DATE", self.expect("STRING"))
        if self.match("("):
            expr = self.expression()
            self.expect(")")
            return expr
        if self.match("CASE"):
            return self.case()
        if tok.kind in AGGREGATES:
            self.advance()
            self.expect("(")
            arg = None if tok.kind == "COUNT" and self.match("*") else self.expression()
            self.expect(")")
            return AggCall(tok.kind, arg)
        if self.match("IDENT"):
            if self.match("."):
                return Col(tok.value, self.expect("IDENT"))
            return Col(None, tok.value)
        if self.match("*"):
            return Star()
        raise self.fail(f"unexpected token {self.shown()}")

    def case(self) -> Case:
        whens = []
        while self.match("WHEN"):
            cond = self.expression()
            self.expect("THEN")
            whens.append((cond, self.expression()))
        if not whens:
            raise self.fail("CASE needs at least one WHEN")
        otherwise = self.expression() if self.match("ELSE") else None
        self.expect("END")
        return Case(whens, otherwise)


# A node's direct sub-expressions, in written order.
def _children(node: Node) -> list[Node]:
    match node:
        case Bin():
            return [node.left, node.right]
        case Unary():
            return [node.operand]
        case AggCall() if node.arg is not None:
            return [node.arg]
        case Case():
            return [part for pair in node.whens for part in pair] + ([node.otherwise] if node.otherwise else [])
    return []


# Every node under this one, parents first, left to right; a shared node once.
def _nodes(root: Node | None) -> list[Node]:
    out, seen, stack = [], set(), [root] if root is not None else []
    while stack:
        node = stack.pop()
        if id(node) not in seen:
            seen.add(id(node))
            out.append(node)
            stack += reversed(_children(node))
    return out


# Top-level AND terms, split as the planner's SplitConjuncts does.
def _conjuncts(root: Node) -> list[Node]:
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        if isinstance(node, Bin) and node.op == "AND":
            stack += [node.right, node.left]
        else:
            out.append(node)
    return out


# The engine's CommonType; None stands for INVALID or unknown.
def _common(a: str | None, b: str | None) -> str | None:
    if a == b:
        return a
    if {a, b} == {"DATE", "VARCHAR"}:
        return "DATE"
    if a in RANK and b in RANK:
        return a if RANK[a] > RANK[b] else b
    return None


# One node's type from its children's, as the planner's ResolveExprType works it out.
def _type(node: Node, names: dict[str, str], types: dict[int, str | None]) -> str | None:
    match node:
        case Col():
            return names.get(node.name)
        case Lit():
            return {"INT": "INTEGER", "FLOAT": "DECIMAL", "STRING": "VARCHAR", "DATE": "DATE"}[node.kind]
        case AggCall(func="COUNT"):
            return "INTEGER"
        case AggCall(func="SUM" | "AVG"):
            return "DECIMAL"
        case AggCall():
            return types[id(node.arg)]
        case Case():
            first = types[id(node.whens[0][1])]
            resolved = first
            for _, result in node.whens[1:]:
                resolved = _common(resolved, types[id(result)])
                if resolved is None:
                    return first
            merged = _common(resolved, types[id(node.otherwise)]) if node.otherwise else None
            return merged or resolved
        
        case Bin(op=op) if op in ARITHMETIC:
            left = types[id(node.left)]
            return _common(left, types[id(node.right)]) or left
        case Bin() | Unary(op="NOT"):
            return "INTEGER"
        case Unary():
            return types[id(node.operand)]
    return None


# The type of every node under root, keyed by id; children before parents.
def _types(root: Node, names: dict[str, str]) -> dict[int, str | None]:
    types: dict[int, str | None] = {}
    for node in reversed(_nodes(root)):
        types[id(node)] = _type(node, names, types)
    return types


# Readable text of an expression; nested operators get parentheses.
def _render(node: Node) -> str:
    def inner(child: Node) -> str:
        return f"({_render(child)})" if isinstance(child, (Bin, Unary)) else _render(child)

    match node:
        case Col():
            return f"{node.table}.{node.name}" if node.table else node.name
        case Lit(kind="STRING"):
            return "'" + node.text.replace("'", "''") + "'"
        case Lit(kind="DATE"):
            return "DATE '" + node.text.replace("'", "''") + "'"
        case Lit():
            return node.text
        case Star():
            return "*"
        case AggCall():
            return f"{node.func}({'*' if node.arg is None else _render(node.arg)})"
        case Unary():
            return f"NOT {inner(node.operand)}" if node.op == "NOT" else f"-{inner(node.operand)}"
        case Bin():
            return f"{inner(node.left)} {node.op} {inner(node.right)}"
        case Case():
            whens = " ".join(f"WHEN {_render(c)} THEN {_render(r)}" for c, r in node.whens)
            otherwise = f" ELSE {_render(node.otherwise)}" if node.otherwise else ""
            return f"CASE {whens}{otherwise} END"
    raise TypeError(f"not a node: {node!r}")


# A COUNT(*) call, starred the same way Agg.star is.
def _is_count_star(node: Node) -> bool:
    return isinstance(node, AggCall) and node.func == "COUNT" and (node.arg is None or isinstance(node.arg, Star))


# Bare column names under root, leaving out the given names (aliases).
def _column_names(root: Node | None, skip: frozenset[str] = frozenset()) -> list[str]:
    return [n.name for n in _nodes(root) if isinstance(n, Col) and n.name not in skip]


# Walk a parsed statement once per clause and gather every fact.
def _facts(stmt: Select, manifest: dict) -> SqlFacts:
    types = {c["name"]: c["type"] for t in manifest["tables"] for c in t["columns"]}
    owner = {c["name"]: t["name"] for t in manifest["tables"] for c in t["columns"]}

    aliases: dict[str, str | None] = {}
    for expr, alias in stmt.items:
        if alias is not None:
            aliases.setdefault(alias, _types(expr, types)[id(expr)])

    f = SqlFacts(ok=True, tables=list(stmt.tables), has_left_join=any(j.left for j in stmt.joins), has_order_by=bool(stmt.order_by), has_limit=stmt.limit is not None)

    # ORDER BY sees the select list's aliases, as the engine's sort does.
    body = [e for e, _ in stmt.items] + [j.on for j in stmt.joins] + [stmt.where] + stmt.group_by + [stmt.having]
    parts = [(e, types, frozenset()) for e in body if e is not None]
    parts += [(e, {**types, **aliases}, frozenset(aliases)) for e in stmt.order_by]

    for expr, names, skip in parts:
        node_types = _types(expr, names)
        for node in _nodes(expr):
            match node:
                case Col() if node.name not in skip:
                    f.columns.add(node.name)
                case AggCall():
                    star = node.arg is None or isinstance(node.arg, Star)
                    f.aggregates.append(Agg(node.func, set(_column_names(node.arg, skip)), star))
                case Bin(op=op) if op in ARITHMETIC:
                    sides = (node_types[id(node.left)], node_types[id(node.right)])
                    f.text_arithmetic |= any(s in TEXT_TYPES for s in sides)
                    if op == "/" and all(s in INT_TYPES for s in sides):
                        f.int_divisions.append(_render(node))

    f.group_by = list(dict.fromkeys(c for expr in stmt.group_by for c in _column_names(expr)))

    for expr in [j.on for j in stmt.joins] + ([stmt.where] if stmt.where else []):
        f.equalities += [(c.left.name, c.right.name) for c in _conjuncts(expr) if isinstance(c, Bin) and c.op == "=" and isinstance(c.left, Col) and isinstance(c.right, Col)]

    if stmt.where:
        f.constant_conjuncts = [_render(c) for c in _conjuncts(stmt.where) if not _column_names(c)]

    f.select_columns = [e.name if isinstance(e, Col) else None for e, _ in stmt.items]
    f.count_star_items = [i for i, (e, _) in enumerate(stmt.items) if any(_is_count_star(n) for n in _nodes(e))]
    f.column_table = {c: owner[c] for c in f.columns if c in owner}
    return f


# Facts about one statement; anything that doesn't parse gives ok=False and the error.
def facts(sql: str, manifest: dict) -> SqlFacts:
    if not isinstance(sql, str):
        return SqlFacts(ok=False, error=f"expected SQL text, got {type(sql).__name__}")
    try:
        return _facts(Parser(sql).statement(), manifest)
    except SqlError as e:
        return SqlFacts(ok=False, error=str(e))
    except RecursionError:
        return SqlFacts(ok=False, error="statement is nested too deeply")
