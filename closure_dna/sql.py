"""SQL execution layer for Closure DNA.

Standard SQL statements are parsed with sqlglot, then mapped to the
existing Closure DNA database surface. Custom DNA statements remain
explicit product syntax and route into the same engine.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .database import Database
from .database import _evaluate_expression as db_evaluate_expression
from .database import _match_join_where, _match_where, _sort_records
from .query import coerce_row

try:
    from sqlglot import exp, parse, parse_one
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".vendor"))
    from sqlglot import exp, parse, parse_one


@dataclass
class SQLResult:
    rows: list[dict] = field(default_factory=list)
    count: int = 0
    message: str = ""
    scalar: object | None = None
    statements: list[dict] = field(default_factory=list)


_AUDIT_RE = re.compile(r"^AUDIT\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\s*$", re.IGNORECASE)
_COMPACT_RE = re.compile(r"^COMPACT\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\s*$", re.IGNORECASE)
_INSPECT_RE = re.compile(
    r"^INSPECT\s+ROW\s+(?P<row>\d+)\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\s*$",
    re.IGNORECASE,
)


@dataclass
class _JoinSpec:
    table: str
    alias: str
    left_expr: str
    right_col: str
    outer: str | None = None


@dataclass
class _SelectSource:
    kind: str
    table: str
    alias: str
    joins: list[_JoinSpec] = field(default_factory=list)


@dataclass
class _Projection:
    alias: str
    expr_sql: str
    aggregate: tuple[str, str] | None = None
    special: str | None = None


def execute(db: Database, sql: str) -> SQLResult:
    statements = _split_sql_script(sql)
    if not statements:
        raise ValueError("empty SQL")
    if len(statements) > 1:
        results = [_execute_single(db, statement) for statement in statements]
        final = results[-1]
        final.statements = [_result_summary(result) for result in results]
        if not final.message:
            final.message = f"Executed {len(results)} statements"
        return final
    return _execute_single(db, statements[0])


def _execute_single(db: Database, sql: str) -> SQLResult:
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise ValueError("empty SQL")

    upper = text.upper()
    if upper.startswith("AUDIT "):
        return _execute_audit(db, text)
    if upper.startswith("COMPACT "):
        return _execute_compact(db, text)
    if upper.startswith("INSPECT ROW "):
        return _execute_inspect(db, text)

    search_values = None
    if upper.startswith("SELECT "):
        text, search_values = _extract_resonate_clause(text)

    try:
        tree = parse_one(text, read="sqlite")
    except Exception as e:  # pragma: no cover - exercised in failure mode
        raise ValueError(f"invalid SQL: {e}") from e

    if isinstance(tree, exp.Transaction):
        return _execute_begin(db)
    if isinstance(tree, exp.Commit):
        return _execute_commit(db)
    if isinstance(tree, exp.Rollback):
        return _execute_rollback(db)

    active_txn = getattr(db, "_sql_txn", None)
    if active_txn is not None:
        db = active_txn._db

    if isinstance(tree, exp.Select):
        return _execute_select(db, tree, search_values)
    if isinstance(tree, exp.Union):
        return _execute_union(db, tree)
    if isinstance(tree, exp.Insert):
        return _execute_insert(db, tree)
    if isinstance(tree, exp.Update):
        return _execute_update(db, tree)
    if isinstance(tree, exp.Delete):
        return _execute_delete(db, tree)
    if isinstance(tree, exp.Alter):
        return _execute_alter(db, tree)
    if isinstance(tree, exp.Create):
        return _execute_create(db, tree)
    if isinstance(tree, exp.Drop):
        return _execute_drop(db, tree)
    raise ValueError(f"unsupported SQL statement: {type(tree).__name__}")


def _result_summary(result: SQLResult) -> dict:
    return {
        "count": result.count,
        "message": result.message,
        "scalar": result.scalar,
        "rows": result.rows,
    }


def _execute_select(db: Database, tree: exp.Select, search_values: list[object] | None) -> SQLResult:
    source = _parse_source(tree)
    where = _ast_to_where(tree.args.get("where").this, db, source) if tree.args.get("where") else None
    having = _ast_to_where(tree.args.get("having").this, db, source) if tree.args.get("having") else None
    projections = [_parse_projection(expr) for expr in tree.expressions]
    order_by, descending = _parse_order(tree.args.get("order"))
    group_by = _parse_group(tree.args.get("group"))
    limit = _parse_int_arg(tree.args.get("limit"))
    offset = _parse_int_arg(tree.args.get("offset"))
    distinct = tree.args.get("distinct") is not None

    special = next((p for p in projections if p.special is not None), None)
    if special is not None:
        if len(projections) != 1 or source.kind != "table":
            raise ValueError("special geometric SELECT functions require a single table source")
        value = _compute_special_select(db, source.table, special.special)
        return SQLResult(rows=[{special.alias: value}], count=1, message=str(value), scalar=value)

    if search_values is not None:
        rows = _execute_search_select(db, source, search_values, limit)
        limit = None
    else:
        rows = _materialize_source_rows(db, source)
    rows = _apply_where(rows, where, source, db)

    if group_by:
        rows = _group_rows(rows, projections, group_by, source)
        if having is not None:
            rows = _apply_group_having(rows, having)
        if order_by is not None:
            rows = _sort_projected_records(rows, _normalize_expr(order_by, source), descending)
        rows = _strip_group_auxiliary(rows, projections)
    elif any(p.aggregate is not None for p in projections):
        if len(projections) != 1 or projections[0].aggregate is None:
            raise ValueError("mixed aggregate and non-aggregate SELECT without GROUP BY is not supported")
        func, target = projections[0].aggregate
        value = _compute_aggregate(rows, _normalize_expr(target, source), func)
        rows = [{projections[0].alias: value}]
    else:
        if order_by is not None:
            rows = _sort_records(rows, _normalize_expr(order_by, source), descending)
        if not _is_star_projection(projections):
            rows = [_project_row(row, projections, source) for row in rows]
        if distinct:
            rows = _distinct_rows(rows)

    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    scalar = rows[0][next(iter(rows[0]))] if len(rows) == 1 and projections and projections[0].aggregate else None
    return SQLResult(rows=rows, count=len(rows), message="" if scalar is None else str(scalar), scalar=scalar)


def _execute_insert(db: Database, tree: exp.Insert) -> SQLResult:
    table_name = _table_name(tree.this)
    values_node = tree.args.get("expression")
    if not isinstance(values_node, exp.Values):
        raise ValueError("INSERT source must be VALUES")
    rows = []
    table = db.table(table_name)
    for tup in values_node.expressions:
        if not isinstance(tup, exp.Tuple):
            raise ValueError("INSERT VALUES must contain tuples")
        rows.append(coerce_row(table.schema(), [_ast_to_literal(e, db, None) for e in tup.expressions]))
    if not rows:
        return SQLResult(count=0, message="Inserted 0 row(s)", scalar=0)
    if len(rows) == 1:
        row = table.insert(rows[0])
        return SQLResult(count=1, message=f"Inserted row {row}", scalar=row)
    inserted = table.insert_many(rows)
    return SQLResult(count=inserted, message=f"Inserted {inserted} row(s)", scalar=inserted)


def _execute_update(db: Database, tree: exp.Update) -> SQLResult:
    table_name = _table_name(tree.this)
    set_values = {}
    for expr_item in tree.expressions:
        if not isinstance(expr_item, exp.EQ) or not isinstance(expr_item.this, exp.Column):
            raise ValueError("unsupported UPDATE SET expression")
        set_values[_column_sql(expr_item.this)] = _ast_to_literal(expr_item.expression, db, None)
    where = _ast_to_where(tree.args.get("where").this, db, None) if tree.args.get("where") else None
    if where is None:
        raise ValueError("UPDATE without WHERE is not supported")
    updated = db.update_where(table_name, set_values, where)
    return SQLResult(count=updated, message=f"{updated} row(s) updated", scalar=updated)


def _execute_delete(db: Database, tree: exp.Delete) -> SQLResult:
    table_name = _table_name(tree.this)
    where = _ast_to_where(tree.args.get("where").this, db, None) if tree.args.get("where") else None
    if where is None:
        raise ValueError("DELETE without WHERE is not supported")
    deleted = db.delete_where(table_name, where)
    return SQLResult(count=deleted, message=f"{deleted} row(s) deleted", scalar=deleted)


def _execute_alter(db: Database, tree: exp.Alter) -> SQLResult:
    table_name = _table_name(tree.this)
    actions = tree.args.get("actions") or []
    if len(actions) != 1 or not isinstance(actions[0], exp.ColumnDef):
        raise ValueError("only ALTER TABLE ADD COLUMN is supported")
    action = actions[0]
    column = {
        "name": action.this.name,
        "type": _map_sql_type(action.kind),
        "indexed": False,
        "not_null": False,
        "unique": False,
    }
    default = None
    for constraint in action.args.get("constraints") or []:
        kind = constraint.args.get("kind")
        if isinstance(kind, exp.NotNullColumnConstraint):
            column["not_null"] = True
        elif isinstance(kind, exp.DefaultColumnConstraint):
            default = _ast_to_literal(kind.this, db, None)
        elif isinstance(kind, exp.UniqueColumnConstraint):
            column["unique"] = True
        elif isinstance(kind, exp.PrimaryKeyColumnConstraint):
            column["not_null"] = True
            column["unique"] = True
        else:
            raise ValueError(f"unsupported column constraint: {type(kind).__name__}")
    db.add_column(table_name, column, default=default)
    return SQLResult(count=0, message=f"Added column {column['name']} to {table_name}")


def _execute_create(db: Database, tree: exp.Create) -> SQLResult:
    if tree.args.get("kind") == "INDEX":
        return _execute_create_index(db, tree)
    if tree.args.get("kind") != "TABLE":
        raise ValueError("only CREATE TABLE is supported")
    if not isinstance(tree.this, exp.Schema) or not isinstance(tree.this.this, exp.Table):
        raise ValueError("CREATE TABLE requires an inline schema")
    table_name = tree.this.this.name
    if db.has_table(table_name):
        if tree.args.get("exists"):
            return SQLResult(count=0, message=f"Table {table_name} already exists")
        raise FileExistsError(f"Table already exists: {table_name}")
    schema = [_column_def_to_schema(entry) for entry in tree.this.expressions]
    db.create_table(table_name, schema)
    return SQLResult(count=0, message=f"Created table {table_name}")


def _execute_drop(db: Database, tree: exp.Drop) -> SQLResult:
    if tree.args.get("kind") == "INDEX":
        return _execute_drop_index(db, tree)
    if tree.args.get("kind") != "TABLE":
        raise ValueError("only DROP TABLE is supported")
    table_name = _table_name(tree.this)
    if not db.has_table(table_name):
        if tree.args.get("exists"):
            return SQLResult(count=0, message=f"Table {table_name} does not exist")
        raise FileNotFoundError(f"Table not found: {table_name}")
    db.drop_table(table_name)
    return SQLResult(count=0, message=f"Dropped table {table_name}")


def _execute_audit(db: Database, sql: str) -> SQLResult:
    match = _AUDIT_RE.match(sql)
    if not match:
        raise ValueError("unsupported AUDIT syntax")
    value = db.audit(match.group("table"))
    return SQLResult(rows=[value], count=1, message="audit", scalar=value)


def _execute_compact(db: Database, sql: str) -> SQLResult:
    match = _COMPACT_RE.match(sql)
    if not match:
        raise ValueError("unsupported COMPACT syntax")
    value = db.compact(match.group("table"))
    return SQLResult(rows=[value], count=1, message="compact", scalar=value)


def _execute_inspect(db: Database, sql: str) -> SQLResult:
    match = _INSPECT_RE.match(sql)
    if not match:
        raise ValueError("unsupported INSPECT syntax")
    table = db.table(match.group("table"))
    drift, base, phase = table.inspect_row(int(match.group("row")))
    value = {"drift": drift, "base": [float(base[0]), float(base[1]), float(base[2])], "phase": phase}
    return SQLResult(rows=[value], count=1, message="inspect", scalar=value)


def _execute_begin(db) -> SQLResult:
    if not hasattr(db, "begin_sql_transaction"):
        raise ValueError("BEGIN is only supported on Database sessions")
    db.begin_sql_transaction()
    return SQLResult(count=0, message="BEGIN")


def _execute_commit(db) -> SQLResult:
    if not hasattr(db, "commit_sql_transaction"):
        raise ValueError("COMMIT is only supported on Database sessions")
    db.commit_sql_transaction()
    return SQLResult(count=0, message="COMMIT")


def _execute_rollback(db) -> SQLResult:
    if not hasattr(db, "rollback_sql_transaction"):
        raise ValueError("ROLLBACK is only supported on Database sessions")
    db.rollback_sql_transaction()
    return SQLResult(count=0, message="ROLLBACK")


def _execute_union(db: Database, tree: exp.Union) -> SQLResult:
    left = _execute_query_node(db, tree.this)
    right = _execute_query_node(db, tree.expression)
    rows = _align_union_rows(left.rows, right.rows)
    if tree.args.get("distinct", True):
        rows = _distinct_rows(rows)
    return SQLResult(rows=rows, count=len(rows))


def _execute_query_node(db: Database, node: exp.Expression) -> SQLResult:
    if isinstance(node, exp.Select):
        return _execute_select(db, node, None)
    if isinstance(node, exp.Union):
        return _execute_union(db, node)
    raise ValueError(f"unsupported query node in SQL composition: {type(node).__name__}")


def _execute_create_index(db: Database, tree: exp.Create) -> SQLResult:
    index = tree.this
    if not isinstance(index, exp.Index) or not isinstance(index.args.get("table"), exp.Table):
        raise ValueError("CREATE INDEX requires a target table and column")
    params = index.args.get("params")
    columns = params.args.get("columns") if params is not None else None
    if not columns or len(columns) != 1:
        raise ValueError("only single-column CREATE INDEX is supported")
    ordered = columns[0]
    column_node = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(column_node, exp.Column):
        raise ValueError("CREATE INDEX requires a simple column")
    index_name = index.name
    table_name = index.args["table"].name
    column_name = column_node.name
    db.create_index(table_name, column_name, index_name, if_not_exists=bool(tree.args.get("exists")))
    return SQLResult(count=0, message=f"Created index {index_name} on {table_name}({column_name})")


def _execute_drop_index(db: Database, tree: exp.Drop) -> SQLResult:
    index_name = _table_name(tree.this)
    db.drop_index(index_name, if_exists=bool(tree.args.get("exists")))
    return SQLResult(count=0, message=f"Dropped index {index_name}")


def _parse_source(tree: exp.Select) -> _SelectSource:
    from_node = tree.args.get("from_")
    if not from_node or not isinstance(from_node.this, exp.Table):
        raise ValueError("SELECT source must be a table")
    left_table = from_node.this.name
    left_alias = from_node.this.alias_or_name
    joins = tree.args.get("joins") or []
    if not joins:
        return _SelectSource(kind="table", table=left_table, alias=left_alias)
    parsed_joins = []
    for join in joins:
        if not isinstance(join.this, exp.Table):
            raise ValueError("JOIN source must be a table")
        right_table = join.this.name
        right_alias = join.this.alias_or_name
        left_on, right_on = _parse_join_on_ast(join.args.get("on"), right_alias)
        side = (join.args.get("side") or "").upper()
        kind = (join.args.get("kind") or "").upper()
        if side not in {"", "LEFT", "RIGHT", "FULL", "INNER"}:
            raise ValueError(f"{side} JOIN is not yet supported")
        if kind not in {"", "OUTER"}:
            raise ValueError(f"{kind} JOIN is not yet supported")
        outer = None
        if side == "LEFT":
            outer = "left"
        elif side == "RIGHT":
            outer = "right"
        elif side == "FULL":
            outer = "full"
        parsed_joins.append(
            _JoinSpec(
                table=right_table,
                alias=right_alias,
                left_expr=left_on,
                right_col=right_on,
                outer=outer,
            )
        )
    return _SelectSource(kind="join", table=left_table, alias=left_alias, joins=parsed_joins)


def _parse_join_on_ast(node: exp.Expression | None, right_alias: str) -> tuple[str, str]:
    if not isinstance(node, exp.EQ) or not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Column):
        raise ValueError("JOIN ON must be a qualified equality")
    left = node.this
    right = node.expression
    if left.table == right_alias and right.table != right_alias:
        return _column_sql(right), left.name
    if right.table == right_alias and left.table != right_alias:
        return _column_sql(left), right.name
    raise ValueError("JOIN ON aliases do not match source aliases")


def _parse_projection(node: exp.Expression) -> _Projection:
    alias = node.alias
    expr = node.this if isinstance(node, exp.Alias) else node
    special = _parse_special_function(expr)
    aggregate = _parse_aggregate_expr(expr)
    if not alias:
        alias = _default_alias(expr, aggregate, special)
    return _Projection(alias=alias, expr_sql=expr.sql(), aggregate=aggregate, special=special)


def _default_alias(expr: exp.Expression, aggregate, special: str | None) -> str:
    if special is not None:
        return special
    if aggregate is not None:
        return aggregate[0]
    if isinstance(expr, exp.Column):
        return _column_sql(expr)
    return expr.sql()


def _parse_special_function(expr: exp.Expression) -> str | None:
    if isinstance(expr, exp.Anonymous):
        name = expr.name.upper()
        if name == "IDENTITY":
            return "identity"
        if name == "DRIFT":
            return "drift"
        if name == "DECOMPOSE_DRIFT":
            return "decompose_drift"
    return None


def _parse_aggregate_expr(expr: exp.Expression) -> tuple[str, str] | None:
    mapping = {
        exp.Count: "count",
        exp.Sum: "sum",
        exp.Avg: "avg",
        exp.Min: "min",
        exp.Max: "max",
    }
    for cls, name in mapping.items():
        if isinstance(expr, cls):
            target = "*"
            target_node = expr.this
            if target_node is not None and not isinstance(target_node, exp.Star):
                target = target_node.sql()
            return name, target
    return None


def _parse_group(group: exp.Group | None) -> list[str]:
    if not group:
        return []
    return [expr.sql() for expr in group.expressions]


def _parse_order(order: exp.Order | None) -> tuple[str | None, bool]:
    if not order or not order.expressions:
        return None, False
    first = order.expressions[0]
    return first.this.sql(), bool(first.args.get("desc"))


def _parse_int_arg(node: exp.Expression | None) -> int | None:
    if node is None:
        return None
    this = getattr(node, "this", None)
    if this is None and hasattr(node, "expression"):
        this = node.expression
    if isinstance(this, exp.Literal):
        return int(this.this)
    if isinstance(node, exp.Literal):
        return int(node.this)
    return int(node.sql())


def _ast_to_where(node: exp.Expression, db: Database, source: _SelectSource | None):
    if isinstance(node, exp.Paren):
        return _ast_to_where(node.this, db, source)
    if isinstance(node, exp.And):
        return {"and": [_ast_to_where(node.this, db, source), _ast_to_where(node.expression, db, source)]}
    if isinstance(node, exp.Or):
        return {"or": [_ast_to_where(node.this, db, source), _ast_to_where(node.expression, db, source)]}
    if isinstance(node, exp.Not):
        inner = node.this
        if isinstance(inner, exp.In):
            left = _expr_sql(inner.this)
            vals = _in_values(inner, db)
            return left, "not in", vals
        if isinstance(inner, exp.Between):
            left = _expr_sql(inner.this)
            low = _ast_to_literal(inner.args["low"], db, source)
            high = _ast_to_literal(inner.args["high"], db, source)
            return {"or": [(left, "<", low), (left, ">", high)]}
        if isinstance(inner, exp.Exists):
            return {"not_exists": inner.this}
        raise ValueError("unsupported NOT expression in WHERE")
    if isinstance(node, exp.In):
        left = _expr_sql(node.this)
        vals = _in_values(node, db)
        return left, "in", vals
    if isinstance(node, exp.Between):
        left = _expr_sql(node.this)
        low = _ast_to_literal(node.args["low"], db, source)
        high = _ast_to_literal(node.args["high"], db, source)
        return {"and": [(left, ">=", low), (left, "<=", high)]}
    if isinstance(node, exp.Exists):
        return {"exists": node.this}
    if isinstance(node, exp.Like):
        return _expr_sql(node.this), "like", _ast_to_literal(node.expression, db, source)
    if isinstance(node, exp.Is):
        left = _expr_sql(node.this)
        right = node.expression
        if isinstance(right, exp.Null):
            return left, "is", None
        return left, "=", _ast_to_literal(right, db, source)
    if isinstance(node, exp.NullSafeEQ):
        return _expr_sql(node.this), "=", _ast_to_literal(node.expression, db, source)

    op_map = {
        exp.EQ: "=",
        exp.NEQ: "!=",
        exp.GT: ">",
        exp.GTE: ">=",
        exp.LT: "<",
        exp.LTE: "<=",
    }
    for cls, op in op_map.items():
        if isinstance(node, cls):
            return _expr_sql(node.this), op, _ast_to_literal(node.expression, db, source)
    raise ValueError(f"unsupported WHERE expression: {type(node).__name__}")


def _in_values(node: exp.In, db: Database) -> list[object]:
    if node.args.get("query") is not None:
        rows = execute(db, node.args["query"].this.sql()).rows
        if not rows:
            return []
        if len(rows[0]) != 1:
            raise ValueError("subquery IN must return exactly one column")
        col = next(iter(rows[0].keys()))
        return [row[col] for row in rows]
    return [_ast_to_literal(expr, db, None) for expr in node.expressions]


def _ast_to_literal(node: exp.Expression, db: Database, source: _SelectSource | None):
    if isinstance(node, exp.Paren):
        return _ast_to_literal(node.this, db, source)
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        text = node.this
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    if isinstance(node, exp.Subquery):
        rows = execute(db, node.this.sql()).rows
        if not rows:
            return None
        if len(rows[0]) != 1:
            raise ValueError("scalar subquery must return exactly one column")
        return next(iter(rows[0].values()))
    return _expr_sql(node)


def _materialize_source_rows(db: Database, source: _SelectSource) -> list[dict]:
    if source.kind == "table":
        rows = db.select(source.table)
        if source.alias == source.table:
            return rows
        return [_alias_base_row(row, source) for row in rows]
    base_schema = db.schema(source.table)
    rows = [_qualified_table_row(source.table, source.alias, row, base_schema) for row in db.select(source.table)]
    seen_sources = [(source.table, source.alias, base_schema)]
    for join in source.joins:
        right_schema = db.schema(join.table)
        right_rows = db.select(join.table)
        rows = _join_rows(rows, right_rows, seen_sources, join, right_schema)
        seen_sources.append((join.table, join.alias, right_schema))
    return rows


def _execute_search_select(db: Database, source: _SelectSource, search_values: list[object], limit: int | None) -> list[dict]:
    if source.kind != "table":
        raise ValueError("RESONATE NEAR currently requires a single table source")
    table = db.table(source.table)
    schema = db.schema(source.table)
    query = coerce_row(table.schema(), search_values)
    hits = table.search(query, k=limit or 1)
    rows = []
    for hit in hits:
        record = {}
        base = table.get_row(hit.position)
        for col, value in zip(schema, base, strict=True):
            record[col["name"]] = value
        record["_row"] = hit.position
        record["_drift"] = hit.drift
        record["_base"] = [float(hit.base[0]), float(hit.base[1]), float(hit.base[2])]
        record["_phase"] = hit.phase
        rows.append(record)
    return rows


def _alias_base_row(row: dict, source: _SelectSource) -> dict:
    out = dict(row)
    for key, value in row.items():
        out[f"{source.alias}.{key}"] = value
    return out


def _qualified_table_row(table: str, alias: str, row: dict, schema: list[dict]) -> dict:
    out = {}
    for col in schema:
        key = col["name"]
        value = row[key]
        out[f"{table}.{key}"] = value
        if alias != table:
            out[f"{alias}.{key}"] = value
    return out


def _qualified_null_row(table: str, alias: str, schema: list[dict]) -> dict:
    return _qualified_table_row(table, alias, {col["name"]: None for col in schema}, schema)


def _null_row_template(sources: list[tuple[str, str, list[dict]]]) -> dict:
    out = {}
    for table, alias, schema in sources:
        out.update(_qualified_null_row(table, alias, schema))
    return out


def _join_rows(
    current_rows: list[dict],
    right_rows: list[dict],
    seen_sources: list[tuple[str, str, list[dict]]],
    join: _JoinSpec,
    right_schema: list[dict],
) -> list[dict]:
    index: dict[object, list[tuple[int, dict]]] = {}
    for idx, row in enumerate(right_rows):
        key = row[join.right_col]
        if key is None:
            continue
        index.setdefault(key, []).append((idx, row))

    joined = []
    matched_right: set[int] = set()
    for left_row in current_rows:
        left_key = _value_from_row(left_row, join.left_expr)
        matches = [] if left_key is None else index.get(left_key, [])
        if not matches and join.outer in {"left", "full"}:
            matches = [(None, None)]
        for right_idx, right_row in matches:
            record = dict(left_row)
            if right_row is None:
                record.update(_qualified_null_row(join.table, join.alias, right_schema))
            else:
                matched_right.add(right_idx)
                record.update(_qualified_table_row(join.table, join.alias, right_row, right_schema))
            joined.append(record)

    if join.outer in {"right", "full"}:
        left_nulls = _null_row_template(seen_sources)
        for idx, right_row in enumerate(right_rows):
            if idx in matched_right:
                continue
            record = dict(left_nulls)
            record.update(_qualified_table_row(join.table, join.alias, right_row, right_schema))
            joined.append(record)
    return joined


def _apply_where(rows: list[dict], where, source: _SelectSource, db: Database) -> list[dict]:
    if where is None:
        return rows
    return [row for row in rows if _row_matches_where(row, where, source, db)]


def _group_rows(rows: list[dict], projections: list[_Projection], group_by: list[str], source: _SelectSource) -> list[dict]:
    normalized_group = [_normalize_expr(expr, source) for expr in group_by]
    buckets: dict[tuple[object, ...], list[dict]] = {}
    for row in rows:
        key = tuple(_value_from_row(row, expr) for expr in normalized_group)
        buckets.setdefault(key, []).append(row)

    result = []
    for bucket in buckets.values():
        entry = {}
        for projection in projections:
            if projection.aggregate is not None:
                func, target = projection.aggregate
                value = _compute_aggregate(bucket, _normalize_expr(target, source), func)
                entry[projection.alias] = value
                entry[projection.expr_sql] = value
            else:
                value = _value_from_row(bucket[0], _normalize_expr(projection.expr_sql, source))
                entry[projection.alias] = value
                entry[projection.expr_sql] = value
        result.append(entry)
    return result


def _project_row(row: dict, projections: list[_Projection], source: _SelectSource) -> dict:
    out = {}
    for projection in projections:
        out[projection.alias] = _value_from_row(row, _normalize_expr(projection.expr_sql, source))
    return out


def _is_star_projection(projections: list[_Projection]) -> bool:
    return len(projections) == 1 and projections[0].expr_sql == "*"


def _normalize_expr(expr: str, source: _SelectSource) -> str:
    normalized = expr.strip()
    if source.kind == "table" and normalized.startswith(f"{source.alias}."):
        return normalized.split(".", 1)[1]
    return normalized


def _compute_aggregate(rows: list[dict], target: str, func: str):
    if func == "count":
        if target == "*":
            return len(rows)
        return sum(1 for row in rows if _value_from_row(row, target) is not None)
    values = [_value_from_row(row, target) for row in rows if _value_from_row(row, target) is not None]
    if func == "sum":
        return sum(float(v) for v in values)
    if func == "avg":
        return sum(float(v) for v in values) / len(values) if values else 0.0
    if func == "min":
        return min(values) if values else None
    if func == "max":
        return max(values) if values else None
    raise ValueError(f"unsupported aggregate: {func}")


def _compute_special_select(db: Database, table_name: str, func: str):
    table = db.table(table_name)
    if func == "identity":
        return table.identity().tolist()
    if func == "drift":
        return float(table.check())
    if func == "decompose_drift":
        drift, base, phase = table.check_hopf()
        return {"drift": drift, "base": [float(base[0]), float(base[1]), float(base[2])], "phase": phase}
    raise ValueError(f"unsupported special function: {func}")


def _apply_group_having(rows: list[dict], having) -> list[dict]:
    schema = [{"name": key, "type": _infer_type_from_value(value)} for key, value in rows[0].items()] if rows else []
    return [row for row in rows if _match_where(row, schema, having)]


def _strip_group_auxiliary(rows: list[dict], projections: list[_Projection]) -> list[dict]:
    keys = [projection.alias for projection in projections]
    return [{key: row.get(key) for key in keys} for row in rows]


def _distinct_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    distinct = []
    for row in rows:
        key = tuple((k, _hashable(v)) for k, v in row.items())
        if key in seen:
            continue
        seen.add(key)
        distinct.append(row)
    return distinct


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _hashable(v)) for k, v in sorted(value.items()))
    return value


def _sort_projected_records(records: list[dict], order_by: str, descending: bool) -> list[dict]:
    present = [record for record in records if _value_from_row(record, order_by) is not None]
    missing = [record for record in records if _value_from_row(record, order_by) is None]
    present.sort(key=lambda record: _value_from_row(record, order_by), reverse=descending)
    return present + missing


def _value_from_row(row: dict, expr: str):
    if expr == "*":
        return row
    if expr in row:
        return row[expr]
    return db_evaluate_expression(expr, row)


def _table_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Table):
        return node.name
    raise ValueError(f"expected table reference, got {type(node).__name__}")


def _expr_sql(node: exp.Expression) -> str:
    if isinstance(node, exp.Column):
        return _column_sql(node)
    return node.sql()


def _column_sql(node: exp.Column) -> str:
    return f"{node.table}.{node.name}" if node.table else node.name


def _infer_type_from_value(value):
    if value is None:
        return "bytes"
    if isinstance(value, bool):
        return "i64"
    if isinstance(value, int):
        return "i64"
    if isinstance(value, float):
        return "f64"
    return "bytes"


def _map_sql_type(node: exp.DataType | None) -> str:
    if node is None:
        raise ValueError("ALTER TABLE ADD COLUMN requires a type")
    raw = node.sql().upper()
    if raw in {"I64", "BIGINT", "INT", "INTEGER", "SMALLINT", "TINYINT"}:
        return "i64"
    if raw in {"FLOAT", "DOUBLE", "DOUBLE PRECISION", "REAL", "DECIMAL", "NUMERIC"}:
        return "f64"
    if raw in {"TEXT", "VARCHAR", "CHAR", "NVARCHAR", "BINARY", "VARBINARY", "BLOB", "BYTES"}:
        return "bytes"
    raise ValueError(f"unsupported SQL type: {node.sql()}")


def _column_def_to_schema(node: exp.ColumnDef) -> dict:
    column = {
        "name": node.this.name,
        "type": _map_sql_type(node.kind),
        "indexed": False,
        "primary": False,
        "references": None,
        "not_null": False,
        "unique": False,
    }
    for constraint in node.args.get("constraints") or []:
        kind = constraint.args.get("kind")
        if isinstance(kind, exp.NotNullColumnConstraint):
            column["not_null"] = True
        elif isinstance(kind, exp.UniqueColumnConstraint):
            column["unique"] = True
        elif isinstance(kind, exp.PrimaryKeyColumnConstraint):
            column["primary"] = True
            column["not_null"] = True
            column["unique"] = True
        elif isinstance(kind, exp.Reference):
            if not isinstance(kind.this, exp.Schema) or not isinstance(kind.this.this, exp.Table):
                raise ValueError("unsupported REFERENCES constraint")
            target_table = kind.this.this.name
            if len(kind.this.expressions) != 1:
                raise ValueError("REFERENCES must target exactly one column")
            target_column = kind.this.expressions[0].name
            column["references"] = f"{target_table}.{target_column}"
        else:
            raise ValueError(f"unsupported CREATE TABLE constraint: {type(kind).__name__}")
    return column


def _extract_resonate_clause(sql: str) -> tuple[str, list[object] | None]:
    upper = sql.upper()
    marker = " RESONATE NEAR "
    idx = upper.find(marker)
    if idx < 0:
        return sql, None
    open_idx = sql.find("(", idx)
    if open_idx < 0:
        raise ValueError("expected RESONATE NEAR (...)")
    values_text, remaining = _extract_parenthesized(sql, open_idx)
    values = [_parse_sql_literal(part) for part in _split_top_level(values_text)]
    stripped = (sql[:idx] + remaining).strip()
    return stripped, values


def _parse_sql_literal(text: str):
    value = text.strip()
    if not value:
        return None
    upper = value.upper()
    if upper == "NULL":
        return None
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        inner = value[1:-1]
        if value.startswith("'"):
            return inner.replace("''", "'")
        return inner.replace('\\"', '"')
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_top_level(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if quote == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _extract_parenthesized(text: str, open_idx: int) -> tuple[str, str]:
    depth = 0
    quote = None
    i = open_idx
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if quote == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], text[i + 1 :]
        i += 1
    raise ValueError("unterminated parenthesized SQL clause")


def _split_sql_script(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if quote == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if ch == ";" and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _align_union_rows(left_rows: list[dict], right_rows: list[dict]) -> list[dict]:
    if not left_rows:
        return list(right_rows)
    if not right_rows:
        return list(left_rows)
    keys = list(left_rows[0].keys())
    aligned = list(left_rows)
    for row in right_rows:
        if list(row.keys()) == keys:
            aligned.append(row)
            continue
        if len(row) != len(keys):
            raise ValueError("UNION queries must return the same number of columns")
        aligned.append({key: value for key, value in zip(keys, row.values(), strict=True)})
    return aligned


def _row_matches_where(row: dict, where, source: _SelectSource, db: Database) -> bool:
    if isinstance(where, dict):
        if "and" in where:
            return all(_row_matches_where(row, clause, source, db) for clause in where["and"])
        if "or" in where:
            return any(_row_matches_where(row, clause, source, db) for clause in where["or"])
        if "exists" in where:
            return _exists_subquery(db, where["exists"], row)
        if "not_exists" in where:
            return not _exists_subquery(db, where["not_exists"], row)
    if isinstance(where, tuple):
        col, op, value = where
        return _match_where(
            row,
            [{"name": key, "type": _infer_type_from_value(val)} for key, val in row.items()],
            (_normalize_expr(col, source), op, value),
        )
    schema = [{"name": key, "type": _infer_type_from_value(value)} for key, value in row.items()]
    return _match_where(row, schema, where)


def _exists_subquery(db: Database, query: exp.Expression, outer_row: dict) -> bool:
    bound = _bind_outer_row(query, outer_row)
    return execute(db, bound.sql()).count > 0


def _bind_outer_row(node: exp.Expression, outer_row: dict) -> exp.Expression:
    query = node.copy()
    local_aliases = _collect_query_aliases(query)

    def transform(expr: exp.Expression):
        if isinstance(expr, exp.Column):
            key = expr.sql()
            if expr.table not in local_aliases:
                if key in outer_row:
                    return _literal_to_ast(outer_row[key])
                if expr.name in outer_row:
                    return _literal_to_ast(outer_row[expr.name])
        return expr

    return query.transform(transform)


def _collect_query_aliases(node: exp.Expression) -> set[str]:
    aliases: set[str] = set()
    for table in node.find_all(exp.Table):
        aliases.add(table.name)
        aliases.add(table.alias_or_name)
    return aliases


def _literal_to_ast(value):
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, int) and not isinstance(value, bool):
        return exp.Literal.number(value)
    if isinstance(value, float):
        return exp.Literal.number(repr(value))
    if isinstance(value, bytes):
        return exp.Literal.string(value.decode("utf-8", errors="replace"))
    return exp.Literal.string(str(value))
