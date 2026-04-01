"""Interactive typed REPL for Closure DNA."""

from __future__ import annotations

import json
import sys

from .database import Database
from .query import coerce_for_column, coerce_row, parse, parse_json_arg


CLAUSE_KEYWORDS = {"WHERE", "ORDER", "LIMIT"}
AGG_FUNCS = {"count", "sum", "avg", "min", "max"}


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _jsonable_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({k: v.decode("utf-8") if isinstance(v, bytes) else v for k, v in row.items()})
    return out


def _parse_literal(text: str):
    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_where_tokens(tokens: list[str]):
    """Parse WHERE tokens into a condition tree supporting AND and OR.

    Returns a single condition tuple, a flat list (all AND), or a
    {"and": [...]} / {"or": [...]} dict for mixed logic.

    Precedence: AND binds tighter than OR, matching standard SQL.
      city = Tokyo OR city = Paris AND age > 18
      → OR( city=Tokyo, AND(city=Paris, age>18) )
    """
    if not tokens:
        raise ValueError("expected WHERE conditions")

    def _read_condition(toks, i):
        if i + 2 >= len(toks):
            raise ValueError(f"expected condition at position {i}")
        return (toks[i], toks[i + 1], _parse_literal(toks[i + 2])), i + 3

    # Split on OR first (lowest precedence)
    or_groups = []
    current = []
    i = 0
    while i < len(tokens):
        if tokens[i].upper() == "OR":
            or_groups.append(current)
            current = []
            i += 1
        else:
            current.append(tokens[i])
            i += 1
    or_groups.append(current)

    def _parse_and_group(toks):
        conditions = []
        i = 0
        while i < len(toks):
            if toks[i].upper() == "AND":
                i += 1
                continue
            cond, i = _read_condition(toks, i)
            conditions.append(cond)
        if len(conditions) == 1:
            return conditions[0]
        return {"and": conditions}

    and_groups = [_parse_and_group(g) for g in or_groups if g]
    if len(and_groups) == 1:
        return and_groups[0]
    return {"or": and_groups}


def _parse_set_tokens(tokens: list[str]) -> dict[str, object]:
    if not tokens:
        raise ValueError("expected SET assignments")
    values: dict[str, object] = {}
    i = 0
    while i < len(tokens):
        if tokens[i].upper() == "AND":
            i += 1
            continue
        if i + 2 >= len(tokens) or tokens[i + 1] != "=":
            raise ValueError("expected assignment: <column> = <value>")
        values[tokens[i]] = _parse_literal(tokens[i + 2])
        i += 3
    return values


def _parse_select_natural(tokens: list[str]) -> tuple[list[tuple[str, str, object]] | None, str | None, bool, int | None]:
    where = None
    order_by = None
    descending = False
    limit = None
    i = 0
    while i < len(tokens):
        head = tokens[i].upper()
        if head == "WHERE":
            i += 1
            start = i
            while i < len(tokens) and tokens[i].upper() not in {"ORDER", "LIMIT"}:
                i += 1
            where = _parse_where_tokens(tokens[start:i])
            continue
        if head == "ORDER":
            i += 1
            if i < len(tokens) and tokens[i].upper() == "BY":
                i += 1
            if i >= len(tokens):
                raise ValueError("expected ORDER BY <column>")
            order_by = tokens[i]
            i += 1
            if i < len(tokens) and tokens[i].upper() in {"ASC", "DESC"}:
                descending = tokens[i].upper() == "DESC"
                i += 1
            continue
        if head == "LIMIT":
            if i + 1 >= len(tokens):
                raise ValueError("expected LIMIT <n>")
            limit = int(tokens[i + 1])
            i += 2
            continue
        raise ValueError(f"unexpected token in SELECT: {tokens[i]}")
    return where, order_by, descending, limit


def _parse_group_aggregates(tokens: list[str]) -> list[tuple[str, str, str]]:
    if not tokens:
        raise ValueError("expected aggregate list after BY <column>")
    out = []
    i = 0
    while i < len(tokens):
        func = tokens[i].lower()
        if func not in AGG_FUNCS:
            raise ValueError(f"unknown aggregate: {tokens[i]}")
        if i + 1 >= len(tokens):
            raise ValueError(f"missing target for aggregate {tokens[i]}")
        target = tokens[i + 1]
        alias = f"{func}_{target.replace('*', 'all')}"
        i += 2
        if i < len(tokens) and tokens[i].upper() == "AS":
            if i + 1 >= len(tokens):
                raise ValueError("expected alias after AS")
            alias = tokens[i + 1]
            i += 2
        out.append((alias, func, target))
    return out


def _parse_group_natural(tokens: list[str]) -> tuple[str, str, list[tuple[str, str, str]], list[tuple[str, str, object]] | None, str | None, bool]:
    if len(tokens) < 3 or tokens[1].upper() != "BY":
        raise ValueError("expected: GROUP <table> BY <column> ...")
    table_name = tokens[0]
    column = tokens[2]
    rest = tokens[3:]

    where = None
    order_by = None
    descending = False

    split = len(rest)
    for idx, token in enumerate(rest):
        if token.upper() in {"WHERE", "ORDER"}:
            split = idx
            break
    aggregates = _parse_group_aggregates(rest[:split])

    i = split
    while i < len(rest):
        head = rest[i].upper()
        if head == "WHERE":
            i += 1
            start = i
            while i < len(rest) and rest[i].upper() != "ORDER":
                i += 1
            where = _parse_where_tokens(rest[start:i])
            continue
        if head == "ORDER":
            i += 1
            if i < len(rest) and rest[i].upper() == "BY":
                i += 1
            if i >= len(rest):
                raise ValueError("expected ORDER BY <column>")
            order_by = rest[i]
            i += 1
            if i < len(rest) and rest[i].upper() in {"ASC", "DESC"}:
                descending = rest[i].upper() == "DESC"
                i += 1
            continue
        raise ValueError(f"unexpected token in GROUP: {rest[i]}")

    return table_name, column, aggregates, where, order_by, descending


def _strip_leading_from(tokens: list[str]) -> list[str]:
    if tokens and tokens[0].upper() == "FROM":
        return tokens[1:]
    return tokens


def _clean_value_token(token: str) -> str:
    return token.strip().strip(",").strip("(").strip(")")


def _parse_insert_values_tokens(tokens: list[str]) -> list[object]:
    if not tokens:
        raise ValueError("expected values after INSERT INTO <table>")
    if tokens[0].upper() == "VALUES":
        values = [_parse_literal(_clean_value_token(tok)) for tok in tokens[1:]]
        return values
    return parse_json_arg(" ".join(tokens))


def _parse_add_column_tokens(tokens: list[str]) -> tuple[str, dict[str, object], object]:
    if len(tokens) < 6 or tokens[0].upper() != "TABLE" or tokens[2].upper() != "ADD" or tokens[3].upper() != "COLUMN":
        raise ValueError("expected: ALTER TABLE <table> ADD COLUMN <name> <type> [INDEXED] [NOT NULL] [UNIQUE] [DEFAULT value]")
    table_name = tokens[1]
    column: dict[str, object] = {
        "name": tokens[4],
        "type": tokens[5],
        "indexed": False,
        "not_null": False,
        "unique": False,
    }
    default = None
    i = 6
    while i < len(tokens):
        head = tokens[i].upper()
        if head == "INDEXED":
            column["indexed"] = True
            i += 1
            continue
        if head == "NOT":
            if i + 1 >= len(tokens) or tokens[i + 1].upper() != "NULL":
                raise ValueError("expected NULL after NOT")
            column["not_null"] = True
            i += 2
            continue
        if head == "UNIQUE":
            column["unique"] = True
            i += 1
            continue
        if head == "DEFAULT":
            if i + 1 >= len(tokens):
                raise ValueError("expected value after DEFAULT")
            default = _parse_literal(tokens[i + 1])
            i += 2
            continue
        raise ValueError(f"unexpected token in ALTER TABLE: {tokens[i]}")
    return table_name, column, default


def repl(database_path: str) -> None:
    db = Database.open(database_path)
    print(f"Closure DNA — {database_path}")
    print("Type HELP for commands. EXIT to quit.\n")

    while True:
        try:
            line = input("cdna> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        q = parse(line)
        if not q.verb:
            continue
        try:
            if _dispatch(db, q):
                break
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)

    db.close()


def _dispatch(db: Database, q) -> bool:
    if q.verb in {"EXIT", "QUIT"}:
        return True
    if q.verb == "HELP":
        print("TABLES")
        print("SCHEMA <table>")
        print("COUNT <table>")
        print("CHECK <table>")
        print("IDENTITY <table>")
        print("GET <table> <row>")
        print("INSERT INTO <table> VALUES <v1> <v2> ...")
        print("INSERT INTO <table> <json-array>")
        print("ALTER TABLE <table> ADD COLUMN <name> <type> [INDEXED] [NOT NULL] [UNIQUE] [DEFAULT value]")
        print("UPDATE <table> <row> <json-array>")
        print("UPDATE <table> SET <col> = <value> [<col> = <value> ...] WHERE <col> <op> <value> [AND ...]")
        print("DELETE <table> <row>")
        print("DELETE FROM <table> WHERE <col> <op> <value> [AND ...]")
        print("FILTER <table> <column> <op> <value>")
        print("SELECT * FROM <table> [WHERE <col> <op> <value> [AND ...]] [ORDER BY <col> [DESC]] [LIMIT n]")
        print("JOIN <left> <right> <left_on> [right_on] [LEFT] [where-json]")
        print("GROUP <table> BY <column> <FUNC col AS alias>... [WHERE <col> <op> <value> [AND ...]] [ORDER BY <col> [DESC]]")
        print("SUM <table> <column>")
        print("AVG <table> <column>")
        print("SORT <table> <column> [DESC]")
        print("SEARCH <table> <json-array> [k]")
        return False

    if q.verb == "TABLES":
        for name in db.tables():
            print(name)
        return False

    if q.verb == "SCHEMA":
        table = db.table(q.args[0])
        print(json.dumps(table.schema()))
        return False

    if q.verb == "COUNT":
        print(db.table(q.args[0]).count())
        return False

    if q.verb == "CHECK":
        print(db.table(q.args[0]).check())
        return False

    if q.verb == "IDENTITY":
        print(db.table(q.args[0]).identity())
        return False

    if q.verb == "GET":
        print(json.dumps(_jsonable(db.table(q.args[0]).get_row(int(q.args[1])))))
        return False

    if q.verb == "INSERT":
        args = q.args
        if args and args[0].upper() == "INTO":
            args = args[1:]
        table = db.table(args[0])
        values = coerce_row(table.schema(), _parse_insert_values_tokens(args[1:]))
        pos = table.insert(values)
        print(pos)
        return False

    if q.verb == "UPDATE":
        if q.args and q.args[0].upper() == "WHERE":
            # UPDATE WHERE <table> <set-json> <where-json>
            table_name = q.args[1]
            set_values = parse_json_arg(q.args[2])
            where = parse_json_arg(q.args[3])
            n = db.update_where(table_name, set_values, where)
            print(f"{n} row(s) updated")
        elif len(q.args) > 1 and q.args[1].upper() == "SET":
            table_name = q.args[0]
            if "WHERE" not in [arg.upper() for arg in q.args]:
                raise ValueError("expected WHERE in UPDATE <table> SET ... WHERE ...")
            where_idx = next(i for i, arg in enumerate(q.args) if arg.upper() == "WHERE")
            set_values = _parse_set_tokens(q.args[2:where_idx])
            where = _parse_where_tokens(q.args[where_idx + 1 :])
            n = db.update_where(table_name, set_values, where)
            print(f"{n} row(s) updated")
        else:
            table = db.table(q.args[0])
            row = int(q.args[1])
            values = coerce_row(table.schema(), parse_json_arg(q.args[2]))
            table.update(row, values)
            print("OK")
        return False

    if q.verb == "DELETE":
        if q.args and q.args[0].upper() == "WHERE":
            # DELETE WHERE <table> <where-json>
            table_name = q.args[1]
            where = parse_json_arg(q.args[2])
            n = db.delete_where(table_name, where)
            print(f"{n} row(s) deleted")
        else:
            args = q.args
            if args and args[0].upper() == "FROM":
                args = args[1:]
            if len(args) > 1 and args[1].upper() == "WHERE":
                table_name = args[0]
                where = _parse_where_tokens(args[2:])
                n = db.delete_where(table_name, where)
                print(f"{n} row(s) deleted")
            else:
                db.table(args[0]).delete(int(args[1]))
                print("OK")
        return False

    if q.verb == "SELECT":
        args = q.args
        columns = None
        if args and args[0] == "*":
            args = args[1:]
        args = _strip_leading_from(args)
        table_name = args[0]
        if len(args) <= 1:
            where = None
            order_by = None
            descending = False
            limit = None
        elif args[1].startswith(("[", "{")) or args[1] == "-":
            where = parse_json_arg(args[1]) if args[1] != "-" else None
            order_by = args[2] if len(args) > 2 and args[2] != "-" else None
            descending = len(args) > 3 and args[3].upper() == "DESC"
            limit = int(args[4]) if len(args) > 4 else None
        else:
            where, order_by, descending, limit = _parse_select_natural(args[1:])
        rows = db.select(table_name, columns=columns, where=where, order_by=order_by, descending=descending, limit=limit)
        print(json.dumps(_jsonable_rows(rows)))
        return False

    if q.verb == "ALTER":
        table_name, column, default = _parse_add_column_tokens(q.args)
        db.add_column(table_name, column, default=default)
        print(table_name)
        return False

    if q.verb == "JOIN":
        left = q.args[0]
        right = q.args[1]
        left_on = q.args[2]
        right_on = None
        outer = None
        where = None
        i = 3
        if i < len(q.args) and q.args[i] != "-" and q.args[i].upper() not in {"LEFT"} and not q.args[i].startswith(("[", "{")):
            right_on = q.args[i]
            i += 1
        if i < len(q.args) and q.args[i].upper() == "LEFT":
            outer = "left"
            i += 1
        if i < len(q.args):
            where = parse_json_arg(q.args[i])
        print(json.dumps(_jsonable_rows(db.join(left, right, left_on, right_on=right_on, outer=outer, where=where))))
        return False

    if q.verb == "GROUP":
        if q.args and q.args[0].upper() == "BY":
            table_name = q.args[1]
            column = q.args[2]
            aggregates = parse_json_arg(q.args[3])
            where = parse_json_arg(q.args[4]) if len(q.args) > 4 and q.args[4] != "-" else None
            order_by = q.args[5] if len(q.args) > 5 and q.args[5] != "-" else None
            descending = len(q.args) > 6 and q.args[6].upper() == "DESC"
        else:
            table_name, column, aggregates, where, order_by, descending = _parse_group_natural(q.args)
        rows = db.group_by(table_name, column, aggregates, where=where, order_by=order_by, descending=descending)
        print(json.dumps(_jsonable_rows(rows)))
        return False

    if q.verb == "FILTER":
        table = db.table(q.args[0])
        column = q.args[1]
        op = q.args[2]
        value = coerce_for_column(table.schema(), column, parse_json_arg(q.args[3]))
        if isinstance(value, bytes):
            print(table.filter_equals(column, value))
        else:
            print(table.filter_cmp(column, op, float(value)))
        return False

    if q.verb == "SUM":
        print(db.table(q.args[0]).sum(q.args[1]))
        return False

    if q.verb == "AVG":
        print(db.table(q.args[0]).avg(q.args[1]))
        return False

    if q.verb == "SORT":
        descending = len(q.args) > 2 and q.args[2].upper() == "DESC"
        print(db.table(q.args[0]).argsort(q.args[1], descending=descending))
        return False

    if q.verb == "SEARCH":
        table = db.table(q.args[0])
        values = coerce_row(table.schema(), parse_json_arg(q.args[1]))
        k = int(q.args[2]) if len(q.args) > 2 else 1
        print(table.search(values, k=k))
        return False

    raise ValueError(f"unknown command: {q.verb}")
