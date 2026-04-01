"""CLI for the Closure DNA product surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .database import Database
from .demo import available_demos, build_all_demo_databases, build_demo_database, ensure_demo_database
from .query import coerce_for_column, coerce_row
from .repl import repl
from .sql import SQLResult
from .web import serve as serve_web


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="closure-dna")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-db")
    p.add_argument("path", type=Path)

    p = sub.add_parser("create-table")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("schema_json")

    p = sub.add_parser("add-column")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("column_json")
    p.add_argument("--default-json", default="null")

    p = sub.add_parser("tables")
    p.add_argument("database", type=Path)

    p = sub.add_parser("schema")
    p.add_argument("database", type=Path)
    p.add_argument("table")

    p = sub.add_parser("count")
    p.add_argument("database", type=Path)
    p.add_argument("table")

    p = sub.add_parser("check")
    p.add_argument("database", type=Path)
    p.add_argument("table")

    p = sub.add_parser("audit")
    p.add_argument("database", type=Path)
    p.add_argument("table", nargs="?")

    p = sub.add_parser("repair")
    p.add_argument("database", type=Path)
    p.add_argument("table", nargs="?")

    p = sub.add_parser("info")
    p.add_argument("database", type=Path)
    p.add_argument("table", nargs="?")

    p = sub.add_parser("get")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("row", type=int)

    p = sub.add_parser("insert")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("values_json")

    p = sub.add_parser("update")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("row", type=int)
    p.add_argument("values_json")

    p = sub.add_parser("delete")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("row", type=int)

    p = sub.add_parser("update-where")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("set_json", help='{"col": value, ...}')
    p.add_argument("where_json", help='[["col", "op", value], ...]')

    p = sub.add_parser("delete-where")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("where_json", help='[["col", "op", value], ...]')

    p = sub.add_parser("group-by")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("column")
    p.add_argument("aggregates_json", help='[["result_name", "func", "col"], ...]')
    p.add_argument("--where-json", default=None)
    p.add_argument("--order-by", default=None)
    p.add_argument("--desc", action="store_true")

    p = sub.add_parser("compact")
    p.add_argument("database", type=Path)
    p.add_argument("table", nargs="?")

    p = sub.add_parser("filter")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("column")
    p.add_argument("op")
    p.add_argument("value_json")

    p = sub.add_parser("select")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("where_json", nargs="?", default="-")
    p.add_argument("--order-by", default=None)
    p.add_argument("--desc", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("join")
    p.add_argument("database", type=Path)
    p.add_argument("left_table")
    p.add_argument("right_table")
    p.add_argument("left_on")
    p.add_argument("--right-on", default=None)
    p.add_argument("--outer", choices=("left",), default=None)
    p.add_argument("--where-json", default=None)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("sum")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("column")

    p = sub.add_parser("avg")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("column")

    p = sub.add_parser("sort")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("column")
    p.add_argument("--desc", action="store_true")

    p = sub.add_parser("search")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("values_json")
    p.add_argument("-k", type=int, default=1)

    p = sub.add_parser("export")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("format", choices=("json", "csv"))
    p.add_argument("output", type=Path)

    p = sub.add_parser("import")
    p.add_argument("database", type=Path)
    p.add_argument("table")
    p.add_argument("format", choices=("json", "csv"))
    p.add_argument("input", type=Path)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--no-header", action="store_true")

    p = sub.add_parser("repl")
    p.add_argument("database", type=Path)

    p = sub.add_parser("sql")
    p.add_argument("database", type=Path)
    p.add_argument("query")

    p = sub.add_parser("web")
    p.add_argument("database", type=Path, nargs="?")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--once", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("demo-databases")

    p = sub.add_parser("build-demo-db")
    p.add_argument("name")
    p.add_argument("--no-replace", action="store_true")

    p = sub.add_parser("web-demo")
    p.add_argument("name")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rebuild", action="store_true")

    args = parser.parse_args(argv)
    return _run(args)


def _run(args) -> int:
    cmd = args.command

    if cmd == "create-db":
        Database.create(args.path)
        print(args.path)
        return 0

    if cmd == "repl":
        repl(str(args.database))
        return 0

    if cmd == "sql":
        with Database.open(args.database) as db:
            result = db.execute(args.query)
            print(_jsonable_sql_result(result))
        return 0

    if cmd == "web":
        return serve_web(
            args.database,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            once=args.once,
        )

    if cmd == "demo-databases":
        print(json.dumps(available_demos()))
        return 0

    if cmd == "build-demo-db":
        if args.name == "all":
            paths = build_all_demo_databases(replace=not args.no_replace)
            print(json.dumps([str(path) for path in paths]))
        else:
            path = build_demo_database(args.name, replace=not args.no_replace)
            print(path)
        return 0

    if cmd == "web-demo":
        path = build_demo_database(args.name, replace=True) if args.rebuild else ensure_demo_database(args.name)
        return serve_web(
            path,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            once=args.once,
        )

    with Database.open(args.database) as db:
        if cmd == "create-table":
            schema = json.loads(args.schema_json)
            db.create_table(args.table, schema)
            print(args.table)
        elif cmd == "add-column":
            column = json.loads(args.column_json)
            default = json.loads(args.default_json)
            db.add_column(args.table, column, default=default)
            print(args.table)
        elif cmd == "tables":
            print(json.dumps(db.tables()))
        elif cmd == "schema":
            print(json.dumps(db.schema(args.table)))
        elif cmd == "count":
            print(db.table(args.table).count())
        elif cmd == "check":
            print(db.table(args.table).check())
        elif cmd == "audit":
            print(json.dumps(db.audit(args.table)))
        elif cmd == "repair":
            print(json.dumps(db.repair(args.table)))
        elif cmd == "info":
            print(json.dumps(db.info(args.table)))
        elif cmd == "get":
            print(json.dumps(_jsonable(db.table(args.table).get_row(args.row))))
        elif cmd == "insert":
            table = db.table(args.table)
            print(table.insert(coerce_row(table.schema(), json.loads(args.values_json))))
        elif cmd == "update":
            table = db.table(args.table)
            table.update(args.row, coerce_row(table.schema(), json.loads(args.values_json)))
            print("OK")
        elif cmd == "delete":
            db.table(args.table).delete(args.row)
            print("OK")
        elif cmd == "update-where":
            n = db.update_where(args.table, json.loads(args.set_json), json.loads(args.where_json))
            print(json.dumps({"updated": n}))
        elif cmd == "delete-where":
            n = db.delete_where(args.table, json.loads(args.where_json))
            print(json.dumps({"deleted": n}))
        elif cmd == "group-by":
            where = json.loads(args.where_json) if args.where_json else None
            rows = db.group_by(
                args.table,
                args.column,
                json.loads(args.aggregates_json),
                where=where,
                order_by=args.order_by,
                descending=args.desc,
            )
            print(json.dumps(_jsonable_rows(rows)))
        elif cmd == "compact":
            print(json.dumps(db.compact(args.table)))
        elif cmd == "filter":
            table = db.table(args.table)
            value = coerce_for_column(table.schema(), args.column, json.loads(args.value_json))
            if isinstance(value, bytes):
                print(json.dumps(table.filter_equals(args.column, value)))
            else:
                print(json.dumps(table.filter_cmp(args.column, args.op, float(value))))
        elif cmd == "select":
            where = None if args.where_json == "-" else json.loads(args.where_json)
            rows = db.select(
                args.table,
                where=where,
                order_by=args.order_by,
                descending=args.desc,
                limit=args.limit,
            )
            print(json.dumps(_jsonable_rows(rows)))
        elif cmd == "join":
            where = json.loads(args.where_json) if args.where_json else None
            rows = db.join(
                args.left_table,
                args.right_table,
                args.left_on,
                right_on=args.right_on,
                outer=args.outer,
                where=where,
                limit=args.limit,
            )
            print(json.dumps(_jsonable_rows(rows)))
        elif cmd == "sum":
            print(db.table(args.table).sum(args.column))
        elif cmd == "avg":
            print(db.table(args.table).avg(args.column))
        elif cmd == "sort":
            print(json.dumps(db.table(args.table).argsort(args.column, descending=args.desc)))
        elif cmd == "search":
            table = db.table(args.table)
            hits = table.search(coerce_row(table.schema(), json.loads(args.values_json)), k=args.k)
            print(json.dumps([hit.__dict__ for hit in hits]))
        elif cmd == "export":
            db.export_table(args.table, args.format, args.output)
            print(args.output)
        elif cmd == "import":
            db.import_table(
                args.table,
                args.format,
                args.input,
                replace=args.replace,
                header=not args.no_header,
            )
            print(args.table)
    return 0


def _jsonable_sql_result(result: SQLResult) -> str:
    payload = {
        "rows": _jsonable_rows(result.rows),
        "count": result.count,
        "message": result.message,
        "scalar": _jsonable(result.scalar),
        "statements": [
            {
                "rows": _jsonable_rows(item.get("rows", [])),
                "count": item.get("count", 0),
                "message": item.get("message", ""),
                "scalar": _jsonable(item.get("scalar")),
            }
            for item in result.statements
        ],
    }
    return json.dumps(payload)


def _jsonable(values):
    if values is None:
        return None
    if isinstance(values, bytes):
        return values.decode("utf-8")
    if not isinstance(values, (list, tuple)):
        return values
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(value)
    return out


def _jsonable_rows(rows):
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, bytes):
                converted[key] = value.decode("utf-8")
            else:
                converted[key] = value
        out.append(converted)
    return out


if __name__ == "__main__":
    sys.exit(main())
