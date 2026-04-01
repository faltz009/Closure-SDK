"""Local web UI for Closure DNA."""

from __future__ import annotations

import json
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .database import Database


def serve(
    database_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    open_browser: bool = True,
    once: bool = False,
) -> int:
    db = Database.open(database_path) if database_path is not None else None
    browse_root = Path.cwd()
    demo_root = browse_root / "closure_ea" / "dna" / "demo" / "databases"
    repaired = []
    if db is not None:
        for name in db.tables():
            tree_path = db.path / f"{name}.cdna" / "tree.q"
            if not tree_path.exists():
                db.repair(name)
                repaired.append(name)
    state = {"db": db, "once": once, "lock": threading.RLock()}

    def current_db() -> Database | None:
        return state["db"]

    def switch_db(new_path: str | Path) -> Database:
        with state["lock"]:
            old = state["db"]
            new_db = Database.open(new_path)
            state["db"] = new_db
            if old is not None:
                old.close()
            return new_db

    def require_db() -> Database:
        db_obj = current_db()
        if db_obj is None:
            raise ValueError("no database loaded")
        return db_obj

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(_json_ready(payload)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            if state["once"]:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            db_obj = current_db()

            try:
                if path == "/":
                    self._send_html(_index_html())
                    return

                if path == "/api/info":
                    self._send_json(db_obj.info() if db_obj is not None else {"path": None, "tables": {}})
                    return

                if path == "/api/meta":
                    self._send_json(
                        {
                            "path": str(db_obj.path) if db_obj is not None else "",
                            "name": db_obj.path.stem if db_obj is not None else "",
                            "tables": db_obj.tables() if db_obj is not None else [],
                        }
                    )
                    return

                if path == "/api/tables":
                    self._send_json(db_obj.tables() if db_obj is not None else [])
                    return

                if path == "/api/schema-graph":
                    self._send_json(_schema_graph(db_obj) if db_obj is not None else {"nodes": [], "edges": []})
                    return

                if path == "/api/databases":
                    raw_path = str(query.get("path", [""])[0]).strip()
                    self._send_json(_demo_database_list(demo_root, demo_root.parent, raw_path))
                    return

                if path.startswith("/api/table/"):
                    parts = path.split("/")
                    if len(parts) < 4:
                        self._send_json({"error": "bad request"}, status=400)
                        return
                    db_obj = require_db()
                    name = urllib.parse.unquote(parts[3])
                    suffix = parts[4] if len(parts) > 4 else ""

                    if suffix == "":
                        limit = int(query.get("limit", ["100"])[0])
                        offset = int(query.get("offset", ["0"])[0])
                        self._send_json(
                            _table_bundle(db_obj, name, limit=limit, offset=offset, include_audit=False)
                        )
                        return

                    if suffix == "rows":
                        limit = int(query.get("limit", ["100"])[0])
                        offset = int(query.get("offset", ["0"])[0])
                        self._send_json(_table_rows(db_obj, name, limit=limit, offset=offset))
                        return

                    if suffix == "schema":
                        self._send_json(db_obj.schema(name))
                        return

                    if suffix == "audit":
                        self._send_json(db_obj.audit(name))
                        return

                    if suffix == "info":
                        self._send_json(db_obj.info(name))
                        return

                self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=400)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            payload = self._read_json()

            try:
                if path == "/api/open-db":
                    path_value = str(payload["path"]).strip()
                    if not path_value:
                        raise ValueError("pick a database first")
                    db_obj = switch_db(Path(path_value).expanduser())
                    self._send_json(
                        {
                            "opened": str(db_obj.path),
                            "meta": {
                                "path": str(db_obj.path),
                                "name": db_obj.path.stem,
                                "tables": db_obj.tables(),
                            },
                        }
                    )
                    return

                db_obj = require_db()
                if path == "/api/create-table":
                    name = str(payload["name"]).strip()
                    schema = payload["schema"]
                    db_obj.create_table(name, schema)
                    self._send_json(
                        {
                            "created": name,
                            "tables": db_obj.tables(),
                            "table": _table_bundle(db_obj, name, include_audit=False),
                        }
                    )
                    return

                if path == "/api/add-column":
                    name = str(payload["table"]).strip()
                    column = payload["column"]
                    default = payload.get("default")
                    db_obj.add_column(name, column, default=default)
                    self._send_json(
                        {
                            "updated": name,
                            "tables": db_obj.tables(),
                            "table": _table_bundle(db_obj, name, include_audit=False),
                        }
                    )
                    return

                if path == "/api/sql":
                    result = db_obj.execute(str(payload["sql"]))
                    self._send_json(
                        {
                            "result": {
                                "rows": result.rows,
                                "count": result.count,
                                "message": result.message,
                                "scalar": result.scalar,
                            },
                            "tables": db_obj.tables(),
                        }
                    )
                    return

                if path.startswith("/api/table/"):
                    parts = path.split("/")
                    if len(parts) < 5:
                        self._send_json({"error": "bad request"}, status=400)
                        return
                    name = urllib.parse.unquote(parts[3])
                    suffix = parts[4]

                    if suffix == "compact":
                        self._send_json(
                            {
                                "result": db_obj.compact(name),
                                "table": _table_bundle(db_obj, name, include_audit=False),
                            }
                        )
                        return

                    if suffix == "repair":
                        self._send_json(
                            {
                                "result": db_obj.repair(name),
                                "table": _table_bundle(db_obj, name, include_audit=True),
                            }
                        )
                        return

                    if suffix == "search":
                        table = db_obj.table(name)
                        hits = table.search(payload["values"], k=int(payload.get("k", 5)))
                        self._send_json({"hits": [hit.__dict__ for hit in hits]})
                        return

                    if suffix == "insert":
                        schema = db_obj.schema(name)
                        row_payload = payload["row"]
                        if isinstance(row_payload, dict):
                            values = [
                                _coerce_web_value(col["type"], row_payload[col["name"]])
                                for col in schema
                            ]
                        else:
                            values = [
                                _coerce_web_value(col["type"], value)
                                for col, value in zip(schema, row_payload, strict=True)
                            ]
                        row_id = db_obj.table(name).insert(values)
                        self._send_json(
                            {
                                "inserted": row_id,
                                "table": _table_bundle(db_obj, name, include_audit=False),
                            }
                        )
                        return

                    if suffix == "update-row":
                        row_id = int(payload["row_id"])
                        schema = db_obj.schema(name)
                        row_payload = payload["row"]
                        if isinstance(row_payload, dict):
                            values = [_coerce_web_value(col["type"], row_payload.get(col["name"])) for col in schema]
                        else:
                            values = [_coerce_web_value(col["type"], value) for col, value in zip(schema, row_payload, strict=True)]
                        db_obj.table(name).update(row_id, values)
                        self._send_json({"updated": row_id, "table": _table_bundle(db_obj, name, include_audit=False)})
                        return

                    if suffix == "delete-row":
                        row_id = int(payload["row_id"])
                        db_obj.table(name).delete(row_id)
                        self._send_json({"deleted": row_id, "table": _table_bundle(db_obj, name, include_audit=False)})
                        return

                self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    if open_browser:
        webbrowser.open(url)
    if repaired:
        print(f"Recovered derived state for: {', '.join(repaired)}")
    print(url)
    try:
        server.serve_forever()
    finally:
        db_obj = current_db()
        if db_obj is not None:
            db_obj.close()
        server.server_close()
    return 0


def _table_bundle(
    db: Database,
    name: str,
    *,
    limit: int = 100,
    offset: int = 0,
    include_audit: bool = False,
) -> dict:
    rows_payload, live_rows = _table_rows(db, name, limit=limit, offset=offset, with_live_count=True)
    info = db.info(name)
    info["schema"] = db.schema(name)
    bundle = {
        "name": name,
        "info": info,
        "schema": info["schema"],
        "rows": rows_payload,
        "paging": {
            "limit": limit,
            "offset": offset,
            "live_rows": live_rows,
            "page_count": len(rows_payload),
        },
    }
    if include_audit:
        bundle["audit"] = db.audit(name)
    return bundle


def _table_rows(
    db: Database,
    name: str,
    *,
    limit: int = 100,
    offset: int = 0,
    with_live_count: bool = False,
) -> list[dict] | tuple[list[dict], int]:
    schema = db.schema(name)
    table = db.table(name)
    live_indices = db._live_row_indices(name)
    selected = live_indices[offset : offset + limit]
    rows = []
    for row_id in selected:
        record = {"_row": row_id}
        for col, value in zip(schema, table.get_row(row_id), strict=True):
            record[col["name"]] = value
        rows.append(record)
    if with_live_count:
        return rows, len(live_indices)
    return rows


def _schema_graph(db: Database) -> dict:
    nodes = []
    edges = []
    for table_name in db.tables():
        schema = db.schema(table_name)
        nodes.append(
            {
                "id": table_name,
                "label": table_name,
                "columns": [col["name"] for col in schema],
                "primary_key": next((col["name"] for col in schema if col.get("primary")), None),
            }
        )
        for col in schema:
            ref = col.get("references")
            if not ref:
                continue
            target_table, target_col = ref.split(".", 1)
            edges.append(
                {
                    "from": table_name,
                    "to": target_table,
                    "label": f"{col['name']} -> {target_col}",
                }
            )
    return {"nodes": nodes, "edges": edges}


def _demo_database_list(demo_root: Path, root: Path, raw_path: str) -> dict[str, Any]:
    if not demo_root.exists():
        return {"path": "", "can_up": False, "entries": []}

    target = Path(raw_path).expanduser() if raw_path else demo_root
    try:
        target = target.resolve()
        root = root.resolve()
    except FileNotFoundError:
        target = demo_root.resolve()
        root = root.resolve()

    if root not in target.parents and target != root:
        target = demo_root.resolve()
    if not target.exists() or not target.is_dir():
        target = demo_root.resolve()

    entries = []
    for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
        if child.name.startswith("."):
            continue
        if child.is_dir() and child.name.endswith(".cdb"):
            entries.append({"name": child.stem, "path": str(child), "kind": "database"})
            continue
        if child.is_dir():
            entries.append({"name": child.name, "path": str(child), "kind": "directory"})

    return {
        "path": str(target),
        "can_up": target != root,
        "parent": str(target.parent) if target != root else "",
        "entries": entries,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _coerce_web_value(column_type: str, value: Any) -> Any:
    if value is None:
        return None
    if column_type == "f64":
        return float(value)
    if column_type == "i64":
        return int(value)
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _index_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Closure DNA</title>
  <style>
    :root{
      --bg:#f5f1e8;
      --paper:#fffdf8;
      --panel:#ffffff;
      --ink:#1f2937;
      --muted:#667085;
      --line:#e7dfcf;
      --accent:#16697a;
      --accent-soft:#dceff2;
      --danger:#b04b34;
      --ok:#1d7d58;
      --shadow:0 4px 14px rgba(31,41,55,.06);
      --radius:22px;
    }
    *{box-sizing:border-box}
    body{margin:0;color:var(--ink);background:var(--bg);font-family:"Avenir Next","Segoe UI",Helvetica,Arial,sans-serif}
    .shell{max-width:1680px;margin:0 auto;padding:28px}
    .hero{display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,560px);gap:24px;align-items:end;margin-bottom:22px}
    .title{margin:0;font-size:clamp(2.2rem,4vw,4rem);line-height:.95;letter-spacing:-.04em}
    .subtitle{margin:12px 0 0;max-width:820px;color:var(--muted);font-size:1.02rem}
    .pill{display:flex;align-items:center;gap:10px;padding:10px 16px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-weight:700;font-size:.84rem;max-width:100%;overflow-wrap:anywhere;justify-self:end}
    .path-label{font-size:.78rem;opacity:.8;text-transform:uppercase;letter-spacing:.12em;white-space:nowrap}
    .grid{display:grid;grid-template-columns:340px minmax(0,1fr);gap:20px;align-items:start}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
    .sidebar,.content{padding:18px}
    .section-title{margin:0 0 12px;color:var(--muted);text-transform:uppercase;font-size:.84rem;letter-spacing:.16em}
    .table-list,.stack,.graph,.edge-list{display:grid;gap:10px}
    .sidebar-block{margin-top:18px;padding-top:18px;border-top:1px solid var(--line)}
    .table-btn{width:100%;text-align:left;padding:14px 16px;border-radius:16px;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer;font:inherit;transition:.18s ease}
    .table-btn:hover,.table-btn.active{background:var(--accent-soft);border-color:var(--accent)}
    .stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:14px}
    .stat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px}
    .stat .label{display:block;color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}
    .stat .value{display:block;margin-top:6px;font-size:1.24rem;font-weight:700}
    .toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}
    button.action,.tiny-btn{cursor:pointer;font:inherit}
    button.action{border:none;border-radius:999px;padding:10px 16px;background:var(--accent);color:#fff}
    button.action.alt{background:#fff;color:var(--ink);border:1px solid var(--line)}
    button.action.danger{background:var(--danger)}
    .tiny-btn{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:999px;padding:8px 12px}
    .tiny-btn.danger{color:var(--danger);border-color:#efcdc5;background:#fff7f5}
    .tiny-btn:disabled{opacity:.45;cursor:not-allowed}
    .searchbox{display:flex;gap:8px;flex:1;min-width:320px}
    .searchbox input{flex:1;border:1px solid var(--line);border-radius:999px;padding:12px 16px;font:inherit;background:#fff}
    textarea.jsonbox,input.textbox,select.textbox{width:100%;border:1px solid var(--line);border-radius:16px;padding:12px 14px;background:#fff;color:var(--ink)}
    textarea.jsonbox,input.textbox{font:13px/1.45 ui-monospace,SFMono-Regular,monospace}
    select.textbox{font:inherit}
    textarea.jsonbox{min-height:120px;resize:vertical}
    .small,.row-editor-meta{font-size:.84rem;color:var(--muted)}
    .small.tight{line-height:1.35}
    .layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,420px);gap:16px;align-items:start}
    .two-up{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:14px}
    .card h3{margin:0 0 10px;color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.14em}
    table{width:100%;border-collapse:collapse;font-size:.95rem}
    th,td{padding:10px 8px;border-bottom:1px solid #eee4d1;text-align:left;vertical-align:top}
    th{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
    .data-table{table-layout:auto;min-width:max-content}
    .data-table th{position:sticky;top:0;background:#fffdf8;z-index:1}
    .data-table td,.data-table th{white-space:nowrap}
    .data-table td.wrap{white-space:normal;max-width:220px}
    pre{margin:0;white-space:pre-wrap;word-break:break-word;font:13px/1.5 ui-monospace,SFMono-Regular,monospace}
    .status{margin-bottom:12px;padding:12px 14px;border-radius:16px;border:1px solid var(--line);background:#fff}
    .status.good{color:var(--ok);border-color:#b9e2d3;background:#f3fbf7}
    .status.bad{color:var(--danger);border-color:#efcdc5;background:#fff3f0}
    .muted{color:var(--muted)}
    .identity{font:12px/1.4 ui-monospace,SFMono-Regular,monospace;color:var(--muted);word-break:break-all}
    .rows-host{max-height:62vh;overflow:auto;contain:content;border:1px solid #eee4d1;border-radius:14px}
    .sidepanel-host{max-height:65vh;overflow:auto;contain:content}
    .pager{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap}
    .inline-actions{display:flex;gap:6px;align-items:center}
    .graph-node{border:1px solid var(--line);border-radius:14px;padding:12px;background:#fffdf8}
    .graph-node.active{border-color:var(--accent);background:var(--accent-soft)}
    .graph-node h4{margin:0 0 8px;font-size:.95rem}
    .graph-node ul{margin:0;padding-left:18px;color:var(--muted);font-size:.9rem}
    .edge-pill{padding:8px 10px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-size:.86rem}
    .opener-card{padding:12px;border:1px solid var(--line);border-radius:16px;background:#fffdf8}
    .opener-head{display:grid;gap:8px}
    .opener-actions{display:flex;gap:6px;justify-content:flex-end}
    .opener-dir{display:flex;align-items:center;gap:8px;min-width:0;padding:6px 8px;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--muted);font:12px/1.2 ui-monospace,SFMono-Regular,monospace}
    .opener-dir span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .dir-icon{font-size:13px;line-height:1}
    .opener-path{margin-top:8px;color:var(--muted);font:11px/1.35 ui-monospace,SFMono-Regular,monospace;word-break:break-all}
    .db-list{display:grid;gap:2px;margin-top:8px;border:1px solid var(--line);border-radius:10px;background:#fff;overflow:hidden}
    .db-link{display:grid;grid-template-columns:18px minmax(0,1fr) auto;align-items:center;gap:8px;padding:7px 10px;border:none;border-bottom:1px solid #efe6d7;background:#fff;color:var(--ink);cursor:pointer;font:inherit;text-align:left}
    .db-link:last-child{border-bottom:none}
    .db-link:hover,.db-link.active{background:var(--accent-soft)}
    .db-link.active{box-shadow:inset 2px 0 0 var(--accent)}
    .db-glyph{font-size:12px;color:var(--muted)}
    .db-name{font-weight:600;font-size:.9rem;line-height:1.2}
    .db-kind{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
    .empty-state{padding:24px 12px;color:var(--muted)}
    .disabled{opacity:.55;pointer-events:none}
    code{font:12px ui-monospace,SFMono-Regular,monospace}
    @media (max-width: 1080px){
      .grid,.layout,.two-up,.hero{grid-template-columns:1fr}
      .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
      .pill{justify-self:start}
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"hero\">
      <div>
        <h1 class=\"title\">Closure DNA</h1>
        <p class=\"subtitle\">A local database workbench for browsing tables, editing rows, running SQL, inspecting integrity, and showing the product like a real embedded database instead of a pile of commands.</p>
      </div>
      <div class=\"pill\"><span class=\"path-label\">Database</span><span id=\"dbPath\">Loading database...</span></div>
    </div>
    <div class=\"grid\">
      <aside class=\"panel sidebar\">
        <div class=\"stack\">
          <h2 class=\"section-title\" style=\"margin-bottom:0\">Open Database</h2>
          <div class=\"opener-card\">
            <div class=\"opener-head\">
              <div id=\"databaseDir\" class=\"opener-dir\"><span class=\"dir-icon\">▣</span><span>closure_ea/dna/demo/databases</span></div>
              <div class=\"opener-actions\">
                <button class=\"tiny-btn\" id=\"dbUpBtn\">Up</button>
                <button class=\"tiny-btn\" id=\"dbRefreshBtn\">Refresh</button>
              </div>
            </div>
            <div id=\"databaseList\" class=\"db-list\"></div>
            <div id=\"selectedDbPath\" class=\"opener-path\">No database selected yet.</div>
          </div>
          <div class=\"small tight\">Built-in demo databases from <code>closure_ea/dna/demo/databases</code>. Click one to open it.</div>
        </div>
        <div class=\"sidebar-block\">
          <h2 class=\"section-title\">Tables</h2>
          <div id=\"tableList\" class=\"table-list\"></div>
        </div>
        <div class=\"sidebar-block stack\">
          <h2 class=\"section-title\" style=\"margin-bottom:0\">Create Table</h2>
          <input id=\"createTableName\" class=\"textbox\" placeholder=\"table name\">
          <textarea id=\"createSchema\" class=\"jsonbox\" spellcheck=\"false\">[
  {\"name\":\"id\",\"type\":\"i64\",\"primary\":true},
  {\"name\":\"name\",\"type\":\"bytes\",\"indexed\":true}
]</textarea>
          <button class=\"action\" id=\"createTableBtn\">Create Table</button>
          <div class=\"small\">Schema JSON: one object per column.</div>
        </div>
        <div class=\"sidebar-block stack\">
          <h2 class=\"section-title\" style=\"margin-bottom:0\">Add Column</h2>
          <textarea id=\"addColumnSchema\" class=\"jsonbox\" spellcheck=\"false\">{\"name\":\"score\",\"type\":\"i64\",\"indexed\":false}</textarea>
          <input id=\"addColumnDefault\" class=\"textbox\" placeholder='default JSON, e.g. 0 or null'>
          <button class=\"action alt\" id=\"addColumnBtn\">Add Column</button>
          <div class=\"small\">Applies to the selected table and backfills existing rows.</div>
        </div>
      </aside>
      <main class=\"panel content\">
        <div id=\"statusBox\" class=\"status\">Loading...</div>
        <div id=\"stats\" class=\"stats\"></div>
        <div class=\"toolbar\">
          <button class=\"action alt\" id=\"refreshBtn\">Refresh</button>
          <button class=\"action\" id=\"auditBtn\">Audit</button>
          <button class=\"action danger\" id=\"repairBtn\">Repair</button>
          <button class=\"action alt\" id=\"compactBtn\">Compact</button>
          <div class=\"searchbox\">
            <input id=\"searchInput\" placeholder='Similarity search row JSON, e.g. [1.0,\"Alice\",\"Tokyo\"]'>
            <button class=\"action\" id=\"searchBtn\">Search</button>
          </div>
        </div>
        <section class=\"card\">
          <h3>Rows</h3>
          <div class=\"pager\">
            <div id=\"pagerStatus\" class=\"small\">Waiting for table load.</div>
            <div class=\"inline-actions\">
              <select id=\"pageSize\" class=\"textbox\" style=\"width:auto\">
                <option value=\"25\">25</option>
                <option value=\"50\">50</option>
                <option value=\"100\" selected>100</option>
                <option value=\"250\">250</option>
              </select>
              <button class=\"tiny-btn\" id=\"prevPageBtn\">Prev</button>
              <button class=\"tiny-btn\" id=\"nextPageBtn\">Next</button>
            </div>
          </div>
          <div id=\"rows\" class=\"rows-host\"></div>
        </section>
        <div class=\"two-up\" style=\"margin-top:16px\">
          <section class=\"card\">
            <h3>Table Details</h3>
            <div id=\"sidepanel\" class=\"muted sidepanel-host\">Pick a table.</div>
          </section>
          <section class=\"card\">
            <h3>Schema Graph</h3>
            <div id=\"schemaGraph\" class=\"graph\">Loading schema relationships...</div>
          </section>
        </div>
        <div class=\"layout\" style=\"margin-top:16px\">
          <section class=\"card\">
            <h3>Insert Row</h3>
            <div class=\"stack\">
              <textarea id=\"insertRow\" class=\"jsonbox\" spellcheck=\"false\" placeholder='{\"id\": 1, \"name\": \"Alice\"}'></textarea>
              <button class=\"action\" id=\"insertBtn\">Insert Row</button>
              <div class=\"small\">Use a JSON object with keys matching the selected table schema.</div>
            </div>
          </section>
          <section class=\"card\">
            <h3>Row Editor</h3>
            <div class=\"stack\">
              <div id=\"rowEditorMeta\" class=\"row-editor-meta\">Pick a row from the browser to edit or delete it.</div>
              <textarea id=\"editRow\" class=\"jsonbox\" spellcheck=\"false\" placeholder='{\"id\": 1, \"name\": \"Alice\"}'></textarea>
              <div class=\"inline-actions\">
                <button class=\"action\" id=\"updateRowBtn\">Update Row</button>
                <button class=\"action danger\" id=\"deleteRowBtn\">Delete Row</button>
                <button class=\"action alt\" id=\"clearRowBtn\">Clear</button>
              </div>
              <div class=\"small\">Delete creates a tombstone; compact later if you want the page rewritten.</div>
            </div>
          </section>
        </div>
        <div class=\"layout\" style=\"margin-top:16px\">
          <section class=\"card\">
            <h3>SQL Workbench</h3>
            <div class=\"stack\">
              <textarea id=\"sqlQuery\" class=\"jsonbox\" spellcheck=\"false\" placeholder=\"SELECT * FROM people ORDER BY id LIMIT 10\"></textarea>
              <button class=\"action\" id=\"sqlBtn\">Run SQL</button>
              <div class=\"small\">Run standard SQL plus DNA-native operations like <code>RESONATE NEAR</code>, <code>DRIFT()</code>, and <code>AUDIT</code>.</div>
            </div>
          </section>
          <section class=\"card\">
            <h3>SQL Result</h3>
            <pre id=\"sqlResult\" class=\"sidepanel-host\">Run a SQL statement to inspect rows, counts, and messages here.</pre>
          </section>
        </div>
      </main>
    </div>
  </div>
  <script>
    let currentTable = null;
    let currentOffset = 0;
    let currentLimit = 100;
    let selectedRowId = null;
    let currentRows = [];
    let lastGraph = null;
    let demoDatabases = [];
    let currentDatabaseDir = "";
    let currentDatabaseParent = "";

    async function api(path, options={}) {
      const res = await fetch(path, {headers: {"Content-Type": "application/json"}, ...options});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "request failed");
      return data;
    }

    function setStatus(kind, text) {
      const box = document.getElementById("statusBox");
      box.className = "status " + (kind || "");
      box.textContent = text;
    }

    function renderDatabaseList(payload, currentPath) {
      demoDatabases = payload.entries;
      currentDatabaseDir = payload.path || "";
      currentDatabaseParent = payload.parent || "";
      document.getElementById("databaseDir").innerHTML = `<span class="dir-icon">▣</span><span>${shortPath(currentDatabaseDir || "closure_ea/dna/demo/databases")}</span>`;
      document.getElementById("databaseDir").title = currentDatabaseDir || "";
      document.getElementById("dbUpBtn").disabled = !payload.can_up;
      const host = document.getElementById("databaseList");
      if (!payload.entries.length) {
        host.innerHTML = '<div class="small">No built-in demo databases found.</div>';
        return;
      }
      host.innerHTML = payload.entries.map(item => `
        <button class="db-link ${item.path === currentPath ? "active" : ""}" data-path="${item.path}">
          <span class="db-glyph">${item.kind === "directory" ? "▸" : "▣"}</span>
          <span class="db-name">${item.name}</span>
          <span class="db-kind">${item.kind === "directory" ? "Dir" : "Demo"}</span>
        </button>
      `).join("");
      host.querySelectorAll(".db-link").forEach(btn => btn.addEventListener("click", () => {
        const item = payload.entries.find(entry => entry.path === btn.dataset.path);
        if (!item) return;
        if (item.kind === "directory") {
          loadDatabaseBrowser(item.path).then(next => {
            renderDatabaseList(next, currentPath);
          }).catch(showError);
          return;
        }
        openDatabase(item.path).catch(showError);
      }));
    }

    async function loadDatabaseBrowser(path="") {
      const query = path ? `?path=${encodeURIComponent(path)}` : "";
      return api(`/api/databases${query}`);
    }

    function setDatabaseActionsEnabled(enabled) {
      const ids = [
        "refreshBtn","auditBtn","repairBtn","compactBtn","searchBtn","createTableBtn",
        "addColumnBtn","insertBtn","updateRowBtn","deleteRowBtn","sqlBtn","prevPageBtn","nextPageBtn","pageSize"
      ];
      ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        if ("disabled" in el) el.disabled = !enabled;
      });
      document.getElementById("rows").classList.toggle("disabled", !enabled);
      document.getElementById("sidepanel").classList.toggle("disabled", !enabled);
      document.getElementById("sqlResult").classList.toggle("disabled", !enabled);
    }

    function shortPath(path) {
      if (!path) return "";
      return path.length <= 72 ? path : "…" + path.slice(-72);
    }

    function renderStats(info) {
      const stats = [
        ["Rows", info.rows],
        ["Live Rows", info.live_rows],
        ["Tombstones", info.tombstones],
        ["Drift", Number(info.drift || 0).toExponential(3)],
        ["Columns", info.schema.length],
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value]) => `
        <div class="stat">
          <span class="label">${label}</span>
          <span class="value">${value ?? "-"}</span>
        </div>
      `).join("");
    }

    function renderPaging(paging) {
      const start = paging.live_rows === 0 ? 0 : paging.offset + 1;
      const end = paging.offset + paging.page_count;
      document.getElementById("pagerStatus").textContent = `Showing ${start}-${end} of ${paging.live_rows} live rows`;
      document.getElementById("prevPageBtn").disabled = paging.offset <= 0;
      document.getElementById("nextPageBtn").disabled = paging.offset + paging.page_count >= paging.live_rows;
    }

    function formatValue(value) {
      return typeof value === "object" && value !== null ? JSON.stringify(value) : value;
    }

    function renderRows(rows) {
      currentRows = rows;
      const host = document.getElementById("rows");
      if (!rows.length) {
        host.innerHTML = '<p class="muted" style="padding:12px">No rows in the current page.</p>';
        return;
      }
      const cols = Object.keys(rows[0]).filter(c => c !== "_row");
      host.innerHTML = `
        <table class="data-table">
          <thead><tr><th>Row</th>${cols.map(c => `<th>${c}</th>`).join("")}<th>Actions</th></tr></thead>
          <tbody>
            ${rows.map(row => `
              <tr data-rowid="${row._row}">
                <td>${row._row}</td>
                ${cols.map(c => `<td class="${typeof row[c] === "string" && row[c].length > 24 ? "wrap" : ""}">${formatValue(row[c])}</td>`).join("")}
                <td>
                  <div class="inline-actions">
                    <button class="tiny-btn" data-edit-row="${row._row}">Edit</button>
                    <button class="tiny-btn danger" data-delete-row="${row._row}">Delete</button>
                  </div>
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      host.querySelectorAll("[data-edit-row]").forEach(btn => btn.addEventListener("click", () => selectRow(Number(btn.dataset.editRow))));
      host.querySelectorAll("[data-delete-row]").forEach(btn => btn.addEventListener("click", () => deleteRow(Number(btn.dataset.deleteRow)).catch(showError)));
    }

    function exampleRowFromSchema(schema) {
      const sample = {};
      for (const col of schema) sample[col.name] = col.not_null ? (col.type === "bytes" ? "" : 0) : null;
      return JSON.stringify(sample, null, 2);
    }

    function renderSidepanel(bundle) {
      const audit = bundle.audit || null;
      const schemaRows = bundle.schema.map(col => `
        <tr>
          <td>${col.name}</td>
          <td>${col.type}</td>
          <td>${col.primary ? "PK" : col.references ? col.references : "-"}</td>
          <td>${col.not_null ? "NOT NULL" : "NULL"}</td>
          <td>${col.unique ? "UNIQUE" : "-"}</td>
        </tr>
      `).join("");
      document.getElementById("sidepanel").innerHTML = `
        <div style="display:grid;gap:14px">
          <div>
            <div class="${audit ? (audit.ok ? "status good" : "status bad") : "status"}" style="margin-bottom:0">
              ${audit ? `${audit.ok ? "Audit clean" : "Audit found drift"}${audit.tree_ok === false ? " and missing tree state" : ""}` : "Audit not run yet for this view."}
            </div>
          </div>
          <div>
            <h3 style="margin:0 0 8px;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.14em">Identity</h3>
            <div class="identity">${bundle.info.identity.join(", ")}</div>
          </div>
          <div>
            <h3 style="margin:0 0 8px;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.14em">Schema</h3>
            <table>
              <thead><tr><th>Name</th><th>Type</th><th>Relation</th><th>Null</th><th>Unique</th></tr></thead>
              <tbody>${schemaRows}</tbody>
            </table>
          </div>
          <div>
            <h3 style="margin:0 0 8px;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.14em">Audit Detail</h3>
            <pre>${audit ? JSON.stringify(audit, null, 2) : "Click Audit to run a full integrity check for this table."}</pre>
          </div>
        </div>
      `;
    }

    function renderGraph(graph, current) {
      lastGraph = graph;
      const nodes = graph.nodes.map(node => `
        <div class="graph-node ${node.id === current ? "active" : ""}">
          <h4>${node.label}</h4>
          <div class="small">${node.primary_key ? `PK: ${node.primary_key}` : "No primary key"}</div>
          <ul>${node.columns.slice(0, 8).map(col => `<li>${col}</li>`).join("")}${node.columns.length > 8 ? `<li>… ${node.columns.length - 8} more</li>` : ""}</ul>
        </div>
      `).join("");
      const edges = graph.edges.length
        ? graph.edges.map(edge => `<div class="edge-pill">${edge.from} → ${edge.to} · ${edge.label}</div>`).join("")
        : '<div class="small">No foreign-key relationships in the current database.</div>';
      document.getElementById("schemaGraph").innerHTML = `<div class="edge-list">${edges}</div><div class="graph">${nodes}</div>`;
    }

    function selectRow(rowId) {
      const row = currentRows.find(r => r._row === rowId);
      if (!row) return;
      selectedRowId = rowId;
      const editable = {...row};
      delete editable._row;
      document.getElementById("editRow").value = JSON.stringify(editable, null, 2);
      document.getElementById("rowEditorMeta").textContent = `${currentTable} row ${rowId} selected for editing.`;
    }

    async function loadDatabase() {
      const meta = await api("/api/meta");
      const databases = await loadDatabaseBrowser(currentDatabaseDir);
      document.getElementById("dbPath").textContent = meta.path ? shortPath(meta.path) : "No database loaded";
      document.getElementById("dbPath").title = meta.path || "";
      document.getElementById("selectedDbPath").textContent = meta.path || "No database selected yet.";
      renderDatabaseList(databases, meta.path || "");
      const names = meta.tables;
      document.getElementById("tableList").innerHTML = names.map(name => `<button class="table-btn ${name === currentTable ? "active" : ""}" data-table="${name}">${name}</button>`).join("");
      document.querySelectorAll(".table-btn").forEach(btn => btn.addEventListener("click", () => {
        currentOffset = 0;
        loadTable(btn.dataset.table).catch(showError);
      }));
      renderGraph(await api("/api/schema-graph"), currentTable);
      setDatabaseActionsEnabled(Boolean(meta.path));
      if (!currentTable && names.length) currentTable = names[0];
      if (currentTable) {
        await loadTable(currentTable);
      } else {
        document.getElementById("stats").innerHTML = "";
        document.getElementById("rows").innerHTML = '<div class="empty-state">Open a database first, then choose a table to start exploring rows.</div>';
        document.getElementById("sidepanel").innerHTML = '<div class="muted">No database loaded yet. Pick one of the built-in demo databases from the compact list on the left.</div>';
        document.getElementById("sqlResult").textContent = "Open a database first, then run SQL here.";
        document.getElementById("pagerStatus").textContent = "No table loaded.";
        setStatus("", meta.path ? "No tables in this database yet." : "No database loaded yet. Pick a demo database from the list.");
      }
    }

    async function loadTable(name) {
      currentTable = name;
      document.querySelectorAll(".table-btn").forEach(btn => btn.classList.toggle("active", btn.dataset.table === name));
      const bundle = await api(`/api/table/${encodeURIComponent(name)}?limit=${currentLimit}&offset=${currentOffset}`);
      renderStats(bundle.info);
      renderRows(bundle.rows);
      renderSidepanel(bundle);
      renderPaging(bundle.paging);
      if (lastGraph) renderGraph(lastGraph, currentTable);
      document.getElementById("insertRow").value = exampleRowFromSchema(bundle.schema);
      setStatus("", `${name}: loaded ${bundle.rows.length} row(s) for browsing.`);
    }

    async function runAction(kind) {
      if (!currentTable) return;
      const result = await api(`/api/table/${encodeURIComponent(currentTable)}/${kind}`, {method: "POST", body: "{}"});
      renderStats(result.table.info);
      renderRows(result.table.rows);
      renderSidepanel(result.table);
      renderPaging(result.table.paging);
      if (lastGraph) renderGraph(await api("/api/schema-graph"), currentTable);
      setStatus(kind === "repair" ? "good" : "", `${kind} finished for ${currentTable}`);
    }

    async function deleteRow(rowId) {
      if (!currentTable) return;
      const result = await api(`/api/table/${encodeURIComponent(currentTable)}/delete-row`, {method: "POST", body: JSON.stringify({row_id: rowId})});
      renderStats(result.table.info);
      renderRows(result.table.rows);
      renderSidepanel(result.table);
      renderPaging(result.table.paging);
      if (selectedRowId === rowId) {
        selectedRowId = null;
        document.getElementById("editRow").value = "";
        document.getElementById("rowEditorMeta").textContent = "Pick a row from the browser to edit or delete it.";
      }
      setStatus("good", `Deleted row ${rowId} from ${currentTable}.`);
    }

    async function openDatabase(path) {
      const result = await api("/api/open-db", {method: "POST", body: JSON.stringify({path})});
      currentTable = null;
      currentOffset = 0;
      selectedRowId = null;
      document.getElementById("editRow").value = "";
      await loadDatabase();
      setStatus("good", `Opened database ${result.opened}.`);
    }

    document.getElementById("refreshBtn").addEventListener("click", () => loadDatabase().catch(showError));
    document.getElementById("dbRefreshBtn").addEventListener("click", async () => {
      try {
        const meta = await api("/api/meta");
        renderDatabaseList(await loadDatabaseBrowser(currentDatabaseDir), meta.path || "");
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("dbUpBtn").addEventListener("click", async () => {
      if (!currentDatabaseParent) return;
      try {
        const meta = await api("/api/meta");
        renderDatabaseList(await loadDatabaseBrowser(currentDatabaseParent), meta.path || "");
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("auditBtn").addEventListener("click", async () => {
      if (!currentTable) return;
      try {
        const audit = await api(`/api/table/${encodeURIComponent(currentTable)}/audit`);
        const info = await api(`/api/table/${encodeURIComponent(currentTable)}?limit=${currentLimit}&offset=${currentOffset}`);
        info.audit = audit;
        renderStats(info.info);
        renderRows(info.rows);
        renderSidepanel(info);
        renderPaging(info.paging);
        setStatus(audit.ok ? "good" : "bad", `${currentTable}: ${audit.ok ? "audit clean" : "audit drift detected"}${audit.tree_ok === false ? " (tree missing)" : ""}`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("repairBtn").addEventListener("click", () => runAction("repair").catch(showError));
    document.getElementById("compactBtn").addEventListener("click", () => runAction("compact").catch(showError));
    document.getElementById("createTableBtn").addEventListener("click", async () => {
      try {
        const name = document.getElementById("createTableName").value.trim();
        const schema = JSON.parse(document.getElementById("createSchema").value);
        const result = await api("/api/create-table", {method: "POST", body: JSON.stringify({name, schema})});
        currentTable = result.created;
        currentOffset = 0;
        await loadDatabase();
        setStatus("good", `Created table ${result.created}.`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("addColumnBtn").addEventListener("click", async () => {
      if (!currentTable) return;
      try {
        const column = JSON.parse(document.getElementById("addColumnSchema").value);
        const defaultText = document.getElementById("addColumnDefault").value.trim();
        const defaultValue = defaultText ? JSON.parse(defaultText) : null;
        const result = await api("/api/add-column", {method: "POST", body: JSON.stringify({table: currentTable, column, default: defaultValue})});
        renderStats(result.table.info);
        renderRows(result.table.rows);
        renderSidepanel(result.table);
        renderPaging(result.table.paging);
        renderGraph(await api("/api/schema-graph"), currentTable);
        document.getElementById("insertRow").value = exampleRowFromSchema(result.table.schema);
        setStatus("good", `Added column ${column.name} to ${currentTable}.`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("insertBtn").addEventListener("click", async () => {
      if (!currentTable) return;
      try {
        const row = JSON.parse(document.getElementById("insertRow").value);
        const result = await api(`/api/table/${encodeURIComponent(currentTable)}/insert`, {method: "POST", body: JSON.stringify({row})});
        renderStats(result.table.info);
        renderRows(result.table.rows);
        renderSidepanel(result.table);
        renderPaging(result.table.paging);
        setStatus("good", `Inserted row into ${currentTable}.`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("updateRowBtn").addEventListener("click", async () => {
      if (!currentTable || selectedRowId === null) return;
      try {
        const row = JSON.parse(document.getElementById("editRow").value);
        const result = await api(`/api/table/${encodeURIComponent(currentTable)}/update-row`, {method: "POST", body: JSON.stringify({row_id: selectedRowId, row})});
        renderStats(result.table.info);
        renderRows(result.table.rows);
        renderSidepanel(result.table);
        renderPaging(result.table.paging);
        setStatus("good", `Updated row ${selectedRowId} in ${currentTable}.`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("deleteRowBtn").addEventListener("click", () => {
      if (selectedRowId === null) return;
      deleteRow(selectedRowId).catch(showError);
    });
    document.getElementById("clearRowBtn").addEventListener("click", () => {
      selectedRowId = null;
      document.getElementById("editRow").value = "";
      document.getElementById("rowEditorMeta").textContent = "Pick a row from the browser to edit or delete it.";
    });
    document.getElementById("searchBtn").addEventListener("click", async () => {
      if (!currentTable) return;
      try {
        const values = JSON.parse(document.getElementById("searchInput").value);
        const result = await api(`/api/table/${encodeURIComponent(currentTable)}/search`, {method: "POST", body: JSON.stringify({values, k: 5})});
        document.getElementById("sidepanel").innerHTML = `<h3 style="margin:0 0 10px;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.14em">Search Results</h3><pre>${JSON.stringify(result.hits, null, 2)}</pre>`;
        setStatus("", `Similarity search returned ${result.hits.length} hit(s).`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("sqlBtn").addEventListener("click", async () => {
      try {
        const sql = document.getElementById("sqlQuery").value.trim();
        const response = await api("/api/sql", {method: "POST", body: JSON.stringify({sql})});
        document.getElementById("sqlResult").textContent = JSON.stringify(response.result, null, 2);
        await loadDatabase();
        setStatus("good", response.result.message || `SQL executed (${response.result.count} row result).`);
      } catch (err) {
        showError(err);
      }
    });
    document.getElementById("pageSize").addEventListener("change", async (event) => {
      currentLimit = Number(event.target.value);
      currentOffset = 0;
      if (currentTable) await loadTable(currentTable);
    });
    document.getElementById("prevPageBtn").addEventListener("click", async () => {
      currentOffset = Math.max(0, currentOffset - currentLimit);
      if (currentTable) await loadTable(currentTable);
    });
    document.getElementById("nextPageBtn").addEventListener("click", async () => {
      currentOffset += currentLimit;
      if (currentTable) await loadTable(currentTable);
    });

    function showError(err) {
      setStatus("bad", err.message || String(err));
      document.getElementById("sidepanel").innerHTML = `<pre>${err.stack || err}</pre>`;
    }

    loadDatabase().catch(showError);
  </script>
</body>
</html>"""
