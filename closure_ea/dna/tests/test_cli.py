import json
import socket
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from closure_ea.dna.demo import demo_database_path

def fresh_dir(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cdna_cli_{name}_"))


def run_cmd(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "closure_ea.dna", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def test_cli_basic_flow():
    root = fresh_dir("basic")
    db = root / "shop.cdb"
    try:
        schema = json.dumps(
            [
                ["id", "f64", False],
                ["name", "bytes", False],
                ["city", "bytes", True],
            ]
        )
        run_cmd("create-db", str(db))
        assert run_cmd("create-table", str(db), "people", schema) == "people"
        assert run_cmd("insert", str(db), "people", '[0.0, "Alice", "Tokyo"]') == "0"
        assert run_cmd("get", str(db), "people", "0") == '[0.0, "Alice", "Tokyo"]'
        assert run_cmd("update", str(db), "people", "0", '[0.0, "Alicia", "Tokyo"]') == "OK"
        assert run_cmd("filter", str(db), "people", "city", "=", '"Tokyo"') == "[0]"
    finally:
        shutil.rmtree(root)


def test_cli_select_and_join():
    root = fresh_dir("query")
    db = root / "shop.cdb"
    try:
        customer_schema = json.dumps(
            [
                {"name": "customer_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ]
        )
        order_schema = json.dumps(
            [
                {"name": "order_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "customer_id", "type": "f64", "indexed": False, "references": "customers.customer_id"},
                {"name": "amount", "type": "f64", "indexed": False},
            ]
        )
        run_cmd("create-db", str(db))
        run_cmd("create-table", str(db), "customers", customer_schema)
        run_cmd("create-table", str(db), "orders", order_schema)

        run_cmd("insert", str(db), "customers", '[1.0, "Alice", "Tokyo"]')
        run_cmd("insert", str(db), "customers", '[2.0, "Bob", "Paris"]')
        run_cmd("insert", str(db), "orders", '[10.0, 1.0, 125.0]')
        run_cmd("insert", str(db), "orders", '[11.0, 2.0, 55.0]')

        rows = json.loads(run_cmd("select", str(db), "customers", '[["city", "=", "Tokyo"]]'))
        assert rows == [{"customer_id": 1.0, "name": "Alice", "city": "Tokyo"}]

        joined = json.loads(
            run_cmd(
                "join",
                str(db),
                "orders",
                "customers",
                "customer_id",
            )
        )
        assert len(joined) == 2
        assert joined[0]["customers.name"] == "Alice"
    finally:
        shutil.rmtree(root)


def test_cli_compact():
    root = fresh_dir("compact")
    db = root / "shop.cdb"
    try:
        schema = json.dumps(
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ]
        )
        run_cmd("create-db", str(db))
        run_cmd("create-table", str(db), "people", schema)
        run_cmd("insert", str(db), "people", '[1.0, "Alice", "Tokyo"]')
        run_cmd("insert", str(db), "people", '[2.0, "Bob", "Paris"]')
        run_cmd("delete", str(db), "people", "1")

        removed = json.loads(run_cmd("compact", str(db), "people"))
        assert removed == {"people": 1}
        rows = json.loads(run_cmd("select", str(db), "people"))
        assert rows == [{"id": 1.0, "name": "Alice", "city": "Tokyo"}]
    finally:
        shutil.rmtree(root)


def test_cli_audit_info_export_import():
    root = fresh_dir("ops")
    db = root / "shop.cdb"
    export_json = root / "people.json"
    try:
        schema = json.dumps(
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ]
        )
        run_cmd("create-db", str(db))
        run_cmd("create-table", str(db), "people", schema)
        run_cmd("insert", str(db), "people", '[1.0, "Alice", "Tokyo"]')
        run_cmd("insert", str(db), "people", '[2.0, "Bob", "Paris"]')

        audit = json.loads(run_cmd("audit", str(db), "people"))
        assert audit["ok"] is True

        info = json.loads(run_cmd("info", str(db), "people"))
        assert info["rows"] == 2
        assert info["live_rows"] == 2

        assert run_cmd("export", str(db), "people", "json", str(export_json)) == str(export_json)
        assert export_json.exists()

        assert run_cmd("import", str(db), "people_copy", "json", str(export_json)) == "people_copy"
        rows = json.loads(run_cmd("select", str(db), "people_copy"))
        assert rows[0]["name"] == "Alice"
    finally:
        shutil.rmtree(root)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_cli_web_serves_ui_and_api():
    root = fresh_dir("web")
    db = root / "shop.cdb"
    proc = None
    try:
        schema = json.dumps(
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ]
        )
        run_cmd("create-db", str(db))
        run_cmd("create-table", str(db), "people", schema)
        run_cmd("insert", str(db), "people", '[1.0, "Alice"]')

        port = _free_port()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "closure_ea.dna",
                "web",
                str(db),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-open",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        base = f"http://127.0.0.1:{port}"
        body = None
        for _ in range(30):
            try:
                body = urllib.request.urlopen(base + "/", timeout=1).read().decode("utf-8")
                break
            except Exception:
                time.sleep(0.1)
        assert body is not None
        assert "Closure DNA" in body
        info = json.loads(urllib.request.urlopen(base + "/api/info", timeout=5).read().decode("utf-8"))
        assert "people" in info["tables"]
        meta = json.loads(urllib.request.urlopen(base + "/api/meta", timeout=5).read().decode("utf-8"))
        assert meta["path"].endswith("shop.cdb")
        graph = json.loads(urllib.request.urlopen(base + "/api/schema-graph", timeout=5).read().decode("utf-8"))
        assert graph["nodes"][0]["id"] == "people"

        req = urllib.request.Request(
            base + "/api/table/people/update-row",
            data=json.dumps({"row_id": 0, "row": {"id": 1.0, "name": "Alicia"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        updated = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert updated["table"]["rows"][0]["name"] == "Alicia"

        req = urllib.request.Request(
            base + "/api/table/people/delete-row",
            data=json.dumps({"row_id": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        deleted = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert deleted["table"]["paging"]["live_rows"] == 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(root)


def test_cli_web_can_open_other_database_path():
    port = _free_port()
    browser_demo = demo_database_path("browser_profile")
    music_demo = demo_database_path("music_streaming")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "closure_ea.dna",
            "web",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-open",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(40):
            try:
                meta = json.loads(urllib.request.urlopen(base + "/api/meta", timeout=1).read().decode("utf-8"))
                if meta["tables"] == []:
                    break
                break
            except Exception:
                time.sleep(0.1)

        req = urllib.request.Request(
            base + "/api/open-db",
            data=json.dumps({"path": str(browser_demo)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loaded = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
        assert loaded["opened"].endswith("browser_profile.cdb")

        req = urllib.request.Request(
            base + "/api/open-db",
            data=json.dumps({"path": str(music_demo)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loaded = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
        assert loaded["opened"].endswith("music_streaming.cdb")

        meta = json.loads(urllib.request.urlopen(base + "/api/meta", timeout=5).read().decode("utf-8"))
        assert "listener_profiles" in meta["tables"]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_cli_sql_executes_query():
    root = fresh_dir("sql")
    db = root / "shop.cdb"
    try:
        schema = json.dumps(
            [
                {"name": "id", "type": "i64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ]
        )
        run_cmd("create-db", str(db))
        run_cmd("create-table", str(db), "people", schema)
        run_cmd("insert", str(db), "people", '[1, "Alice"]')

        result = json.loads(run_cmd("sql", str(db), "SELECT * FROM people WHERE id = 1"))
        assert result["count"] == 1
        assert result["rows"] == [{"id": 1, "name": "Alice"}]
    finally:
        shutil.rmtree(root)


def test_cli_sql_executes_script():
    root = fresh_dir("sql_script")
    db = root / "shop.cdb"
    try:
        schema = json.dumps(
            [
                {"name": "id", "type": "i64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ]
        )
        run_cmd("create-db", str(db))
        run_cmd("create-table", str(db), "people", schema)

        result = json.loads(
            run_cmd(
                "sql",
                str(db),
                "INSERT INTO people VALUES (1, 'Alice'); INSERT INTO people VALUES (2, 'Bob'); SELECT * FROM people ORDER BY id",
            )
        )
        assert result["count"] == 2
        assert result["rows"] == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        assert len(result["statements"]) == 3
    finally:
        shutil.rmtree(root)


def test_cli_demo_database_registry_and_build():
    demos = json.loads(run_cmd("demo-databases"))
    names = {entry["name"] for entry in demos}
    assert {"browser_profile", "chat_app", "music_streaming"} <= names

    built = Path(run_cmd("build-demo-db", "browser_profile", "--no-replace"))
    assert built.exists()

    tables = json.loads(run_cmd("tables", str(built)))
    assert {"profiles", "devices", "bookmarks", "history_visits", "downloads", "open_tabs"} <= set(tables)
