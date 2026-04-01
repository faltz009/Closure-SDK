from pathlib import Path
import json
import shutil
import tempfile

from closure_dna.demo.coffee_store.closure_dna_demo import build_summary as build_dna_summary
from closure_dna.demo.coffee_store.sqlite_demo import build_summary as build_sqlite_summary


def fresh_path(name: str, suffix: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"cdna_demo_{name}_"))
    return root / suffix

def test_coffee_store_demo_shared_business_results():
    dna_path = fresh_path("dna", "shop.cdb")
    sqlite_path = fresh_path("sqlite", "shop.sqlite3")

    dna = build_dna_summary(dna_path)
    sqlite = build_sqlite_summary(sqlite_path)

    assert dna["counts"] == sqlite["counts"]
    assert dna["checkout"] == sqlite["checkout"]
    assert dna["avg_order_total"] == sqlite["avg_order_total"]
    assert dna["largest_product_sort_head"] == sqlite["largest_product_sort_head"]
    assert dna["top_city_revenue"] == sqlite["top_city_revenue"]
    assert dna["tokyo_customers"] == sqlite["tokyo_customers"]
    assert dna["top_paid_orders"] == sqlite["top_paid_orders"]
    assert dna["tokyo_order_join"] == sqlite["tokyo_order_join"]

    assert dna["table_check"] >= 0.0
    assert dna["genome"]["depth"] >= 1
    assert dna["genome"]["codons"] >= 1
    assert len(dna["search_hits"]) == 3
    assert dna["search_hits"][0]["drift"] <= dna["search_hits"][1]["drift"] <= dna["search_hits"][2]["drift"]
    assert dna["transaction_demo"]["checkout_committed"] is True
    assert dna["foreign_key_demo"]["rejected_missing_customer"] is True
    assert dna["compaction_demo"]["orders"] == 1

    shutil.rmtree(dna_path.parent)
    shutil.rmtree(sqlite_path.parent)

def test_demo_data_manifests_are_present_and_coherent():
    root = Path(__file__).resolve().parents[1] / "demo_data"
    index = json.loads((root / "datasets.json").read_text(encoding="utf-8"))
    manifests = {}
    for entry in index["datasets"]:
        manifest_path = root / entry["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[manifest["name"]] = manifest
        assert manifest["table_order"]
        assert manifest["schemas"]
        assert manifest["files"]
        for table_name in manifest["table_order"]:
            table_path = manifest_path.parent / manifest["files"][table_name]
            assert table_path.exists()
            assert manifest["counts"][table_name] > 0

    assert manifests["browser_profile"]["counts"]["history_visits"] >= 40000
    assert manifests["chat_app"]["counts"]["messages"] >= 10000
    assert manifests["music_streaming"]["counts"]["listening_events"] >= 100000
    assert manifests["music_streaming"]["counts"]["listener_profiles"] >= 2000
