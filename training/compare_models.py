"""Compare genuine model metrics according to the required business ranking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.common import PROJECT_ROOT, write_json


def load_metric(path: Path, kind: str) -> dict:
    if not path.is_file():
        return {"kind": kind, "available": False, "reason": f"Metrics file not found: {path}"}
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(kind=kind, available=True)
    return value


def better(left: dict, right: dict) -> dict:
    recall_gap = left["recall"] - right["recall"]
    if abs(recall_gap) >= 0.02:
        return left if recall_gap > 0 else right
    for key in ("map50", "map50_95"):
        if left[key] != right[key]:
            return left if left[key] > right[key] else right
    left_time = left.get("speed_ms", {}).get("inference", float("inf"))
    right_time = right.get("speed_ms", {}).get("inference", float("inf"))
    if left_time != right_time:
        return left if left_time < right_time else right
    return left if left["model_size_mb"] <= right["model_size_mb"] else right


def compare(smoke: bool = False) -> dict:
    prefix = "smoke_" if smoke else ""
    models = [
        load_metric(PROJECT_ROOT / f"models/metrics/{prefix}full_internal_metrics.json", "full"),
        load_metric(PROJECT_ROOT / f"models/metrics/{prefix}lora_internal_metrics.json", "lora"),
    ]
    available = [model for model in models if model["available"]]
    selected = better(available[0], available[1]) if len(available) == 2 else (available[0] if available else None)
    payload = {
        "smoke_only": smoke,
        "models": models,
        "selection_complete": len(available) == 2,
        "selected_kind": selected["kind"] if selected else None,
        "selected_model_path": selected.get("model_path") if selected else None,
        "deployment_eligible": bool(selected and not smoke and len(available) == 2),
    }
    stem = "smoke_model_comparison" if smoke else "model_comparison"
    write_json(PROJECT_ROOT / f"models/metrics/{stem}.json", payload)
    csv_path = PROJECT_ROOT / f"models/metrics/{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "available", "precision", "recall", "map50", "map50_95", "model_size_mb"])
        writer.writeheader()
        for model in models:
            writer.writerow({key: model.get(key) for key in writer.fieldnames})
    if payload["deployment_eligible"]:
        write_json(PROJECT_ROOT / "models/metrics/selected_model.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare(args.smoke), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
