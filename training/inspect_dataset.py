"""Inspect the untouched source dataset and emit a machine-readable report."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from training.common import (
    PROJECT_ROOT,
    iter_images,
    label_path_for,
    parse_annotation_line,
    parse_names,
    read_yaml,
    resolve_split_images,
    validate_bbox,
    write_json,
)


def inspect_dataset(dataset: Path) -> dict:
    yaml_path = dataset / "data.yaml"
    data = read_yaml(yaml_path)
    names = parse_names(data)
    report: dict = {
        "dataset": str(dataset),
        "class_names": names,
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    total_formats: Counter[str] = Counter()

    for split in ("train", "valid", "test"):
        images_dir = resolve_split_images(yaml_path, data, split)
        images = iter_images(images_dir)
        class_counts: Counter[str] = Counter()
        missing_labels = 0
        empty_labels = 0
        invalid_rows = 0
        if images_dir is None:
            report["splits"][split] = {"present": False, "images": 0}
            continue
        if not images_dir.is_dir():
            report["errors"].append(f"{split} image directory does not exist: {images_dir}")
        for image_path in images:
            label_path = label_path_for(image_path, images_dir)
            if not label_path.is_file():
                missing_labels += 1
                continue
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                try:
                    annotation = parse_annotation_line(line)
                    if annotation.class_id not in names:
                        raise ValueError(f"class ID {annotation.class_id} is out of range")
                    bbox_issue = validate_bbox(annotation.bbox)
                    if bbox_issue:
                        raise ValueError(bbox_issue)
                except ValueError as exc:
                    invalid_rows += 1
                    report["errors"].append(f"{label_path}:{line_number}: {exc}")
                    continue
                class_counts[names[annotation.class_id]] += 1
                total_formats[annotation.source_format] += 1
        report["splits"][split] = {
            "present": images_dir.is_dir(),
            "images": len(images),
            "class_objects": dict(class_counts),
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "invalid_rows": invalid_rows,
        }
    report["annotation_formats"] = dict(total_formats)
    if total_formats.get("polygon"):
        report["warnings"].append("Polygon rows will be converted to axis-aligned bounding boxes.")
    if not report["splits"].get("test", {}).get("present"):
        report["warnings"].append("No source test split; valid will be stratified into valid/test.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/pipeline_leakage1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/metrics/dataset_inspection.json")
    args = parser.parse_args()
    report = inspect_dataset(args.dataset.resolve())
    write_json(args.output, report)
    print(f"Classes: {report['class_names']}")
    for split, stats in report["splits"].items():
        print(f"{split}: {stats}")
    if report["errors"]:
        raise SystemExit(f"Dataset inspection found {len(report['errors'])} error(s).")
    print(f"Inspection report: {args.output}")


if __name__ == "__main__":
    main()
