"""Block training unless the processed dataset satisfies all invariants."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from training.common import (
    PROJECT_ROOT,
    TARGET_NAMES,
    iter_images,
    label_path_for,
    parse_annotation_line,
    parse_names,
    read_yaml,
    resolve_split_images,
    sha256_file,
    validate_bbox,
    write_json,
)


def audit(dataset: Path) -> dict:
    yaml_path = dataset / "data.yaml"
    data = read_yaml(yaml_path)
    names = parse_names(data)
    errors: list[str] = []
    if tuple(names.values()) != TARGET_NAMES or tuple(names) != (0, 1, 2):
        errors.append(f"Expected classes {TARGET_NAMES}, found {names}")
    hashes: dict[str, tuple[str, Path]] = {}
    stats: dict[str, dict] = {}

    for split in ("train", "valid", "test"):
        images_dir = resolve_split_images(yaml_path, data, split)
        images = iter_images(images_dir)
        if not images_dir or not images_dir.is_dir():
            errors.append(f"Missing {split} image directory")
            continue
        class_counts: Counter[int] = Counter()
        target_histogram: Counter[int] = Counter()
        negatives = 0
        for image_path in images:
            digest = sha256_file(image_path)
            if digest in hashes:
                previous_split, previous_path = hashes[digest]
                errors.append(f"Duplicate image across/within splits: {previous_split}:{previous_path.name} and {split}:{image_path.name}")
            else:
                hashes[digest] = (split, image_path)
            label_path = label_path_for(image_path, images_dir)
            if not label_path.is_file():
                errors.append(f"Missing label: {label_path}")
                continue
            rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                negatives += 1
            valid_objects = 0
            for line_number, row in enumerate(rows, start=1):
                try:
                    annotation = parse_annotation_line(row)
                    if annotation.source_format != "bbox":
                        raise ValueError("processed labels must contain only bounding boxes")
                    if annotation.class_id not in (0, 1, 2):
                        raise ValueError(f"class ID {annotation.class_id} is not allowed")
                    issue = validate_bbox(annotation.bbox)
                    if issue:
                        raise ValueError(issue)
                except ValueError as exc:
                    errors.append(f"{label_path}:{line_number}: {exc}")
                    continue
                class_counts[annotation.class_id] += 1
                valid_objects += 1
            target_histogram[valid_objects] += 1
        stats[split] = {
            "images": len(images),
            "class_objects": {TARGET_NAMES[index]: class_counts[index] for index in range(3)},
            "negative_images": negatives,
            "objects_per_image": dict(sorted(target_histogram.items())),
        }

    report = {"dataset": str(dataset), "classes": names, "splits": stats, "errors": errors, "passed": not errors}
    write_json(PROJECT_ROOT / "models/metrics/dataset_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/processed")
    args = parser.parse_args()
    report = audit(args.dataset.resolve())
    for split, stats in report["splits"].items():
        print(f"{split}: {stats}")
    if report["errors"]:
        for error in report["errors"][:50]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Dataset audit passed.")


if __name__ == "__main__":
    main()
