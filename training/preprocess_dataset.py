"""Convert raw mixed annotations into a clean three-class detection dataset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

from training.common import (
    PROJECT_ROOT,
    TARGET_NAMES,
    build_class_mapping,
    format_bbox,
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


@dataclass
class Sample:
    image: Path
    label: Path
    source_split: str
    labels: set[int]


def transformed_labels(label_path: Path, mapping: dict[int, int | None]) -> tuple[list[str], set[int], list[dict]]:
    output: list[str] = []
    labels: set[int] = set()
    findings: list[dict] = []
    if not label_path.is_file():
        findings.append({"reason": "missing_label", "path": str(label_path)})
        return output, labels, findings
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            annotation = parse_annotation_line(line)
        except ValueError as exc:
            findings.append({"reason": "invalid_annotation", "path": str(label_path), "line": line_number, "detail": str(exc)})
            continue
        if annotation.class_id not in mapping:
            findings.append({"reason": "class_out_of_range", "path": str(label_path), "line": line_number, "class_id": annotation.class_id})
            continue
        target_id = mapping[annotation.class_id]
        if target_id is None:
            findings.append({"reason": "no_leak_removed", "path": str(label_path), "line": line_number})
            continue
        issue = validate_bbox(annotation.bbox)
        if issue:
            findings.append({"reason": issue, "path": str(label_path), "line": line_number})
            continue
        area = annotation.bbox[2] * annotation.bbox[3]
        if area > 0.95:
            findings.append({"reason": "suspicious_large_bbox", "path": str(label_path), "line": line_number, "area": area})
        elif area < 0.0001:
            findings.append({"reason": "suspicious_small_bbox", "path": str(label_path), "line": line_number, "area": area})
        output.append(format_bbox(target_id, annotation.bbox))
        labels.add(target_id)
    return output, labels, findings


def stratified_half_split(samples: list[Sample], test_size: int, seed: int) -> tuple[list[Sample], list[Sample]]:
    if test_size <= 0 or test_size >= len(samples):
        raise ValueError("test_size must leave at least one sample in each split")
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    categories = tuple(range(len(TARGET_NAMES))) + (len(TARGET_NAMES),)

    def sample_categories(sample: Sample) -> set[int]:
        return sample.labels or {len(TARGET_NAMES)}

    totals = Counter(category for sample in shuffled for category in sample_categories(sample))
    desired = {category: totals[category] * test_size / len(shuffled) for category in categories}
    selected: list[Sample] = []
    remaining = list(shuffled)
    current: Counter[int] = Counter()
    tie_order = {id(sample): index for index, sample in enumerate(shuffled)}

    while len(selected) < test_size:
        def score(sample: Sample) -> tuple[float, int]:
            cats = sample_categories(sample)
            benefit = sum((desired[c] - current[c]) / max(desired[c], 1.0) for c in cats)
            overshoot = sum(max(0.0, current[c] + 1 - desired[c]) for c in cats)
            rarity = sum(1 / max(totals[c], 1) for c in cats)
            return benefit - 0.25 * overshoot + rarity, -tie_order[id(sample)]

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        current.update(sample_categories(chosen))
    return remaining, selected


def preprocess(source: Path, output: Path, seed: int = 42, force: bool = False) -> dict:
    yaml_path = source / "data.yaml"
    data = read_yaml(yaml_path)
    source_names = parse_names(data)
    mapping = build_class_mapping(source_names)
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"Output is not empty; pass --force to regenerate: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    findings: list[dict] = []
    samples_by_source: dict[str, list[Sample]] = {"train": [], "valid": [], "test": []}
    transformed_cache: dict[Path, list[str]] = {}
    seen_hashes: dict[str, Path] = {}

    for source_split in ("train", "valid", "test"):
        images_dir = resolve_split_images(yaml_path, data, source_split)
        for image_path in iter_images(images_dir):
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                findings.append({"reason": "unreadable_image", "path": str(image_path), "detail": str(exc)})
                continue
            digest = sha256_file(image_path)
            if digest in seen_hashes:
                findings.append({"reason": "duplicate_image", "path": str(image_path), "duplicate_of": str(seen_hashes[digest]), "sha256": digest})
                continue
            seen_hashes[digest] = image_path
            label_path = label_path_for(image_path, images_dir)
            rows, labels, row_findings = transformed_labels(label_path, mapping)
            findings.extend(row_findings)
            transformed_cache[image_path] = rows
            samples_by_source[source_split].append(Sample(image_path, label_path, source_split, labels))

    assignments: dict[str, list[Sample]] = {"train": samples_by_source["train"], "valid": [], "test": []}
    if samples_by_source["test"]:
        assignments["valid"] = samples_by_source["valid"]
        assignments["test"] = samples_by_source["test"]
    else:
        valid_source = samples_by_source["valid"]
        if len(valid_source) != 502:
            raise ValueError(f"Expected 502 source validation images before split, found {len(valid_source)}")
        assignments["valid"], assignments["test"] = stratified_half_split(valid_source, 251, seed)

    split_stats: dict[str, dict] = {}
    for target_split, samples in assignments.items():
        images_output = output / target_split / "images"
        labels_output = output / target_split / "labels"
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        counts: Counter[int] = Counter()
        negatives = 0
        for sample in samples:
            destination_image = images_output / sample.image.name
            destination_label = labels_output / f"{sample.image.stem}.txt"
            shutil.copy2(sample.image, destination_image)
            rows = transformed_cache[sample.image]
            destination_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            if not rows:
                negatives += 1
            for row in rows:
                counts[int(row.split()[0])] += 1
        split_stats[target_split] = {
            "images": len(samples),
            "objects": {TARGET_NAMES[index]: counts[index] for index in range(len(TARGET_NAMES))},
            "negative_images": negatives,
        }

    output_yaml = {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(TARGET_NAMES)},
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output_yaml, handle, allow_unicode=True, sort_keys=False)

    suspicious_path = PROJECT_ROOT / "models/metrics/suspicious_samples.csv"
    suspicious_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for finding in findings for key in finding}) or ["reason"]
    with suspicious_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(findings)

    report = {
        "source": str(source),
        "output": str(output),
        "seed": seed,
        "source_names": source_names,
        "class_mapping": {str(key): value for key, value in mapping.items()},
        "target_names": list(TARGET_NAMES),
        "splits": split_stats,
        "finding_counts": dict(Counter(finding["reason"] for finding in findings)),
        "suspicious_samples_csv": str(suspicious_path),
    }
    write_json(PROJECT_ROOT / "models/metrics/preprocess_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/pipeline_leakage1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = preprocess(args.source.resolve(), args.output.resolve(), args.seed, args.force)
    print(yaml.safe_dump(report["splits"], allow_unicode=True, sort_keys=False))
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
