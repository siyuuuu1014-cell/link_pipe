"""Shared, side-effect-free helpers for dataset tooling."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_NAMES = ("crack", "water", "leak")
TARGET_NAME_TO_ID = {name: index for index, name in enumerate(TARGET_NAMES)}


@dataclass(frozen=True)
class ParsedAnnotation:
    class_id: int
    bbox: tuple[float, float, float, float]
    source_format: str


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = project_path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return value


def write_json(path: str | Path, payload: Any) -> None:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_names(data: dict[str, Any]) -> dict[int, str]:
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset class IDs must be integers") from exc
    else:
        raise ValueError("data.yaml must define 'names' as a list or mapping")
    if not names:
        raise ValueError("Dataset contains no class names")
    return dict(sorted(names.items()))


def canonical_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def build_class_mapping(source_names: dict[int, str]) -> dict[int, int | None]:
    aliases = {"crack": "crack", "water": "water", "leak": "leak", "noleak": None}
    mapping: dict[int, int | None] = {}
    for source_id, source_name in source_names.items():
        normalized = canonical_name(source_name)
        if normalized not in aliases:
            raise ValueError(f"Unsupported source class: id={source_id}, name={source_name!r}")
        target_name = aliases[normalized]
        mapping[source_id] = None if target_name is None else TARGET_NAME_TO_ID[target_name]
    return mapping


def resolve_split_images(dataset_yaml: Path, data: dict[str, Any], split: str) -> Path | None:
    key = "val" if split == "valid" else split
    raw = data.get(key)
    if not raw:
        return None
    path = Path(str(raw))
    base = Path(str(data.get("path", dataset_yaml.parent)))
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def label_path_for(image_path: Path, images_dir: Path) -> Path:
    labels_dir = images_dir.parent / "labels"
    return labels_dir / f"{image_path.stem}.txt"


def iter_images(images_dir: Path | None) -> list[Path]:
    if images_dir is None or not images_dir.is_dir():
        return []
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_annotation_line(line: str) -> ParsedAnnotation:
    fields = line.split()
    if len(fields) < 5:
        raise ValueError(f"expected at least 5 fields, got {len(fields)}")
    try:
        class_value = float(fields[0])
        values = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError("annotation contains a non-numeric value") from exc
    if not class_value.is_integer() or class_value < 0:
        raise ValueError(f"invalid class ID: {fields[0]}")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("annotation contains a non-finite coordinate")

    if len(fields) == 5:
        bbox = tuple(values)
        source_format = "bbox"
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        bbox = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        source_format = "polygon"
    else:
        raise ValueError(f"invalid polygon field count: {len(fields)}")
    return ParsedAnnotation(int(class_value), bbox, source_format)


def validate_bbox(bbox: Iterable[float]) -> str | None:
    x_center, y_center, width, height = bbox
    if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
        return "non_finite_bbox"
    if width <= 0 or height <= 0:
        return "non_positive_bbox"
    if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
        return "coordinate_out_of_range"
    return None


def format_bbox(class_id: int, bbox: Iterable[float]) -> str:
    return f"{class_id} " + " ".join(f"{value:.8f}".rstrip("0").rstrip(".") for value in bbox)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
