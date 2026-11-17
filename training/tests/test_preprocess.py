from pathlib import Path

from training.common import build_class_mapping, parse_annotation_line
from training.preprocess_dataset import Sample, stratified_half_split, transformed_labels


def test_dynamic_class_order_and_no_leak_removal(tmp_path: Path) -> None:
    mapping = build_class_mapping({0: "Water", 1: "NO_LEAK", 2: "crack", 3: "leak"})
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.8 0.8\n", encoding="utf-8")
    rows, labels, findings = transformed_labels(label, mapping)
    assert rows == ["1 0.5 0.5 0.2 0.2"]
    assert labels == {1}
    assert any(item["reason"] == "no_leak_removed" for item in findings)


def test_polygon_is_converted_to_bbox() -> None:
    annotation = parse_annotation_line("0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8")
    assert annotation.source_format == "polygon"
    assert annotation.bbox == (0.3, 0.5, 0.4, 0.6000000000000001)


def test_stratified_split_is_deterministic() -> None:
    samples = [Sample(Path(f"{index}.jpg"), Path(f"{index}.txt"), "valid", {index % 3}) for index in range(20)]
    first_valid, first_test = stratified_half_split(samples, 10, 42)
    second_valid, second_test = stratified_half_split(samples, 10, 42)
    assert [sample.image for sample in first_test] == [sample.image for sample in second_test]
    assert len(first_valid) == len(first_test) == 10
