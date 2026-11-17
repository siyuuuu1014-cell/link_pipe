from yolo_train.config import load_yaml


def test_training_config_has_required_keys() -> None:
    config = load_yaml("configs/train.yaml")
    assert {"model", "data", "epochs", "imgsz", "batch"} <= config.keys()


def test_dataset_config_has_class_names() -> None:
    config = load_yaml("datasets/data.yaml")
    assert config["names"]
