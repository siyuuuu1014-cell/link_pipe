from training.train_lora_v3 import FORMAL_NAME, SMOKE_NAME, v3_recipe


def test_v3_formal_recipe_is_isolated_and_optimized() -> None:
    recipe = v3_recipe()

    assert FORMAL_NAME == "yolo11s_lora_formal_v3"
    assert FORMAL_NAME != "yolo11s_lora_formal_v2"
    assert recipe["epochs"] == 100
    assert recipe["imgsz"] == 640
    assert recipe["seed"] == 42
    assert recipe["lr0"] == 3e-4
    assert recipe["lora_r"] == 16
    assert recipe["lora_alpha"] == 16
    assert recipe["lora_dropout"] == 0.02
    assert recipe["mosaic"] == 0.5
    assert recipe["close_mosaic"] == 40
    assert recipe["patience"] == 30
    assert recipe["amp"] is False


def test_v3_smoke_recipe_is_short_and_separate() -> None:
    recipe = v3_recipe(smoke=True)

    assert SMOKE_NAME == "yolo11s_lora_v3_smoke"
    assert recipe["epochs"] == 3
    assert recipe["patience"] == 3
    assert recipe["close_mosaic"] == 1
