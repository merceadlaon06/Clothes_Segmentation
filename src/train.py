from pathlib import Path

import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = ROOT / "configs" / "data.yaml"
BASE_MODEL_PATH = ROOT / "local_only" / "models" / "yolo26n-seg.pt"


def main():
    ultralytics.checks()
    print(ROOT)

    with DATA_CONFIG_PATH.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    print("Classes:", data_config["names"])
    print("Number of classes:", data_config["nc"])
    print("Train path:", data_config.get("train"))
    print("Val path:", data_config.get("val"))
    print("Test path:", data_config.get("test"))

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Base segmentation model not found: {BASE_MODEL_PATH}")

    model = YOLO(str(BASE_MODEL_PATH))
    results = model.train(
        task="segment",
        data=str(DATA_CONFIG_PATH),
        epochs=50,
        save_period=10,
        batch=4,
        imgsz=640,
        cache=False,
        workers=1,
        patience=20,
        plots=True,
        name="clothes_seg",
    )
    print(results)


if __name__ == "__main__":
    main()
