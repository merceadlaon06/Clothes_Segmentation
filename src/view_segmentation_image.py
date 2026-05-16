import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "local_only" / "models" / "clothes_seg_best.pt"
IMAGE_PATH = ROOT / "assets" / "demo" / "sample_frame.jpg"
PATTERN_PATH = ROOT / "assets" / "patterns" / "animal_print.jpg"
CONFIDENCE_THRESHOLD = 0.25
PATTERN_OPACITY = 0.9


def scaled(value, image_width, image_height, minimum, maximum):
    scale = min(image_width, image_height) / 640
    return int(max(minimum, min(maximum, round(value * scale))))


def fit_pattern_to_box(pattern, box_width, box_height):
    pattern_height, pattern_width = pattern.shape[:2]
    scale = max(box_width / pattern_width, box_height / pattern_height)
    resized_width = max(1, int(round(pattern_width * scale)))
    resized_height = max(1, int(round(pattern_height * scale)))
    resized = cv2.resize(pattern, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    x_start = max(0, (resized_width - box_width) // 2)
    y_start = max(0, (resized_height - box_height) // 2)
    return resized[y_start:y_start + box_height, x_start:x_start + box_width]


def draw_prediction(image_path, model_path, pattern_path, conf):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pattern = cv2.imread(str(pattern_path))
    if pattern is None:
        raise FileNotFoundError(f"Could not read pattern image: {pattern_path}")

    result = YOLO(str(model_path))(str(image_path), conf=conf, verbose=False)[0]
    height, width = image.shape[:2]
    overlay = image.copy()

    font_scale = max(0.45, min(1.8, min(width, height) / 900))
    thickness = scaled(2, width, height, 1, 5)
    box_thickness = scaled(3, width, height, 2, 6)
    pad = scaled(5, width, height, 3, 14)

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        rng = np.random.default_rng(42)
        colors = rng.integers(60, 256, size=(len(result.names), 3), dtype=np.uint8)

        for mask, box, class_id, confidence in zip(masks, boxes, classes, confidences):
            color = tuple(int(channel) for channel in colors[class_id])
            resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5

            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            box_width = x2 - x1
            box_height = y2 - y1
            fitted_pattern = fit_pattern_to_box(pattern, box_width, box_height)
            mask_crop = resized_mask[y1:y2, x1:x2]
            image_crop = overlay[y1:y2, x1:x2]
            blended_crop = (1 - PATTERN_OPACITY) * image_crop + PATTERN_OPACITY * fitted_pattern
            image_crop[mask_crop] = blended_crop[mask_crop].astype(np.uint8)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

            label = f"{result.names[class_id]} {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_y = max(y1, text_h + pad * 2)
            cv2.rectangle(
                overlay,
                (x1, label_y - text_h - pad * 2),
                (x1 + text_w + pad * 2, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (x1 + pad, label_y - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def show_image(image_rgb):
    root = tk.Tk()
    root.title("Clothes Segmentation Result")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    max_width = int(screen_width * 0.9)
    max_height = int(screen_height * 0.85)

    image = Image.fromarray(image_rgb)
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    container = tk.Frame(root, bg="black")
    container.pack(fill="both", expand=True)

    label = tk.Label(container, image=photo, borderwidth=0)
    label.image = photo
    label.pack()

    root.mainloop()


def main():
    image_path = IMAGE_PATH.expanduser().resolve()
    model_path = MODEL_PATH.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pattern_path = PATTERN_PATH.expanduser().resolve()

    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern image not found: {pattern_path}")

    annotated = draw_prediction(image_path, model_path, pattern_path, CONFIDENCE_THRESHOLD)
    show_image(annotated)


if __name__ == "__main__":
    main()
