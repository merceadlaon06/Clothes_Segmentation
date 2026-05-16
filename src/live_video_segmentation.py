from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "local_only" / "models" / "clothes_seg_best.pt"
VIDEO_PATH = ROOT / "assets" / "demo" / "sample_video.mp4"
PATTERN_PATH = ROOT / "assets" / "patterns" / "animal_print.jpg"

CONFIDENCE_THRESHOLD = 0.25
PATTERN_OPACITY = 0.9
DISPLAY_SCALE = 0.75
WINDOW_NAME = "Live Clothes Segmentation"


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


def draw_frame(frame, result, pattern):
    height, width = frame.shape[:2]
    output = frame.copy()

    font_scale = max(0.45, min(1.8, min(width, height) / 900))
    thickness = scaled(2, width, height, 1, 5)
    box_thickness = scaled(3, width, height, 2, 6)
    pad = scaled(5, width, height, 3, 14)

    if result.masks is None or result.boxes is None:
        return output

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
        frame_crop = output[y1:y2, x1:x2]
        blended_crop = (1 - PATTERN_OPACITY) * frame_crop + PATTERN_OPACITY * fitted_pattern
        frame_crop[mask_crop] = blended_crop[mask_crop].astype(np.uint8)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        label = f"{result.names[class_id]} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1, text_h + pad * 2)
        cv2.rectangle(
            output,
            (x1, label_y - text_h - pad * 2),
            (x1 + text_w + pad * 2, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + pad, label_y - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def resize_for_display(frame):
    if DISPLAY_SCALE == 1:
        return frame

    height, width = frame.shape[:2]
    return cv2.resize(
        frame,
        (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
        interpolation=cv2.INTER_AREA,
    )


def main():
    model_path = MODEL_PATH.resolve()
    video_path = VIDEO_PATH.resolve()
    pattern_path = PATTERN_PATH.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern image not found: {pattern_path}")

    model = YOLO(str(model_path))
    pattern = cv2.imread(str(pattern_path))
    if pattern is None:
        raise FileNotFoundError(f"Could not read pattern image: {pattern_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        output = draw_frame(frame, result, pattern)
        cv2.imshow(WINDOW_NAME, resize_for_display(output))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
