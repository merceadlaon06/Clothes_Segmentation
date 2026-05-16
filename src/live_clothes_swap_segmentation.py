import importlib.util
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_MODEL_PATH = ROOT / "local_only" / "models" / "clothes_seg_best.pt"
VIDEO_PATH = ROOT / "assets" / "demo" / "sample_video.mp4"
DEPTH_REPO_DIR = ROOT / "local_only" / "external" / "Depth-Anything-V2"

CLASS_PATTERN_PATHS = {
    "trousers": ROOT / "assets" / "patterns" / "chatgpt_pattern.png",
    "long sleeve top": ROOT / "assets" / "patterns" / "blue_pattern.jpg",
}

# vits is the practical choice for live video. vitb/vitl improve depth quality but are much slower.
DEPTH_ENCODER = "vits"
SEGMENT_CONFIDENCE = 0.35
PATTERN_OPACITY = 0.8
SHADING_STRENGTH = 0.80
DEPTH_STRENGTH = 0.50
DISPLAY_SCALE = 0.65
WINDOW_NAME = "Live Clothes Pattern With Depth Shading"
AUTO_INSTALL_DEPENDENCIES = True
MASK_SMOOTHING_ALPHA = 0.65
SHADING_SMOOTHING_ALPHA = 0.72
MASK_KEEP_FRAMES = 4
MASK_THRESHOLD = 0.48
SHOW_BOXES_AND_LABELS = True
MAX_SCREEN_FRACTION = 0.9

toggle_button_rect = None
class_button_rects = {}
runtime_patterns = {}
tk_root = None
window_initialized = False


DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

DEPTH_ENCODER_TO_NAME = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
}


def run_command(command):
    subprocess.check_call(command, cwd=ROOT)


def ensure_python_package(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is not None:
        return

    if not AUTO_INSTALL_DEPENDENCIES:
        package = pip_name or import_name
        raise ModuleNotFoundError(
            f"Missing dependency '{import_name}'. Install it with: {sys.executable} -m pip install {package}"
        )

    run_command([sys.executable, "-m", "pip", "install", pip_name or import_name])


def ensure_depth_repo():
    dpt_file = DEPTH_REPO_DIR / "depth_anything_v2" / "dpt.py"
    if dpt_file.exists():
        return

    DEPTH_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "git",
        "clone",
        "https://github.com/DepthAnything/Depth-Anything-V2.git",
        str(DEPTH_REPO_DIR),
    ])


def load_depth_model():
    ensure_python_package("huggingface_hub")
    ensure_python_package("timm")
    ensure_depth_repo()

    sys.path.insert(0, str(DEPTH_REPO_DIR))

    from depth_anything_v2.dpt import DepthAnythingV2
    from huggingface_hub import hf_hub_download

    if DEPTH_ENCODER not in DEPTH_MODEL_CONFIGS:
        raise ValueError(f"Unsupported depth encoder: {DEPTH_ENCODER}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = DEPTH_ENCODER_TO_NAME[DEPTH_ENCODER]
    checkpoint_path = hf_hub_download(
        repo_id=f"depth-anything/Depth-Anything-V2-{model_name}",
        filename=f"depth_anything_v2_{DEPTH_ENCODER}.pth",
        repo_type="model",
    )

    model = DepthAnythingV2(**DEPTH_MODEL_CONFIGS[DEPTH_ENCODER])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval(), device


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


def cleanup_mask(mask):
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_uint8 > 127


class TemporalSmoother:
    def __init__(self):
        self.masks = {}
        self.boxes = {}
        self.missing_frames = {}
        self.shading_map = None

    def smooth_shading(self, shading_map):
        if self.shading_map is None or self.shading_map.shape != shading_map.shape:
            self.shading_map = shading_map
        else:
            self.shading_map = (
                SHADING_SMOOTHING_ALPHA * self.shading_map
                + (1 - SHADING_SMOOTHING_ALPHA) * shading_map
            ).astype(np.float32)
        return self.shading_map

    def smooth_detections(self, detections, frame_shape):
        height, width = frame_shape[:2]
        current_keys = set()
        smoothed = []

        for detection in detections:
            key = detection["class_name"]
            current_keys.add(key)
            mask = detection["mask"].astype(np.float32)
            box = detection["box"].astype(np.float32)

            previous_mask = self.masks.get(key)
            previous_box = self.boxes.get(key)
            if previous_mask is not None and previous_mask.shape == mask.shape:
                mask = MASK_SMOOTHING_ALPHA * previous_mask + (1 - MASK_SMOOTHING_ALPHA) * mask
            if previous_box is not None:
                box = MASK_SMOOTHING_ALPHA * previous_box + (1 - MASK_SMOOTHING_ALPHA) * box

            self.masks[key] = mask
            self.boxes[key] = box
            self.missing_frames[key] = 0

            smoothed.append({**detection, "mask": cleanup_mask(mask > MASK_THRESHOLD), "box": box.astype(int)})

        for key in list(self.masks):
            if key in current_keys:
                continue

            self.missing_frames[key] = self.missing_frames.get(key, 0) + 1
            if self.missing_frames[key] > MASK_KEEP_FRAMES:
                self.masks.pop(key, None)
                self.boxes.pop(key, None)
                self.missing_frames.pop(key, None)
                continue

            faded_mask = self.masks[key] * MASK_SMOOTHING_ALPHA
            self.masks[key] = faded_mask
            if faded_mask.max() > MASK_THRESHOLD:
                smoothed.append({
                    "class_name": key,
                    "class_id": -1,
                    "confidence": 0.0,
                    "mask": cleanup_mask(faded_mask > MASK_THRESHOLD),
                    "box": self.boxes[key].astype(int),
                    "color": (120, 120, 120),
                })

        return [clip_detection(d, width, height) for d in smoothed]


def clip_detection(detection, width, height):
    x1, y1, x2, y2 = detection["box"]
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return {**detection, "box": np.array([x1, y1, x2, y2], dtype=np.int32)}


def normalize_map(values):
    low, high = np.percentile(values, (2, 98))
    if high - low < 1e-6:
        return np.ones_like(values, dtype=np.float32) * 0.5

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def build_shading_map(frame, depth):
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    luminance = cv2.GaussianBlur(luminance, (0, 0), sigmaX=3, sigmaY=3)

    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    depth = normalize_map(depth.astype(np.float32))
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=5, sigmaY=5)

    # Luminance preserves shadows/folds; depth adds large-scale body/clothing curvature.
    luminance_factor = 0.55 + SHADING_STRENGTH * luminance
    depth_factor = 1.0 + DEPTH_STRENGTH * (depth - 0.5)
    return np.clip(luminance_factor * depth_factor, 0.35, 1.45).astype(np.float32)


def extract_pattern_detections(frame, segmentation_result, class_filter=None):
    height, width = frame.shape[:2]

    if segmentation_result.masks is None or segmentation_result.boxes is None:
        return []

    masks = segmentation_result.masks.data.cpu().numpy()
    boxes = segmentation_result.boxes.xyxy.cpu().numpy()
    classes = segmentation_result.boxes.cls.cpu().numpy().astype(int)
    confidences = segmentation_result.boxes.conf.cpu().numpy()

    rng = np.random.default_rng(42)
    colors = rng.integers(60, 256, size=(len(segmentation_result.names), 3), dtype=np.uint8)

    detections = []
    for mask, box, class_id, confidence in zip(masks, boxes, classes, confidences):
        class_name = segmentation_result.names[class_id]
        if class_filter is not None and class_name not in class_filter:
            continue

        color = tuple(int(channel) for channel in colors[class_id])
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
        detections.append({
            "class_name": class_name,
            "class_id": class_id,
            "confidence": float(confidence),
            "mask": cleanup_mask(resized_mask),
            "box": np.array(box, dtype=np.float32),
            "color": color,
        })

    return detections


def apply_pattern_to_masks(frame, detections, patterns, shading_map, show_boxes_and_labels):
    height, width = frame.shape[:2]
    output = frame.copy()

    font_scale = max(0.45, min(1.8, min(width, height) / 900))
    thickness = scaled(2, width, height, 1, 5)
    box_thickness = scaled(3, width, height, 2, 6)
    pad = scaled(5, width, height, 3, 14)

    for detection in detections:
        pattern = patterns.get(detection["class_name"])
        if pattern is None:
            continue

        color = detection["color"]
        x1, y1, x2, y2 = detection["box"]

        box_width = x2 - x1
        box_height = y2 - y1
        fitted_pattern = fit_pattern_to_box(pattern, box_width, box_height).astype(np.float32)
        local_shading = shading_map[y1:y2, x1:x2, None]
        shaded_pattern = np.clip(fitted_pattern * local_shading, 0, 255)

        mask_crop = detection["mask"][y1:y2, x1:x2]
        frame_crop = output[y1:y2, x1:x2].astype(np.float32)
        blended_crop = (1 - PATTERN_OPACITY) * frame_crop + PATTERN_OPACITY * shaded_pattern
        output[y1:y2, x1:x2][mask_crop] = blended_crop[mask_crop].astype(np.uint8)

        if not show_boxes_and_labels:
            continue

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        label = detection["class_name"]
        if detection["confidence"] > 0:
            label = f"{label} {detection['confidence']:.2f}"
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
    height, width = frame.shape[:2]
    screen_width = 1920
    screen_height = 1080

    scale = min(
        (screen_width * MAX_SCREEN_FRACTION) / width,
        (screen_height * MAX_SCREEN_FRACTION) / height,
    )
    scale *= DISPLAY_SCALE

    if scale >= 1:
        return frame

    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def load_patterns():
    patterns = {}
    for class_name, pattern_path in CLASS_PATTERN_PATHS.items():
        resolved_path = pattern_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Pattern image for '{class_name}' not found: {resolved_path}")

        pattern = cv2.imread(str(resolved_path))
        if pattern is None:
            raise FileNotFoundError(f"Could not read pattern image for '{class_name}': {resolved_path}")
        patterns[class_name] = pattern

    return patterns


def get_file_dialog_root():
    global tk_root

    if tk_root is None:
        tk_root = tk.Tk()
        tk_root.withdraw()
        tk_root.attributes("-topmost", True)
    return tk_root


def pick_pattern_image():
    root = get_file_dialog_root()
    root.update()
    file_path = filedialog.askopenfilename(
        title="Choose pattern image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")],
    )
    root.update()
    if not file_path:
        return None

    pattern = cv2.imread(file_path)
    if pattern is None:
        return None
    return file_path, pattern


def draw_toggle_button(frame, show_boxes_and_labels):
    global toggle_button_rect

    text = "Hide boxes" if show_boxes_and_labels else "Show boxes"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    pad_x = 16
    pad_y = 10
    margin = 18
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x2 = frame.shape[1] - margin
    y1 = margin
    x1 = x2 - text_w - pad_x * 2
    y2 = y1 + text_h + baseline + pad_y * 2

    toggle_button_rect = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        text,
        (x1 + pad_x, y2 - pad_y - baseline),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def class_button_colors(class_name):
    seed = sum(ord(char) for char in class_name)
    blue = 80 + (seed * 37) % 140
    green = 80 + (seed * 53) % 140
    red = 80 + (seed * 71) % 140
    fill_color = (blue, green, red)
    border_color = tuple(min(255, channel + 35) for channel in fill_color)
    return fill_color, border_color


def compose_display_frame(video_frame, detected_classes, show_boxes_and_labels, assigned_patterns):
    global class_button_rects

    display_video = resize_for_display(video_frame)
    video_height, video_width = display_video.shape[:2]

    sidebar_width = max(210, min(320, int(video_width * 0.28)))
    canvas_width = sidebar_width + video_width
    canvas = np.zeros((video_height, canvas_width, 3), dtype=np.uint8)
    canvas[:, :sidebar_width] = (28, 28, 28)
    canvas[:, sidebar_width:] = display_video

    title_scale = max(0.55, min(1.0, video_height / 900))
    title_thickness = max(1, int(round(title_scale * 2)))
    button_scale = max(0.5, min(0.9, video_height / 1000))
    button_thickness = max(1, int(round(button_scale * 2)))
    header_margin = max(14, int(video_height * 0.02))
    button_gap = max(8, int(video_height * 0.012))
    button_height = max(34, int(video_height * 0.065))
    button_pad_x = max(10, int(sidebar_width * 0.05))

    cv2.putText(
        canvas,
        "Detected Classes",
        (header_margin, header_margin + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        title_thickness,
        cv2.LINE_AA,
    )

    class_button_rects = {}
    y = header_margin + 42
    button_x1 = header_margin
    button_x2 = sidebar_width - header_margin

    for class_name in detected_classes:
        y2 = min(video_height - header_margin, y + button_height)
        is_assigned = class_name in assigned_patterns
        fill_color, border_color = class_button_colors(class_name)
        if not is_assigned:
            fill_color = tuple(max(35, int(channel * 0.55)) for channel in fill_color)
            border_color = tuple(max(120, int(channel * 0.8)) for channel in border_color)
        cv2.rectangle(canvas, (button_x1, y), (button_x2, y2), fill_color, -1)
        cv2.rectangle(canvas, (button_x1, y), (button_x2, y2), border_color, 2)

        label = class_name if not is_assigned else f"{class_name} *"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, button_scale, button_thickness
        )
        text_x = button_x1 + button_pad_x
        text_y = y + max(text_h + baseline, (button_height + text_h) // 2)
        if text_x + text_w > button_x2 - button_pad_x:
            while text_x + text_w > button_x2 - button_pad_x and len(label) > 4:
                label = label[:-4] + "..."
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, button_scale, button_thickness
                )
            text_y = y + max(text_h + baseline, (button_height + text_h) // 2)

        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            button_scale,
            (255, 255, 255),
            button_thickness,
            cv2.LINE_AA,
        )
        class_button_rects[class_name] = (button_x1, y, button_x2, y2)
        y = y2 + button_gap
        if y + button_height > video_height - header_margin:
            break

    draw_toggle_button(canvas, show_boxes_and_labels)
    return canvas


def on_mouse(event, x, y, flags, param):
    global SHOW_BOXES_AND_LABELS

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if toggle_button_rect is not None:
        x1, y1, x2, y2 = toggle_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            SHOW_BOXES_AND_LABELS = not SHOW_BOXES_AND_LABELS
            return

    for class_name, (x1, y1, x2, y2) in class_button_rects.items():
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue

        selected = pick_pattern_image()
        if selected is None:
            return

        _, pattern = selected
        runtime_patterns[class_name] = pattern
        return


def main():
    global window_initialized, runtime_patterns

    segmentation_model_path = SEGMENTATION_MODEL_PATH.expanduser().resolve()
    video_path = VIDEO_PATH.expanduser().resolve()

    if not segmentation_model_path.exists():
        raise FileNotFoundError(f"Segmentation model not found: {segmentation_model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    segmentation_model = YOLO(str(segmentation_model_path))
    depth_model, depth_device = load_depth_model()
    default_patterns = load_patterns()
    runtime_patterns = {}
    smoother = TemporalSmoother()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    window_flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        window_flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(WINDOW_NAME, window_flags)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            segmentation_result = segmentation_model(frame, conf=SEGMENT_CONFIDENCE, verbose=False)[0]
            depth = depth_model.infer_image(frame)
            shading_map = smoother.smooth_shading(build_shading_map(frame, depth))
            active_patterns = {**default_patterns, **runtime_patterns}
            detections = extract_pattern_detections(frame, segmentation_result)
            detections = smoother.smooth_detections(detections, frame.shape)
            output = apply_pattern_to_masks(
                frame,
                detections,
                active_patterns,
                shading_map,
                SHOW_BOXES_AND_LABELS,
            )

            cv2.putText(
                output,
                f"YOLO masks + Depth Anything V2 ({DEPTH_ENCODER} on {depth_device}) - press q to quit",
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            detected_classes = sorted({detection["class_name"] for detection in detections})
            display_frame = compose_display_frame(
                output,
                detected_classes,
                SHOW_BOXES_AND_LABELS,
                active_patterns,
            )
            if not window_initialized:
                cv2.resizeWindow(WINDOW_NAME, display_frame.shape[1], display_frame.shape[0])
                window_initialized = True
            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
