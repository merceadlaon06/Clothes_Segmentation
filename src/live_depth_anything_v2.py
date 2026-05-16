import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "assets" / "demo" / "sample_video.mp4"
MODEL_REPO_DIR = ROOT / "local_only" / "external" / "Depth-Anything-V2"

# Use vits for practical live video speed. Change to vitb or vitl for higher quality but slower inference.
ENCODER = "vits"
DISPLAY_SCALE = 0.6
WINDOW_NAME = "Live Depth Anything V2"

AUTO_INSTALL_DEPENDENCIES = True
SHOW_SIDE_BY_SIDE = True


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

ENCODER_TO_MODEL_NAME = {
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


def ensure_depth_anything_repo():
    dpt_file = MODEL_REPO_DIR / "depth_anything_v2" / "dpt.py"
    if dpt_file.exists():
        return

    MODEL_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "git",
        "clone",
        "https://github.com/DepthAnything/Depth-Anything-V2.git",
        str(MODEL_REPO_DIR),
    ])


def load_model():
    ensure_python_package("huggingface_hub")
    ensure_python_package("timm")
    ensure_depth_anything_repo()

    sys.path.insert(0, str(MODEL_REPO_DIR))

    from depth_anything_v2.dpt import DepthAnythingV2
    from huggingface_hub import hf_hub_download

    if ENCODER not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported encoder: {ENCODER}. Use one of: {', '.join(MODEL_CONFIGS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = ENCODER_TO_MODEL_NAME[ENCODER]
    checkpoint_path = hf_hub_download(
        repo_id=f"depth-anything/Depth-Anything-V2-{model_name}",
        filename=f"depth_anything_v2_{ENCODER}.pth",
        repo_type="model",
    )

    model = DepthAnythingV2(**MODEL_CONFIGS[ENCODER])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval(), device


def colorize_depth(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max - depth_min < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    color_map = matplotlib.colormaps.get_cmap("Spectral_r")
    colored = (color_map(normalized)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def resize_for_display(frame):
    if DISPLAY_SCALE == 1:
        return frame

    height, width = frame.shape[:2]
    return cv2.resize(
        frame,
        (max(1, int(width * DISPLAY_SCALE)), max(1, int(height * DISPLAY_SCALE))),
        interpolation=cv2.INTER_AREA,
    )


def main():
    video_path = VIDEO_PATH.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model, device = load_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            depth = model.infer_image(frame)
            depth_frame = colorize_depth(depth)

            if SHOW_SIDE_BY_SIDE:
                depth_frame = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                output = np.hstack((frame, depth_frame))
            else:
                output = depth_frame

            cv2.putText(
                output,
                f"Depth Anything V2 ({ENCODER}) on {device} - press q to quit",
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, resize_for_display(output))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
