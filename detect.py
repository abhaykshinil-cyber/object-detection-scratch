"""
Run inference on an image, a directory of images, or a live webcam.

Usage:
    python detect.py --checkpoint runs/train/best.pt --source photo.jpg
    python detect.py --checkpoint runs/train/best.pt --source images/
    python detect.py --checkpoint runs/train/best.pt --source 0          # webcam
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.augment import letterbox
from model.detector import CustomDetector
from utils.nms import multiclass_nms
from utils.visualize import draw_boxes

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CustomDetector(
        num_classes=ckpt["num_classes"],
        anchors=ckpt["anchors"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    class_names = ckpt.get("class_names", [str(i) for i in range(ckpt["num_classes"])])
    img_size    = ckpt.get("img_size", 640)
    return model, class_names, img_size


def preprocess(image: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    padded = letterbox(image, img_size)
    t = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def run_single(model, image: np.ndarray, img_size: int, device, conf: float, nms_iou: float):
    tensor = preprocess(image, img_size, device)
    t0 = time.perf_counter()
    results = model.predict(tensor, conf_thresh=conf)
    ms = (time.perf_counter() - t0) * 1000

    res = results[0]
    boxes, scores, classes = multiclass_nms(
        res["boxes"], res["scores"], res["classes"], iou_threshold=nms_iou
    )
    return boxes, scores, classes, ms


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def detect_images(source: Path, model, class_names, img_size, device, conf, nms_iou, save_dir: Path):
    img_paths = list(source.glob("*")) if source.is_dir() else [source]
    img_paths = [p for p in img_paths if p.suffix.lower() in IMG_EXTS]

    if not img_paths:
        print(f"No images found at {source}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes, scores, classes, ms = run_single(model, image, img_size, device, conf, nms_iou)

        print(f"{img_path.name}  {len(boxes)} detection(s)  ({ms:.1f} ms)")
        for box, score, cls in zip(
            boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()
        ):
            name = class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))
            coords = [round(float(v), 1) for v in box]
            print(f"  {name:<15s} conf={float(score):.2f}  bbox={coords}")

        # Draw on the letterboxed image — same coordinate space as model predictions
        image_lb = letterbox(image, img_size)
        annotated = draw_boxes(
            image_lb,
            boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy(),
            class_names,
        )
        out = save_dir / img_path.name
        Image.fromarray(annotated).save(out)
        print(f"  Saved → {out}")


def detect_webcam(model, class_names, img_size, device, conf, nms_iou, cam_idx: int = 0):
    try:
        import cv2
    except ImportError:
        print("Install opencv-python:  pip install opencv-python")
        return

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Could not open camera {cam_idx}")
        return

    print("Webcam running - press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes, ms = run_single(model, rgb, img_size, device, conf, nms_iou)

        # Draw on the letterboxed image — same coordinate space as model predictions
        rgb_lb = letterbox(rgb, img_size)
        annotated = draw_boxes(rgb_lb, boxes.cpu().numpy(), scores.cpu().numpy(),
                               classes.cpu().numpy(), class_names)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"{ms:.1f} ms  {len(boxes)} obj",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Custom Detector", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with custom detector")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--source",     required=True,
                        help="Image path, directory, or webcam index (e.g. 0)")
    parser.add_argument("--conf",     type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",      type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--save-dir", default="runs/detect",    help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_model(args.checkpoint, device)

    src = args.source
    if src.isdigit():
        detect_webcam(model, class_names, img_size, device, args.conf, args.iou, int(src))
    else:
        detect_images(Path(src), model, class_names, img_size, device,
                      args.conf, args.iou, Path(args.save_dir))


if __name__ == "__main__":
    main()
