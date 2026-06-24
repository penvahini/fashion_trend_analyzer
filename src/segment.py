# src/segment.py
"""
Detects garments with YOLO, then segments the whole outfit with SAM: every
detection's mask is unioned into one combined mask per source photo, and
everything outside it is zeroed to black. One output image per source photo
(not per garment) -- this is the original approach, restored after a
per-garment-crop experiment (each detection cropped+masked separately) made
clustering meaningfully worse: a per-garment crop has a much smaller bbox
relative to its mask, so a much larger fraction of each crop ends up being
solid black background. CLIP picked up on "this image is mostly black" as a
shared feature across unrelated garments, collapsing clusters into a few
black-background catch-alls instead of real style groupings. At outfit
scale the black fraction is smaller and the embedding is dominated by the
actual clothing again.
"""
import os
from datetime import date
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import supervision as sv

from observability import track_run

# ---------- Paths (script assumed inside src/segmentation or src/) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
ORIGINAL_IMAGES_DIR = PROJECT_ROOT / "images" / "original_images"
SEGMENTED_IMAGES_DIR = PROJECT_ROOT / "images" / "segmented_images"
WEIGHTS_DIR = PROJECT_ROOT / "models"

# ---------- Utils ----------
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_image_paths(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [str(image_dir / f) for f in os.listdir(image_dir) if Path(f).suffix.lower() in exts]

# ---------- Core ----------
def segment_image(yolo, mask_predictor, image_path, seg_dir: Path):
    yolo_output = yolo.predict(image_path, conf=0.5)

    # Collect boxes [x1,y1,x2,y2,cls]
    r = []
    for result in yolo_output:
        if result.boxes is None or result.boxes.data is None:
            continue
        boxes = result.boxes.data.cpu().numpy().astype(int)
        for b in boxes:
            r.append([b[0], b[1], b[2], b[3], b[5]])

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    mask_combined = np.zeros((h, w), dtype=bool)
    output = np.zeros_like(image)

    mask_predictor.set_image(image)

    for box in r:
        xyxy = np.array(box[:-1])  # drop class id
        masks, scores, _ = mask_predictor.predict(box=xyxy, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue

        # Keep the largest mask among SAM's candidate outputs for this box
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]
        for m in detections.mask:
            mask_combined = np.logical_or(mask_combined, m)

    output[mask_combined] = image[mask_combined]
    save_path = seg_dir / (Path(image_path).stem + "_segmented.png")
    cv2.imwrite(str(save_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved to {save_path}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=date.today().isoformat(),
                     help="Identifier for the scrape run to segment, defaults to today's date (YYYY-MM-DD)")
    args = ap.parse_args()

    image_dir = ORIGINAL_IMAGES_DIR / args.run_id
    seg_dir = SEGMENTED_IMAGES_DIR / args.run_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = WEIGHTS_DIR / "sam_weights.pth"   # place file in models/
    YOLO_WEIGHTS = WEIGHTS_DIR / "yolo_weights.pt"      # place file in models/

    with track_run("segment", run_id=args.run_id) as record:
        yolo = YOLO(str(YOLO_WEIGHTS))
        sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(get_device())
        mask_predictor = SamPredictor(sam)

        image_paths = get_image_paths(image_dir)
        if not image_paths:
            print(f"No images found in {image_dir}. Did you run Step 1 scraper for run-id '{args.run_id}'?")
            record["n_source_images"] = 0
            record["n_garments_segmented"] = 0
            return

        n_segmented = 0
        for image_path in image_paths:
            out_path = seg_dir / (Path(image_path).stem + "_segmented.png")
            if out_path.exists():
                print(f"{out_path} already exists. Skipping.")
                continue
            if segment_image(yolo, mask_predictor, image_path, seg_dir):
                n_segmented += 1

        record["n_source_images"] = len(image_paths)
        record["n_garments_segmented"] = n_segmented

if __name__ == "__main__":
    main()
