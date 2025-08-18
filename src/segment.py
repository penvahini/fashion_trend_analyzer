import os
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import supervision as sv

# ---------- Paths (script assumed inside src/segmentation or src/) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
IMAGE_DIR = PROJECT_ROOT / "images" / "original_images"
SEG_DIR = PROJECT_ROOT / "images" / "segmented_images"
WEIGHTS_DIR = PROJECT_ROOT / "models"
SEG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
def convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return x1, y1, w, h

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_image_paths(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [str(image_dir / f) for f in os.listdir(image_dir) if Path(f).suffix.lower() in exts]

# ---------- Core ----------
def segment_image(yolo, mask_predictor, image_path):
    # YOLO detection
    yolo_output = yolo.predict(image_path, conf=0.5)

    # Collect boxes [x1,y1,x2,y2,cls]
    r = []
    for result in yolo_output:
        if result.boxes is None or result.boxes.data is None:
            continue
        boxes = result.boxes.data.cpu().numpy().astype(int)
        for b in boxes:
            # keep original coords + class idx (b[5]) like your code
            r.append([b[0], b[1], b[2], b[3], b[5]])

    # Load image (RGB)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare masks
    h, w = image.shape[:2]
    mask_combined = np.zeros((h, w), dtype=bool)
    output = np.zeros_like(image)

    # Set predictor image ONCE per image
    mask_predictor.set_image(image)

    for box in r:
        xyxy = np.array(box[:-1])  # drop class id
        # SAM predict
        masks, scores, _ = mask_predictor.predict(box=xyxy, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue

        # Keep largest mask from SAM outputs
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]
        for m in detections.mask:
            mask_combined = np.logical_or(mask_combined, m)

    # Write segmented output
    output[mask_combined] = image[mask_combined]
    save_path = SEG_DIR / (Path(image_path).stem + "_segmented.png")
    cv2.imwrite(str(save_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved to {save_path}")

def main():
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = WEIGHTS_DIR / "sam_weights.pth"   # place file in models/
    YOLO_WEIGHTS = WEIGHTS_DIR / "yolo_weights.pt"      # place file in models/

    # Load models
    yolo = YOLO(str(YOLO_WEIGHTS))
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(get_device())
    mask_predictor = SamPredictor(sam)

    # Process images
    image_paths = get_image_paths(IMAGE_DIR)
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Did you run Step 1 scraper?")
        return

    for image_path in image_paths:
        out_path = SEG_DIR / (Path(image_path).stem + "_segmented.png")
        if out_path.exists():
            print(f"{out_path} already exists. Skipping.")
            continue
        segment_image(yolo, mask_predictor, image_path)

if __name__ == "__main__":
    main()
