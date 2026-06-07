"""
Ghép ảnh baseline (trên) và model của mình (dưới) thành 1 ảnh để dễ so sánh.
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import numpy as np

BASELINE_DIR = Path("vis_model_correct_baseline_wrong")
OUR_DIR      = Path("vis_model_our")
OUTPUT_DIR   = Path("vis_comparison")
LABEL_HEIGHT = 36
JPEG_QUALITY = 88
NUM_WORKERS  = 8

LABEL_COLOR_BASELINE = (50,  50,  220)  # BGR đỏ
LABEL_COLOR_OUR      = (50,  180, 50)   # BGR xanh lá


def add_label(img: np.ndarray, text: str, color: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    banner = np.full((LABEL_HEIGHT, w, 3), color, dtype=np.uint8)
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (LABEL_HEIGHT + th) // 2
    cv2.putText(banner, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([banner, img])


def combine(baseline_path: Path, our_path: Path, out_path: Path):
    img_base = cv2.imread(str(baseline_path))
    img_our  = cv2.imread(str(our_path))

    if img_base is None or img_our is None:
        return

    # Resize nếu khác chiều ngang
    if img_base.shape[1] != img_our.shape[1]:
        w = img_base.shape[1]
        h = int(img_our.shape[0] * w / img_our.shape[1])
        img_our = cv2.resize(img_our, (w, h), interpolation=cv2.INTER_AREA)

    img_base = add_label(img_base, "BASELINE",  LABEL_COLOR_BASELINE)
    img_our  = add_label(img_our,  "OUR MODEL", LABEL_COLOR_OUR)

    combined = np.vstack([img_base, img_our])
    cv2.imwrite(str(out_path.with_suffix(".jpg")), combined,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    baseline_files = {f.name: f for f in BASELINE_DIR.glob("*.png")}
    our_files      = {f.name: f for f in OUR_DIR.glob("*.png")}
    common = sorted(set(baseline_files) & set(our_files))

    print(f"Matched pairs: {len(common)}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                combine,
                baseline_files[fname],
                our_files[fname],
                OUTPUT_DIR / fname,
            ): fname
            for fname in common
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Combining"):
            f.result()

    print(f"Done. Saved {len(common)} images → '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
