"""
T2.1 — Dataset preparation with sensor emulation.
Converts LLVIP VOC XML annotations → YOLO txt format.
Uses infrared (thermal) images only — person class → class 0.

SENSOR EMULATION (matches physical Lepton 3.5 / 160x120, f/1.1):
  - Downscales full LLVIP frames (1280x1024) → 160x120 (sensor resolution)
  - Applies Gaussian blur (sigma=0.8) to simulate Lepton optics/diffusion
  - Adds low-level Gaussian readout noise (sigma=2 dn, uint8-clamped)
  - Filters out annotations where person height < 4px at sensor resolution
    (below reliable detection threshold at real sensor scale)

Splits: train (from LLVIP train), val (10% of train), test (LLVIP test).
Outputs:
  data/processed/llvip/images/{train,val,test}/   (160x120 sensor-emulated)
  data/processed/llvip/labels/{train,val,test}/
  configs/llvip.yaml  (YOLO dataset descriptor, imgsz=160)
  reports/dataset_prep.md
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ── paths ------------------------------------------------------------------ #
ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw" / "llvip" / "LLVIP"
PROC      = ROOT / "data" / "processed" / "llvip"
ANN_DIR   = RAW / "Annotations"
IR_TRAIN  = RAW / "infrared" / "train"
IR_TEST   = RAW / "infrared" / "test"
CONFIGS   = ROOT / "configs"
REPORTS   = ROOT / "reports"

VAL_FRAC  = 0.10
SEED      = 42

# ── sensor emulation parameters ------------------------------------------- #
SENSOR_W  = 160
SENSOR_H  = 120
BLUR_SIGMA        = 0.8   # Lepton optics spread (pixels at sensor res)
NOISE_SIGMA_DN    = 2.0   # read noise in digital numbers (uint8)
MIN_PERSON_PX_H   = 4     # min person height in pixels at sensor res to keep


def _emulate_sensor(img_gray: np.ndarray) -> np.ndarray:
    """
    Downscale full-res thermal frame to sensor resolution and add
    optical blur + readout noise matching Lepton 3.5 characteristics.
    Input: HxW uint8 grayscale at LLVIP native resolution (1280x1024).
    Output: SENSOR_H x SENSOR_W uint8, sensor-realistic.
    """
    # 1. Downscale with INTER_AREA (correct for minification)
    small = cv2.resize(img_gray, (SENSOR_W, SENSOR_H), interpolation=cv2.INTER_AREA)

    # 2. Optical blur (Lepton 3.5 has ~1.6px Airy radius, model as Gaussian)
    if BLUR_SIGMA > 0:
        small = cv2.GaussianBlur(small, (0, 0), BLUR_SIGMA)

    # 3. Readout noise
    if NOISE_SIGMA_DN > 0:
        noise = np.random.normal(0.0, NOISE_SIGMA_DN, small.shape)
        small = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return small


def voc_xml_to_yolo_sensor(
    xml_path: Path,
    src_w: int,
    src_h: int,
) -> list[str]:
    """
    Parse LLVIP VOC XML, scale boxes to sensor resolution, filter tiny boxes.
    Returns YOLO-format lines (cx cy w h normalised to sensor dims).
    """
    tree = ET.parse(xml_path)
    root_el = tree.getroot()
    lines = []
    for obj in root_el.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name != "person":
            continue
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Scale box to sensor resolution
        scale_x = SENSOR_W / src_w
        scale_y = SENSOR_H / src_h
        sx1 = xmin * scale_x
        sy1 = ymin * scale_y
        sx2 = xmax * scale_x
        sy2 = ymax * scale_y

        # Filter: drop boxes too small for reliable detection
        box_h_px = sy2 - sy1
        box_w_px = sx2 - sx1
        if box_h_px < MIN_PERSON_PX_H or box_w_px < 2:
            continue

        # Convert to YOLO normalised format
        cx = ((sx1 + sx2) / 2) / SENSOR_W
        cy = ((sy1 + sy2) / 2) / SENSOR_H
        w  = (sx2 - sx1) / SENSOR_W
        h  = (sy2 - sy1) / SENSOR_H
        cx, cy, w, h = [max(0.0, min(1.0, v)) for v in (cx, cy, w, h)]
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def process_split(
    stems: list[str],
    src_img_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    ann_dir: Path,
    stats: Counter,
    rng: np.random.Generator,
) -> tuple[int, list[float]]:
    """
    Apply sensor emulation and write images + YOLO labels.
    Returns (num_copied, list_of_person_heights_px_at_sensor_res).
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    heights_px: list[float] = []

    for stem in stems:
        img_src = src_img_dir / f"{stem}.jpg"
        if not img_src.exists():
            img_src = src_img_dir / f"{stem}.png"
        if not img_src.exists():
            stats["missing_img"] += 1
            continue

        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue

        # Read source image
        img = cv2.imread(str(img_src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["unreadable"] += 1
            continue
        src_h, src_w = img.shape[:2]

        # Get source annotation size (prefer XML <size> tag for speed)
        tree = ET.parse(xml_path)
        size_el = tree.getroot().find("size")
        if size_el is not None:
            src_w = int(size_el.find("width").text)
            src_h = int(size_el.find("height").text)

        # Convert boxes (filter tiny) at sensor resolution
        lines = voc_xml_to_yolo_sensor(xml_path, src_w, src_h)
        if not lines:
            stats["no_person_after_filter"] += 1
            continue

        # Collect per-person heights for the report
        for line in lines:
            parts = line.split()
            h_norm = float(parts[4])
            heights_px.append(h_norm * SENSOR_H)

        stats["persons"] += len(lines)

        # Sensor-emulate and save image
        sensor_img = _emulate_sensor(img)
        out_img_path = dst_img_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), sensor_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write label
        (dst_lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        copied += 1

    stats["images"] += copied
    return copied, heights_px


def _bucket_stats(heights: list[float]) -> dict[str, int]:
    """Bin person heights (px at sensor res) into distance buckets."""
    buckets = {"<4px (filtered)": 0, "4-8px (far)": 0,
               "8-15px (mid)": 0, "15-30px (close)": 0, ">30px (near)": 0}
    for h in heights:
        if h < 4:
            buckets["<4px (filtered)"] += 1
        elif h < 8:
            buckets["4-8px (far)"] += 1
        elif h < 15:
            buckets["8-15px (mid)"] += 1
        elif h < 30:
            buckets["15-30px (close)"] += 1
        else:
            buckets[">30px (near)"] += 1
    return buckets


def main() -> None:
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Collect all stems
    train_stems = sorted(
        [p.stem for p in IR_TRAIN.glob("*.jpg")] +
        [p.stem for p in IR_TRAIN.glob("*.png")]
    )
    test_stems = sorted(
        [p.stem for p in IR_TEST.glob("*.jpg")] +
        [p.stem for p in IR_TEST.glob("*.png")]
    )

    print(f"Raw train images: {len(train_stems)}")
    print(f"Raw test  images: {len(test_stems)}")

    random.shuffle(train_stems)
    n_val = max(1, int(len(train_stems) * VAL_FRAC))
    val_stems   = train_stems[:n_val]
    train_stems = train_stems[n_val:]
    print(f"Split  -> train={len(train_stems)}  val={n_val}  test={len(test_stems)}")

    stats: Counter = Counter()

    n_train, h_train = process_split(train_stems, IR_TRAIN, PROC/"images"/"train",
                                     PROC/"labels"/"train", ANN_DIR, stats, rng)
    n_val_c, h_val   = process_split(val_stems,   IR_TRAIN, PROC/"images"/"val",
                                     PROC/"labels"/"val",   ANN_DIR, stats, rng)
    n_test,  h_test  = process_split(test_stems,  IR_TEST,  PROC/"images"/"test",
                                     PROC/"labels"/"test",  ANN_DIR, stats, rng)

    print(f"Processed -> train={n_train}  val={n_val_c}  test={n_test}")
    print(f"Stats: {dict(stats)}")

    all_heights = h_train + h_val + h_test
    buckets = _bucket_stats(all_heights)

    # Write dataset YAML — imgsz matches sensor resolution
    yaml_content = f"""# LLVIP Infrared — sensor-emulated person detection
# Auto-generated by scripts/prepare_dataset.py
# Sensor: 160x120 (Lepton 3.5 class), blur_sigma={BLUR_SIGMA}, noise_sigma={NOISE_SIGMA_DN}dn
path: {PROC.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['person']
"""
    (CONFIGS / "llvip.yaml").write_text(yaml_content, encoding="utf-8")
    print("Wrote configs/llvip.yaml")

    # Write report with bucket breakdown
    REPORTS.mkdir(exist_ok=True)
    bucket_rows = "\n".join(
        f"| {k} | {v} |" for k, v in buckets.items()
    )
    report = f"""# Dataset Preparation Report — LLVIP Infrared (Sensor-Emulated)

Generated by `scripts/prepare_dataset.py`

## Sensor Emulation Parameters

| Parameter | Value |
|-----------|-------|
| Output resolution | {SENSOR_W}x{SENSOR_H} px |
| Optical blur (Gaussian sigma) | {BLUR_SIGMA} px |
| Readout noise (sigma) | {NOISE_SIGMA_DN} DN |
| Min person height (kept) | {MIN_PERSON_PX_H} px |
| Interpolation | cv2.INTER_AREA |

> **Why this matters:** LLVIP native resolution is 1280x1024.
> A Lepton 3.5-class sensor captures 160x120 at ~9 Hz. Evaluating at 640px
> overstates performance — a person 25m away occupies ~6px at sensor scale.
> These numbers reflect what the physical hardware would actually see.

## Splits

| Split | Images | Person annotations |
|-------|--------|--------------------|
| train | {n_train} | ~{stats['persons']} (approx, all splits combined) |
| val   | {n_val_c} | — |
| test  | {n_test} | — |

## Person Height Distribution (sensor resolution, all splits)

Person height in pixels at {SENSOR_W}x{SENSOR_H}. Recall will drop sharply
below 8px (far-distance targets). Report recall by bucket in eval harness —
do not quote one aggregate number in pitch materials.

| Height bucket | Annotation count |
|---------------|-----------------|
{bucket_rows}

## Notes
- Source: LLVIP infrared channel only (thermal, 8-bit grayscale)
- Class: `person` only (class 0 in YOLO format)
- Images with zero qualifying person annotations excluded
- Val fraction: {VAL_FRAC*100:.0f}% of raw train, seed={SEED}
- Missing images skipped: {stats['missing_img']}
- Missing XMLs skipped: {stats['missing_xml']}
- Images dropped (all persons too small): {stats['no_person_after_filter']}
"""
    (REPORTS / "dataset_prep.md").write_text(report, encoding="utf-8")
    print("Wrote reports/dataset_prep.md")


if __name__ == "__main__":
    main()
