"""
Option B experiment (Phase 3): crop individual people out of each safety scene, and
classify each *crop* (not the whole image) as compliant/violation, based on whether a
NO-* violation box's center falls inside that person's box. Then score the crops
directly, on the hypothesis that a single-person crop is a better match for
AnomalyCLIP's texture/local-region strengths than a whole busy multi-person scene.

Quick, bounded experiment - if this doesn't meaningfully improve separation over the
whole-scene scores from Phase 3, drop it and fall back to the whole-scene approach
(deferred, revisit later per the project decision).

Usage:
    python -m scripts.crop_people_safety --split train --n_images 40
"""
import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image

VIOLATION_CLASS_IDS = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest
PERSON_CLASS_ID = 5


def parse_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (class_id, x_center, y_center, w, h), all normalized 0-1."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        boxes.append((class_id, x, y, w, h))
    return boxes


def crop_people(image_path: Path, label_path: Path, out_dir: Path, padding: float = 0.1):
    boxes = parse_label_file(label_path)
    people = [b for b in boxes if b[0] == PERSON_CLASS_ID]
    violations = [b for b in boxes if b[0] in VIOLATION_CLASS_IDS]

    if not people:
        return 0

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    n_saved = 0

    for i, (_, px, py, pw, ph) in enumerate(people):
        # person box in pixel coords, with padding
        x1 = (px - pw / 2 - padding * pw) * W
        y1 = (py - ph / 2 - padding * ph) * H
        x2 = (px + pw / 2 + padding * pw) * W
        y2 = (py + ph / 2 + padding * ph) * H
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            continue  # too small to be a useful crop

        # is any violation box's center inside this person's (unpadded) box?
        is_violation = False
        for _, vx, vy, _, _ in violations:
            vx_px, vy_px = vx * W, vy * H
            if (px - pw / 2) * W <= vx_px <= (px + pw / 2) * W and \
               (py - ph / 2) * H <= vy_px <= (py + ph / 2) * H:
                is_violation = True
                break

        crop = img.crop((x1, y1, x2, y2))
        bucket = "violation" if is_violation else "compliant"
        bucket_dir = out_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        crop.save(bucket_dir / f"{image_path.stem}_person{i}.jpg")
        n_saved += 1

    return n_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/ppe_safety/css-data")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--n_images", type=int, default=40, help="how many source images to scan")
    parser.add_argument("--out_dir", default="data/processed/safety_person_crops")
    args = parser.parse_args()

    images_dir = Path(args.data_root) / args.split / "images"
    labels_dir = Path(args.data_root) / args.split / "labels"
    out_dir = Path(args.out_dir)

    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.n_images]

    total_crops = 0
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        total_crops += crop_people(img_path, label_path, out_dir)

    n_compliant = len(list((out_dir / "compliant").glob("*"))) if (out_dir / "compliant").exists() else 0
    n_violation = len(list((out_dir / "violation").glob("*"))) if (out_dir / "violation").exists() else 0
    print(f"Scanned {len(image_paths)} images, saved {total_crops} person crops")
    print(f"  compliant: {n_compliant}")
    print(f"  violation: {n_violation}")
