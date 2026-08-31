"""
Split the Construction Site Safety (Roboflow, YOLOv8-format) dataset into
"compliant" and "violation" images, based on their label files, then build the
safety reference bank from the compliant set.

Unlike WFDD (Phase 1), this dataset doesn't ship with a pre-made normal/anomaly split
- it's an object-detection dataset (images/ + labels/ folders, one .txt per image with
YOLO-format bounding boxes). We derive the normal-vs-anomaly split ourselves: an image
whose label file contains none of the "NO-*" violation classes is compliant; an image
containing at least one is a violation.

Class map (from the dataset's own documentation):
    0: Hardhat        1: Mask           2: NO-Hardhat     3: NO-Mask
    4: NO-Safety Vest  5: Person         6: Safety Cone    7: Safety Vest
    8: machinery       9: vehicle

Usage:
    python scripts/split_safety_images.py --split train --n_reference 5
"""
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

VIOLATION_CLASS_IDS = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest


def classify_image(label_path: Path) -> str:
    """Return 'compliant' or 'violation' for one image's label file."""
    if not label_path.exists():
        return "compliant"  # no annotations at all -> nothing flagged as a violation
    class_ids = set()
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        class_ids.add(int(line.split()[0]))
    return "violation" if class_ids & VIOLATION_CLASS_IDS else "compliant"


def split_dataset(data_root: str, split: str) -> Tuple[List[Path], List[Path]]:
    images_dir = Path(data_root) / split / "images"
    labels_dir = Path(data_root) / split / "labels"

    compliant, violation = [], []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        bucket = classify_image(label_path)
        (compliant if bucket == "compliant" else violation).append(img_path)

    return compliant, violation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="data/raw/ppe_safety/css-data",
        help="Folder containing train/valid/test subfolders",
    )
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--n_reference", type=int, default=5)
    args = parser.parse_args()

    compliant, violation = split_dataset(args.data_root, args.split)
    print(f"{args.split}: {len(compliant)} compliant, {len(violation)} violation")

    ref_dir = Path("data/reference_bank/safety")
    ref_dir.mkdir(parents=True, exist_ok=True)
    for img in compliant[: args.n_reference]:
        shutil.copy(img, ref_dir / img.name)
    print(f"Copied {min(args.n_reference, len(compliant))} reference images to {ref_dir}")

    # Also stash a handful of test-worthy images of each kind, for the same kind of
    # before/after scoring comparison we did with fabric in Phase 1.
    sample_dir = Path("data/processed/safety_samples")
    (sample_dir / "compliant").mkdir(parents=True, exist_ok=True)
    (sample_dir / "violation").mkdir(parents=True, exist_ok=True)
    for img in compliant[args.n_reference : args.n_reference + 5]:
        shutil.copy(img, sample_dir / "compliant" / img.name)
    for img in violation[:5]:
        shutil.copy(img, sample_dir / "violation" / img.name)
    print(f"Copied sample compliant/violation test images to {sample_dir}")
