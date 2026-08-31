"""
Pick 3-5 'normal' images for a category and copy them into
data/reference_bank/<category>/ (Phase 1, step 2 / Phase 3, step 2 / Phase 4, step 2).

Usage:
    python scripts/build_reference_bank.py --category fabric --source data/raw/fabric_defect/normal --n 5
"""
import argparse
import shutil
from pathlib import Path


def build_reference_bank(category: str, source_dir: str, n: int = 5):
    src = Path(source_dir)
    dst = Path("data/reference_bank") / category
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:n]

    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    for img in images:
        shutil.copy(img, dst / img.name)

    print(f"Copied {len(images)} reference images to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=["fabric", "safety", "machinery"])
    parser.add_argument("--source", required=True, help="Folder of candidate 'normal' images")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    build_reference_bank(args.category, args.source, args.n)
