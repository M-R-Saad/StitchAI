"""
Quick manual test: score every image in a folder using our own AnomalyCLIPWrapper
(not the standalone AnomalyCLIP CLI script), so we can validate a new category the
same way Phase 1 validated fabric - by comparing score distributions.

Usage:
    python scripts/score_folder.py --folder data/processed/safety_samples/compliant --category safety
    python scripts/score_folder.py --folder data/processed/safety_samples/violation --category safety
"""
import argparse
from pathlib import Path

import yaml

from backbone.model_wrapper import AnomalyCLIPWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    with open("backbone/config.yaml") as f:
        config = yaml.safe_load(f)

    model = AnomalyCLIPWrapper(
        checkpoint_path=config["model"]["checkpoint_path"],
        device=config["model"].get("device", "cpu"),
    )
    model.load()

    folder = Path(args.folder)
    images = sorted(
        p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in images:
        result = model.score_image(str(img_path), args.category, reference_bank=[])
        print(f"{img_path.name}: {result.score:.4f}")
