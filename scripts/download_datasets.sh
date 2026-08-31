#!/usr/bin/env bash
# Download the public datasets used by each category. Fill in real dataset
# sources/URLs as you pick them (Phase 1 for fabric, Phase 3 for safety, Phase 4 for
# machinery/MVTec-AD).
set -euo pipefail

mkdir -p data/raw/fabric_defect data/raw/ppe_safety data/raw/mvtec_ad

echo "TODO (Phase 1): download a public fabric-defect benchmark into data/raw/fabric_defect/"
echo "TODO (Phase 3): download a construction-site PPE dataset into data/raw/ppe_safety/"
echo "TODO (Phase 4): download MVTec-AD (or a relevant object subset) into data/raw/mvtec_ad/"
