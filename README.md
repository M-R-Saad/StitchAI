# StitchAI

**A Zero-Shot/Few-Shot Vision-Language Framework for Unified Defect, Safety, and Machinery Anomaly Detection in Bangladesh's Garment Sector**

Team: HexaMind | United International University
Competition: BCOLBD 2026 — AI Category

## What this is

One shared vision-language anomaly-detection backbone (AnomalyCLIP), adapted via small
per-category "few-shot reference banks" (3–5 normal images) to cover three RMG inspection
tasks — fabric quality, worker safety, and machinery wear — through a single pipeline,
with a decoupled plain-language explanation layer and a unified audit log.

See `docs/whitepaper.pdf` for the full write-up and `docs/demo_script.md` (to be filled in
during Phase 7) for the live-demo walkthrough.

## Status

This is a **Phase 0 skeleton** — folder structure and stub code matching the architecture
described in the whitepaper. Nothing here is trained or wired up yet. Build order follows
the phased workflow below; each phase leaves you with something demoable.

## Repository layout

```
stitchai/
├── data/            # raw datasets, few-shot reference banks, processed images
├── backbone/        # THE CORE — shared AnomalyCLIP wrapper (category-agnostic)
├── explanation/      # decoupled VLM explanation layer + cached fallbacks
├── backend/          # FastAPI app: /infer, /logs, orchestrator
├── frontend/         # Streamlit upload UI
├── storage/          # SQLite DB + InferenceLog / ReferenceImage models
├── evaluation/       # shared-approach vs. baseline comparison
├── notebooks/        # per-category exploration/eval notebooks
├── docs/             # whitepaper, diagrams, demo script
└── scripts/          # dataset download, reference bank builder, demo seeding
```

Category-specific behavior lives ONLY in `data/reference_bank/<category>/` (which normal
images you feed the model) and in `backbone/prompts.py` (which text prompts you use) —
never in separate model code per category. That separation is the whole point of the
project; keep `backbone/model_wrapper.py` category-agnostic.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in GEMINI_API_KEY etc.
```

Clone the official AnomalyCLIP implementation into `backbone/anomalyclip/` (see that
folder's own README for the expected layout) — it's kept out of `requirements.txt`
since it's used as source, not an installable package.

### Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### Run the frontend

```bash
streamlit run frontend/app.py
```

## Build order

Follow the phased plan (Phase 0 → Phase 8) from the project workflow doc:

0. Setup — environment, AnomalyCLIP smoke test, hello-world backend/frontend loop.
1. Single-category proof (fabric) — the minimum viable demo.
2. Add the explanation layer + cached fallback.
3. Extend to the safety category through the *same* shared pipeline.
4. Add machinery (MVTec-AD proxy data), clearly labeled as proof-of-concept in the UI.
5. Unified audit log (`/logs`).
6. Baseline comparison & metrics table.
7. Polish, risk-proofing, demo script.
8. (Stretch) Docker, self-hosted VLM, more baselines, threshold-tuning UI.

If you have to cut scope, cut in reverse order — protect Phases 0–3 no matter what.

## Disclosed limitations (carry through to any pitch/demo)

- No training/eval data was captured on a real Bangladeshi factory floor; this is a proof
  of methodology, not a validated deployment.
- The machinery category uses MVTec-AD as a disclosed proxy — no RMG-specific machinery
  dataset exists yet. Always label this "proof of concept" in the UI.
- The system is human-in-the-loop by design and never takes autonomous action.

## License

TBD.
