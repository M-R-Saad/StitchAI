# StitchAI

> [!NOTE]
> A zero-shot/few-shot vision-language inspection platform for fabric defects, worker
> safety, and machinery anomalies in Bangladesh's garment sector.

**Team HexaMind** · United International University · BCOLBD 2026 AI Category

## Project idea

Bangladesh's ready-made garment (RMG) industry depends on consistent product quality,
safe working conditions, and reliable production equipment. A missed fabric defect can
affect an entire order, an unsafe condition can put workers at risk, and visible machine
wear can lead to downtime or costly repairs.

StitchAI is designed as one practical inspection pipeline for all three problems. A
supervisor uploads a photo of fabric, a safety condition, or machinery; the system returns
an anomaly score, highlights suspicious regions, and optionally explains the result in
plain language. A human remains responsible for the final decision.

## Why it matters for Bangladesh

- **Protects quality and buyer trust:** Earlier defect detection can reduce rework,
  waste, rejected batches, and delays.
- **Supports safer factories:** Visual safety checks can help teams spot recognizable
  PPE and workplace risks during routine inspections.
- **Reduces maintenance blind spots:** A visible warning sign on equipment can be logged
  and reviewed before it becomes a larger production problem.
- **Fits real data constraints:** Many factories cannot create large, labeled datasets
  for every new defect. StitchAI uses one shared vision-language model and a small bank of
  normal reference images for each inspection category.

StitchAI combines one AnomalyCLIP backbone with small category-specific reference banks.
Upload an image through the Streamlit dashboard, receive a normal/anomalous verdict and
score, inspect the heatmap, and keep the result in a unified audit log.

## At a glance

| Inspection area | What the prototype checks | Reference bank |
| --- | --- | --- |
| Fabric | Surface defects and irregular texture | `data/reference_bank/fabric/` |
| Worker safety | PPE and unsafe visual conditions | `data/reference_bank/safety/` |
| Machinery | Wear and surface anomalies | `data/reference_bank/machinery/` |

## Defect examples

These uploaded fabric samples contain visible surface defects detected during inspection.
Each real input is shown beside its corresponding anomaly heatmap. Warmer colors indicate
regions that contributed more strongly to the anomaly score.

<p align="center">
  <img src="docs/images/defect_image1.png" alt="Fabric sample with two defects" width="320" />
  <img src="docs/images/fabric-defect-heatmap.png" alt="Heatmap for two fabric defects" width="320" />
</p>

<p align="center">
  <em>Two-defect input</em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Two-defect heatmap</em>
</p>

<p align="center">
  <img src="docs/images/defect_image2.png" alt="Fabric sample with one defect" width="320" />
  <img src="docs/images/defect_image2-heatmap.png" alt="Heatmap for one fabric defect" width="320" />
</p>

<p align="center">
  <em>One-defect input</em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Single-defect heatmap</em>
</p>

### Other inspection categories

<p align="center">
  <img src="docs/images/safety-example.jpg" alt="Worker safety inspection example" width="250" />
  <img src="docs/images/machinery-example.png" alt="Machinery inspection example" width="250" />
</p>

## Status

This is an active **prototype**. The upload dashboard, FastAPI inference route, reference
banks, generated heatmaps, explanation fallback, and unified logs are wired together. The
model and thresholds still require broader validation before production use.

See `docs/01_PROJECT_OVERVIEW.md` for the full project context and `docs/03_WORKFLOW.md`
for the development workflow.

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

On Windows PowerShell, activate the environment with:

```powershell
.\venv\Scripts\Activate.ps1
```

Clone the official AnomalyCLIP implementation into `backbone/anomalyclip/` (see that
folder's own README for the expected layout) — it's kept out of `requirements.txt`
since it's used as source, not an installable package.

### Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Check that it is ready at `http://localhost:8000/health`.

### Run the frontend

```bash
streamlit run frontend/app.py
```

Open the dashboard at `http://localhost:8501`. Start the backend before uploading an
image so the frontend can reach `/infer` and `/logs`.

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
