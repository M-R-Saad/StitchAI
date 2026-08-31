# StitchAI — Project Structure

This is a suggested repository layout that matches the architecture in the whitepaper. It's built to be built **incrementally** — each folder maps to a phase in the workflow doc, so you're never blocked waiting on a piece you haven't built yet.

Tech choices below (FastAPI, Streamlit, SQLite, Gemini Flash) are the whitepaper's defaults — **swap any of them freely** (Flask, React, Postgres, a local VLM, etc.) as long as the folder *roles* stay the same.

```
stitchai/
├── README.md
├── .env.example                     # API keys, config (never commit real .env)
├── .gitignore
├── requirements.txt                 # or pyproject.toml / poetry
│
├── data/
│   ├── raw/                         # untouched downloaded datasets
│   │   ├── fabric_defect/
│   │   ├── ppe_safety/
│   │   └── mvtec_ad/
│   ├── reference_bank/              # the "few-shot normal images" per category
│   │   ├── fabric/
│   │   ├── safety/
│   │   └── machinery/
│   └── processed/                   # resized/cleaned versions used at inference/eval
│
├── backbone/                        # THE CORE — shared detection model
│   ├── anomalyclip/                 # cloned/adapted official AnomalyCLIP implementation
│   ├── model_wrapper.py             # your unified interface: image + ref_bank -> score + heatmap
│   ├── prompts.py                   # object-agnostic text prompts per category
│   └── config.yaml                  # thresholds, checkpoint paths, per-category settings
│
├── explanation/                     # decoupled explanation layer
│   ├── explain_service.py           # wraps whichever VLM API/model you use
│   ├── cached_explanations/         # pre-generated fallback explanations for demo safety
│   └── prompt_templates.py
│
├── backend/                         # application layer (FastAPI or equivalent)
│   ├── main.py                      # app entrypoint
│   ├── routes/
│   │   ├── inference.py             # POST /infer  -> runs backbone + threshold + (maybe) explanation
│   │   └── logs.py                  # GET /logs    -> audit/compliance log retrieval
│   ├── orchestrator.py              # routes image -> model -> threshold -> explanation -> log
│   └── schemas.py                   # request/response data models
│
├── frontend/                        # client layer (Streamlit / plain HTML-CSS / React)
│   ├── app.py                       # upload UI + verdict/heatmap/explanation display
│   └── assets/
│
├── storage/                         # data & audit layer
│   ├── db.py                        # SQLite/Postgres connection + schema
│   ├── models.py                    # InferenceLog, ReferenceImage table definitions
│   └── stitchai.db                  # local dev DB (gitignored in real use)
│
├── evaluation/                      # comparing shared approach vs. baselines
│   ├── baselines/                   # simple supervised classifier(s) per category, for comparison
│   ├── metrics.py                   # detection perf, false-negative rate, localization quality, inference time
│   └── run_eval.py                  # produces the comparison table for the whitepaper/demo
│
├── notebooks/                       # exploration, Colab training/eval notebooks
│   ├── 01_anomalyclip_smoke_test.ipynb
│   ├── 02_fabric_eval.ipynb
│   ├── 03_safety_eval.ipynb
│   └── 04_machinery_proxy_eval.ipynb
│
├── docs/
│   ├── whitepaper.pdf                # the original document
│   ├── architecture_diagram.png
│   ├── infrastructure_diagram.png
│   └── demo_script.md                # what you'll say/click during the live demo
│
└── scripts/
    ├── download_datasets.sh
    ├── build_reference_bank.py
    └── seed_demo_data.py
```

## Why it's organized this way

- **`backbone/` is isolated and untouchable-by-frontend** — this is the one piece that must work identically no matter what category is being scored. Keep it category-agnostic; category-specific behavior should live only in `reference_bank/` (which images you feed it) and `prompts.py` (which text prompts you use), never in separate model files per category. That separation *is* the whole point of the project.
- **`explanation/` is a separate module on purpose** — the whitepaper is explicit that detection must keep working even if the explanation API is down. If this were merged into the backbone code, that guarantee would be easy to accidentally break.
- **`evaluation/` exists from day one** — you need the "shared approach vs. task-specific baseline" comparison for the whitepaper's central claim. Don't leave this until the end; run it incrementally as each category comes online.
- **`storage/` unifies logs across categories** — one `InferenceLog` table with a `category` column, not three separate log tables. This is what makes the "unified audit trail" claim real instead of just marketing language.
- **`notebooks/` per category** — since you're validating each of the 3 categories somewhat independently before wiring them into one pipeline, it's useful to keep exploration notebooks split by category.

## Minimal path if time is very tight

If the timeline gets crunched, the folders you cannot skip are:
`backbone/`, `backend/routes/inference.py`, a bare-bones `frontend/`, and `storage/`. `evaluation/` can be reduced to a single script producing a small metrics table for 1–2 categories, and `explanation/` can temporarily just be a single function call with a hardcoded fallback string.
