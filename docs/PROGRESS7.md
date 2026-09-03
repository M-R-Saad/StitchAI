# StitchAI — Progress Tracker

Team HexaMind | BCOLBD 2026 — AI Category
Last updated: this session

---

## Environment

- **Local machine**: Windows PC, RTX 3060 (12GB VRAM), CUDA-enabled
- **Project path**: `E:\Projects\stitchai`
- **Python**: 3.10, virtual environment (`venv`) active
- **Model dev also uses**: Google Colab (free T4 GPU) — used for exploration/experiments; local machine is now the primary target for the actual app

---

## ✅ Phase 0 — Setup: COMPLETE

**Goal:** environment ready, nothing model-related yet.

- [x] Cloned/unzipped the project skeleton to `E:\Projects\stitchai`, matching the folder structure in `02_PROJECT_STRUCTURE.md`
- [x] Created and activated a Python venv, installed `requirements.txt`
- [x] Confirmed FastAPI backend runs (`uvicorn backend.main:app --reload --port 8000`) and `/health` returns `{"status": "ok"}`
- [x] Confirmed Streamlit frontend runs (`streamlit run frontend/app.py`) and successfully reaches the backend
- [x] Cloned the official AnomalyCLIP repo into `backbone/anomalyclip/`
- [x] Resolved dependency conflicts along the way:
  - AnomalyCLIP's own `requirements.txt` had outdated pins (`scikit-image==0.20.0`, `scikit-learn==1.2.2`) incompatible with the installed Python — fixed by de-pinning versions in that file
  - Missing `thop` package (used by `AnomalyCLIP.py`) — installed separately
  - Installing AnomalyCLIP's requirements silently downgraded `torch` back to a CPU-only build twice — fixed both times by force-reinstalling the CUDA build:
    `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`
  - That reinstall pulled in newer `numpy`/`pillow` that conflicted with `scipy`/`streamlit` — fixed with `pip install numpy==1.24.4 pillow==10.4.0`
  - **Still to do:** comment out/remove the `torch`/`torchvision` lines in `backbone/anomalyclip/requirements.txt` so this doesn't recur, and update the top-level `requirements.txt` torch pins to `2.5.1`/`0.20.1` to match what's actually installed
- [x] Ran AnomalyCLIP's `test_one_example.py` **locally on the RTX 3060** using a real photo (a tomato with a blemish) — the model correctly localized the defect in the output heatmap, confirming the backbone genuinely works end-to-end on real (non-benchmark) images
- **Note:** the repo's bundled `assets/*.png` files (hazelnut, capsule, etc.) turned out to be README illustration figures, not real single-object test photos — don't use those as inputs; use your own real photos instead

**Exit criteria met:** ✅ backend/frontend hello-world loop works, ✅ AnomalyCLIP produces a real score+heatmap on a real image, locally, on the target GPU.

---

## 🚧 Phase 1 — Single-Category Proof (Fabric): IN PROGRESS

**Goal:** prove the core loop end-to-end on ONE category (fabric) before touching safety or machinery.

- [x] Compared candidate fabric-defect datasets and selected **WFDD (Woven Fabric Defect Detection)** from Kaggle — chosen because its folder structure (train=normal only, test=normal+anomaly, with pixel-level masks) already matches the MVTec-AD-style format AnomalyCLIP's own data loaders expect, minimizing glue code
- [x] Downloaded WFDD via the Kaggle API (`kaggle datasets download -d hodinhtrieu/the-woven-fabric-defect-detection-wfdd -p data/raw/fabric_defect --unzip`) into `data/raw/fabric_defect/WFDD/`
  - Confirmed structure: 4 fabric patterns (`grey_cloth`, `grid_cloth`, `pink_flower`, `yellow_cloth`), each with `train/good/`, `test/{good, contaminated, flecked, line, string}/`, `ground_truth/{defect_type}/` (pixel masks)
  - Using `grey_cloth` as the first category to validate the pipeline on
- [x] Built the fabric **reference bank**: 5 clean images copied from `train/good/` into `data/reference_bank/fabric/` via `scripts/build_reference_bank.py`
- [x] Sanity-tested AnomalyCLIP's zero-shot script (`test_one_example.py`) directly against real WFDD images (not just the tomato/hazelnut demo assets) — confirmed it correctly localizes real fabric defects (e.g. a thin scratch/line defect) with a strong, extended heatmap, vs. weak/diffuse signal on clean images
- [x] **Important finding:** `test_one_example.py`'s pipeline is zero-shot only — it does NOT actually consume a reference bank of normal images. The few-shot reference-bank comparison described in the whitepaper is custom logic our project still needs to add (in `backbone/model_wrapper.py`); the base AnomalyCLIP repo doesn't do this out of the box.
- [x] Discovered the numeric image-level anomaly score (`text_probs`) was computed internally but never printed — added one `print(f"ANOMALY_SCORE: {text_probs.item():.4f}")` line after the `visualizer(...)` call in `test_one_example.py` to expose it
- [x] Collected real score samples on `grey_cloth`: defect images (`test/line/001-004`) scored **0.9527–0.9923**; clean images (`test/good/001-004`) scored **0.6839–0.8917**. Clusters don't overlap — clear separation, though closer than an early 4-sample check suggested.
- [x] Set a real, data-driven starting threshold: updated `backbone/config.yaml` fabric threshold from the placeholder `0.5` to **`0.9`** (sits between max-clean 0.89 and min-defect 0.95)
  - Caveat: based on only 8 images, one pattern (`grey_cloth`), one defect type (`line`). Needs revalidation across more images and the other 3 defect types (`contaminated`, `flecked`, `string`) before Phase 6.
- [x] Implemented the real forward pass in `backbone/model_wrapper.py` (`AnomalyCLIPWrapper.score_image`): model loads once at backend startup (not per-request), zero-shot scoring mirrors the validated `test_one_example.py` logic, returns a reusable `AnomalyResult` (score + heatmap array)
- [x] Wired it into the backend: `backend/orchestrator.py` calls the wrapper, applies the 0.9 threshold, saves a heatmap overlay image, calls the (stubbed) explanation layer when anomalous; `POST /infer` works for `category=fabric`
- [x] Confirmed it works through the Streamlit UI end-to-end:
  - Defect image → "ANOMALOUS", confidence 0.95, heatmap correctly overlaid on the defect region, explanation fallback message shown (expected — Gemini not wired up yet, that's Phase 2)
  - Clean image → "NORMAL", confidence 0.68, correctly below threshold
  - Fixed one small bug along the way: Streamlit's `st.image()` needed the backend's full base URL prefixed onto the relative `/heatmaps/...` path returned by the API

**Exit criteria: ✅ MET.** Upload a fabric photo through the UI → get a correct Normal/Anomalous verdict with a heatmap, entirely through the app (not a manual script). **Phase 1 complete.**

---

## ✅ Phase 2 — Explanation Layer: COMPLETE

**Goal:** turn a bare score into a supervisor-usable output.

- [x] Got a Gemini API key (Google AI Studio, free tier) and confirmed it loads correctly from `.env`
- [x] Discovered along the way that the originally-planned `google-generativeai` SDK and `gemini-1.5-flash` model are both deprecated/retired — switched to the current `google-genai` SDK and, after hitting a live 404 from the API itself, `gemini-3.6-flash` (the model `gemini-2.5-flash` now redirects new users to)
- [x] Implemented the real `_call_gemini()` call in `explanation/explain_service.py` using the current SDK
- [x] Verified end-to-end through the Streamlit UI: an anomalous fabric image now returns a real, well-written plain-language explanation (e.g. correctly describing possible pulled/broken threads, discoloration, weave distortion) instead of the fallback message
- [x] Cached fallback path already implemented and proven working in Phase 1 testing (before the API key was set up, every anomalous result correctly showed the fallback message) — the try/except structure in `explain_service.py` handles this automatically
- [x] Unified frontend display already implemented in `frontend/app.py` from the initial skeleton — verdict, score, heatmap, and explanation all show together

**Exit criteria: ✅ MET.** An anomalous fabric image produces a full "verdict + score + heatmap + plain-language explanation" output, with a working fallback if the API fails (already demonstrated, since that's exactly what happened before the key was configured).

**Still worth doing before Phase 7:** pre-generate a few real cached fallback `.txt` files in `explanation/cached_explanations/` for actual demo images (currently the folder only has a README explaining the convention — Phase 2's workflow step 3 in `03_WORKFLOW.md`).

## 🚧 Phase 3 — Extend to Safety Category: IN PROGRESS

**Goal:** prove the "shared backbone, not shared logic" claim by adding a second, structurally different category through the *same* pipeline.

- [x] Selected and downloaded the **Construction Site Safety Image Dataset (Roboflow, via Kaggle, `snehilsanyal/construction-site-safety-image-dataset-roboflow`)** into `data/raw/ppe_safety/css-data/`
  - Structure: YOLOv8 format — `train/valid/test`, each with `images/` + `labels/` (one `.txt` per image, class-id + normalized bbox per line)
  - 10 classes: `Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle`
  - **Different from WFDD**: no pre-made normal/anomaly split — had to derive one ourselves
- [x] Wrote `scripts/split_safety_images.py`: classifies each image as "compliant" (no `NO-*` classes present) or "violation" (at least one `NO-*` class present) by parsing its YOLO label file. Also copies sample compliant/violation images into `data/processed/safety_samples/` for score testing.
  - Result on `train` split: **310 compliant / 2,295 violation** (most real-world site photos contain at least one violation somewhere in frame)
- [x] Built the safety reference bank: 5 compliant images copied into `data/reference_bank/safety/`
- [x] Wrote a reusable scoring test script, `scripts/score_folder.py`, using our own `AnomalyCLIPWrapper` directly (not the standalone CLI) — run as `python -m scripts.score_folder --folder <dir> --category safety`
- [x] **Important finding — current implementation doesn't actually use `backbone/prompts.py`'s category-specific text prompts.** Both our wrapper and the official `test_one_example.py` script condition on AnomalyCLIP's own *learned* prompt embeddings (from the checkpoint), not hand-written text templates per category. That's *why* zero-shot generalized to a tomato, hazelnut, and fabric equally well — `prompts.py` is currently a placeholder for future refinement, not active in the scoring path.
- [x] **Key finding — safety scores show much weaker separation than fabric did.** 5 compliant scores (0.68–0.97) and 5 violation scores (0.68–0.99) heavily overlap — one compliant image scored 0.6752, one violation scored 0.6755, nearly identical. No clean threshold exists in this sample.
  - **Likely cause:** fabric defects are local texture irregularities — exactly what AnomalyCLIP's own training data (MVTec-AD/VisA) consists of. PPE compliance is a semantic, object-presence question ("is this one specific person's head bare") in a busy multi-person scene — a different kind of task the shared texture-anomaly backbone isn't naturally suited to when scoring the whole image at once.
  - This isn't an unanticipated failure — the whitepaper's own Section 1.2 flags that safety conditions may need "a lightweight task-specific visual component," rather than assuming the shared backbone handles everything equally well.
- [x] **Tried Option B — cropped individual people, scored each crop separately.** Wrote `scripts/crop_people_safety.py`: for each image, crops out every "Person" box; a crop is labeled "violation" if a `NO-*` box's center falls inside it, else "compliant". Ran on 40 source images → 46 compliant crops, 158 violation crops. Scored both with `score_folder.py`.
  - **Result: made separation worse, not better.** Compliant crops clustered tightly high (0.84–0.99). Violation crops were *more* spread out, including a real cluster of unusually *low* scores (0.49, 0.54, 0.67, 0.76...) that compliant crops never showed — backwards from what's needed (violations should score higher, not lower, if higher = more anomalous).
  - **Root cause, now clearer:** almost every cropped person — compliant or not — scores high, because a cropped human figure is wildly out-of-distribution for what this AnomalyCLIP checkpoint learned as "normal" (industrial textures/objects like fabric, capsules, hazelnuts — see MVTec-AD/VisA). The model isn't judging PPE presence at all; it's reacting to "does this look like a normal industrial texture," and a person essentially never does, regardless of compliance. The scattered low violation scores are likely crop artifacts (odd framing/background-heavy crops), not real signal.
  - **Decision: parking the safety category here per plan.** Confirmed this isn't a quick fix — the mismatch is between the checkpoint's training domain and what "safety compliance" actually requires, not something a preprocessing trick resolves. Moving on to Phase 4 (machinery) and the rest of the plan; will revisit safety later with a clearer head (likely needs either a different/task-specific model for this category, or a fundamentally different scoring approach — not just cropping).
- [ ] **Safety category: ON HOLD.** Not enabled in `backbone/config.yaml` (`enabled: false`, as originally stubbed). Reference bank and sample data already collected for whenever this is revisited.

**Exit criteria: PARTIALLY MET / DEFERRED.** Fabric works cleanly through the shared pipeline. Safety does not yet have usable separation — parked deliberately rather than shipped with a broken/misleading verdict. Revisit after Phases 4-7.

---

## ✅ Phase 4 — Machinery Category (MVTec-AD Proxy Data): COMPLETE

**Goal:** third category, explicitly disclosed as proof-of-concept using proxy data (MVTec-AD), not RMG-validated.

- [x] Downloaded the full MVTec-AD benchmark (Kaggle mirror `ipythonx/mvtec-ad`) into `data/raw/mvtec_ad/` — confirmed same folder format as WFDD (`train/good`, `test/{good, defect_types...}`)
- [x] Selected **`metal_nut`** as the machinery object subset (real metal industrial component, defect types: `bent`, `color`, `flip`, `scratch`) — a good fit for "visible wear/damage on machinery" per the whitepaper
- [x] Built the machinery reference bank: 5 clean images from `train/good` copied into `data/reference_bank/machinery/`
- [x] Scored 22 `good` + 23 `scratch` images with `score_folder.py`. **Real separation found, though tighter/more compressed than fabric's:**
  - good: mean ≈ 0.981, range 0.9631–0.9893
  - scratch: mean ≈ 0.991, range 0.9732–0.9966
  - At threshold **0.99**: zero good images cross it (false-positive rate 0 in this sample), ~74% of scratch images do (some very faint scratches missed — plausible, not a red flag)
- [x] Set `backbone/config.yaml` machinery threshold to **0.99**, `enabled: true`, `proof_of_concept: true`
- [x] Verified end-to-end through the Streamlit UI: correctly flagged a real scratch defect as anomalous, correctly left a clean metal nut as normal. "Proof of concept" notice (from the Phase 0 skeleton) displays for this category.

**Exit criteria: ✅ MET.** All three categories now run through the one shared pipeline (fabric working cleanly; machinery working with a tighter but real threshold; safety deliberately parked per Phase 3 findings, not broken but not shipped either).

---
---

## ⬜ Not yet started

- **Note carried forward:** current scoring is zero-shot only (works cleanly for fabric; works with a tighter threshold for machinery; fundamentally mismatched for safety — see Phase 3, parked deliberately). True few-shot reference-bank comparison was deliberately deferred for fabric — planned as a refinement, slots into `AnomalyCLIPWrapper.score_image()` without changing its public interface.
- **Safety category** — parked (see Phase 3). Revisit after the phases below, likely needs a different approach than the shared AnomalyCLIP backbone (whitepaper Section 1.2 anticipated this: "a lightweight task-specific visual component" for safety-type conditions).
## ✅ Phase 5 — Audit Log / Unified Output Layer: COMPLETE

**Goal:** make the "unified compliance record" claim real, not just described.

- [x] Wired `storage/db.py`'s `init_db()` into `backend/orchestrator.py` (runs at startup, creates `storage/stitchai.db` + tables if missing)
- [x] Every `/infer` call now writes one `InferenceLog` row (category, score, verdict, image_ref, timestamp) — same table regardless of category
- [x] `backend/routes/logs.py` now actually queries the table (most recent first) instead of returning an empty list
- [x] `frontend/app.py` restructured into two tabs: "Run Inspection" (unchanged behavior) and a new "Audit Log" tab showing all entries as a table, with a per-category count summary and a refresh button. Also added an in-UI heads-up banner for the parked "safety" category, so it's disclosed upfront rather than discovered mid-demo.
- [x] Verified end-to-end: ran several fabric and machinery inspections, confirmed all of them appear together in the Audit Log tab with correct categories/verdicts/scores, and successfully exported the log as CSV via Streamlit's built-in dataframe export

**Exit criteria: ✅ MET.** A single log/table shows entries from multiple categories (fabric + machinery so far), proving the "one audit trail instead of three" claim. (Configurable per-category threshold via `config.yaml` was already effectively in place since Phase 1/4 — no separate UI control was added, since editing the YAML directly is a reasonable, low-risk way to configure this for now.)

---
- **Phase 6** — Baseline comparison & metrics (shared few-shot approach vs. a simple supervised baseline)
- **Phase 7** — Polish, risk-proofing, demo script, repo cleanup
- **Phase 8** (stretch) — Docker, self-hosted VLM swap, second baseline category, live threshold-tuning UI

---

## Known housekeeping items to circle back to

- Update `requirements.txt` torch/torchvision pins to match the installed CUDA versions (2.5.1 / 0.20.1)
- Strip torch/torchvision pins from `backbone/anomalyclip/requirements.txt` to prevent future accidental CPU-build reinstalls
- `requirements.txt`'s `google-generativeai==0.8.1` line is stale — replaced in practice with `google-genai>=1.0.0`; update the file to match
- Decide on safety and machinery datasets (deferred — safety: Construction Site Safety dataset from the original list; machinery: MVTec-AD) once Phase 1 is solid
