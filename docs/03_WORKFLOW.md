# StitchAI — Build Workflow (Phased)

This is a practical, "do this, then this" build order — designed so that at the end of every phase you have *something demoable*, even if you run out of time before the last phase.

---

## Phase 0 — Setup (Day 0–1)

**Goal:** environment ready, nothing model-related yet.

1. Create the repo, set up the folder structure (see `02_PROJECT_STRUCTURE.md`).
2. Set up a free-tier GPU environment (Colab T4, Kaggle, or equivalent).
3. Get the official AnomalyCLIP implementation running on a single sample image, using its own demo data — **don't touch RMG data yet.** Goal here is purely: "does the backbone run and produce a score + heatmap at all."
4. Set up API access for whichever explanation-layer model you'll use (Gemini Flash free tier, or a local VLM if you'd rather avoid API dependency from the start).
5. Pick your backend/frontend stack (FastAPI+Streamlit is the whitepaper default, but any stack works) and get a trivial "hello world" endpoint + page running end-to-end, so your deployment pipeline is proven before you add complexity.

✅ **Exit criteria:** AnomalyCLIP produces a score+heatmap on a stock image; your backend/frontend "hello world" loop works; explanation API responds to a test prompt.

---

## Phase 1 — Single-Category Proof (Fabric) (Day 1–3)

**Goal:** prove the core loop end-to-end on ONE category before touching the other two. Fabric first because public fabric-defect datasets are easiest to get.

1. Download a public fabric defect dataset (e.g. any commonly-used fabric-defect benchmark).
2. Build the fabric **reference bank**: pick 3–5 clean "normal" fabric images.
3. Write `prompts.py` category prompts for "fabric" (generic normal vs. anomalous visual-semantic wording — object-agnostic, not "hole" vs "stain" specific).
4. Run AnomalyCLIP with the fabric reference bank against a handful of known-defective fabric images. Check: does the anomaly score go up on defective images, does the heatmap roughly localize the defect?
5. Pick and hardcode a confidence threshold for now (you'll make it configurable later).
6. Wire this into the backend: `POST /infer` takes an image, returns score + heatmap for the fabric category.
7. Basic frontend: upload a fabric image, see verdict + heatmap.

✅ **Exit criteria:** you can upload a fabric photo through the UI and get back a sensible Normal/Anomalous verdict with a heatmap. This is your minimum viable demo — protect time to reach this point above everything else.

---

## Phase 2 — Add Explanation Layer (Day 3–4)

**Goal:** turn a bare score into a supervisor-usable output.

1. When verdict = Anomalous, pass the flagged region/score to the explanation service.
2. Prompt-engineer the explanation call to describe *what* looks wrong in plain language, referencing the flagged region.
3. Build the "cached fallback explanations" folder — pre-generate a few explanations for your demo images now, so that if the API is down/rate-limited during the actual presentation, you have a backup ready (whitepaper Section 3.8 risk).
4. Update frontend to display: verdict, confidence score, heatmap, and explanation text together as one unified output.

✅ **Exit criteria:** an anomalous fabric image produces a full "verdict + score + heatmap + plain-language explanation" output, with a working fallback if the API fails.

---

## Phase 3 — Extend to Safety Category (Day 4–6)

**Goal:** prove the "shared backbone, not shared logic" claim by adding a second, structurally different category using the *same* pipeline code.

1. Download a public construction-site PPE dataset.
2. Build the safety **reference bank** (3–5 "normal"/compliant images).
3. Add safety-specific prompts to `prompts.py` — but do **not** duplicate the backend/orchestrator logic. The same `/infer` endpoint should accept a `category` parameter and route to the right reference bank + prompts.
4. Test: does the exact same backbone code correctly flag non-compliant safety images using only a different reference bank + prompts?
5. Update frontend to let the user pick/upload for either category.

✅ **Exit criteria:** two categories work through one shared pipeline, differing only in reference bank + prompts — no per-category model code. This is the core proof of the project's thesis.

---

## Phase 4 — Add Machinery Category (Proxy Data) (Day 6–7)

**Goal:** third category, explicitly disclosed as proof-of-concept using proxy data (MVTec-AD), not RMG-validated.

1. Download the MVTec-AD industrial anomaly benchmark (a relevant object subset — e.g. metal/mechanical parts — is enough; you don't need the whole benchmark).
2. Build the machinery reference bank from MVTec-AD "good" samples.
3. Add machinery prompts to `prompts.py`.
4. Run the same pipeline. Confirm it produces sensible scores/heatmaps on the proxy data.
5. **Explicitly label this category in the UI/demo as "proof of concept — proxy industrial data, not RMG-specific"** so you're never overclaiming during judging.

✅ **Exit criteria:** all three categories run through the one shared pipeline. This is your "full" demo state.

---

## Phase 5 — Audit Log / Unified Output Layer (Day 7–8)

**Goal:** make the "unified compliance record" claim real, not just described.

1. Build the `InferenceLog` table (category, score, verdict, timestamp, image reference).
2. Every call to `/infer`, regardless of category, writes one row.
3. Build a simple `/logs` view (even a basic table in the frontend) so a judge/demo viewer can see: "here's one log spanning all three inspection types."
4. Add the configurable confidence threshold (per category) mentioned in Phase 1 as a real setting now, not hardcoded.

✅ **Exit criteria:** you can show a single log/table with entries from all 3 categories, proving the "one audit trail instead of three" claim.

---

## Phase 6 — Baseline Comparison & Metrics (Day 8–10)

**Goal:** produce the evidence for the whitepaper's central research question.

1. For at least one category (ideally fabric, since data is easiest), train or find a simple conventional supervised baseline classifier (a small CNN is enough — doesn't need to be sophisticated).
2. Run both the shared few-shot approach and the baseline on the same held-out test images.
3. Measure and record: detection performance (accuracy/AUROC), false-negative rate, localization quality (does the heatmap land on the actual defect region), inference time, and how much labeled/task-specific data each approach needed.
4. Put this into a small results table — this becomes your strongest "look, the trade-off is real and it's worth it" evidence for judges.

✅ **Exit criteria:** one clear table comparing shared-approach vs. baseline on the metrics above.

---

## Phase 7 — Polish, Risk-Proofing, and Demo Prep (Day 10–12)

**Goal:** make sure the live demo cannot fail, and that the pitch matches what's actually built.

1. Re-test the "explanation API down" fallback path — literally kill your network/API key temporarily and confirm the cached fallback kicks in cleanly.
2. Prepare 3–5 pre-selected demo images per category that reliably produce good results (don't rely on live random uploads for the main demo — have a "safe path" ready, with live upload as a bonus/backup).
3. Write the `docs/demo_script.md`: exact click-by-click flow for the presentation.
4. Double check every claim in your pitch matches a real, demoable piece of the system — especially the "proof of concept" disclosure for the machinery category, and the "no real Bangladeshi factory data yet" disclosure.
5. Clean up the public GitHub repo: README with setup instructions, architecture diagram, and a clear "what's real vs. what's the planned next step" section.

✅ **Exit criteria:** you can run through the full 3-category demo without touching code, with a rehearsed script, and every slide/claim traceable to something in the repo.

---

## Phase 8 (Stretch, only if time allows) — Pilot-Readiness Extras

Only attempt these if Phases 0–7 are solid and you have spare time:

- Containerize the model-serving layer (Docker) — shows the "production scale-up path is just infra substitution" claim is real, not just described.
- Swap the hosted explanation API for a small self-hosted open-weight VLM, to demonstrate the on-premise privacy path.
- Add a second baseline category comparison (safety) alongside fabric.
- Add a simple confidence-threshold tuning UI control for live demo interactivity.

---

## Quick Priority Rule If You Run Out of Time

If you have to cut scope, cut in this order (last item cut first, i.e. protect earlier phases):
Phase 8 → Phase 6 (reduce to 1 category, rough numbers) → Phase 4 (machinery) → Phase 5 (log UI can be minimal) → keep Phases 0–3 solid no matter what, since fabric + safety through one shared pipeline with explanations is the minimum version of the actual thesis.
