# StitchAI — Project Overview

**A Zero-Shot/Few-Shot Vision-Language Framework for Unified Defect, Safety, and Machinery Anomaly Detection in Bangladesh's Garment Sector**

Team: HexaMind | United International University
Competition: BCOLBD 2026 — AI Category

---

## 1. What This Project Is

StitchAI is **one shared AI pipeline** that can look at a photo from a garment factory and tell you if something is wrong — whether that photo is of a piece of fabric, a worker's safety gear, or a piece of sewing/production machinery. Instead of building three separate AI models (one per problem), the project uses **a single vision-language backbone** that has been shown in published research to generalize across many types of visual anomalies with very little training data.

The core bet of the project: **one adaptable model + a small bank of "normal" reference images per category** can replace **three separately-trained, separately-maintained classifiers**, while still being accurate enough to be useful, and while being honest about where it currently falls short.

This is explicitly framed as an **adaptation and integration project**, not a "we invented a new architecture" project. The novelty is in applying an existing, peer-reviewed zero-shot/few-shot anomaly detection method (AnomalyCLIP) to a domain — Bangladeshi RMG (Ready-Made Garments) manufacturing — where it has not been applied before, and wrapping it in a real, usable, explainable, auditable product.

---

## 2. The Problem Being Solved

- RMG is ~80.62% of Bangladesh's exports (FY 2025–26) — quality, safety, and equipment reliability all directly affect the sector's competitiveness and buyer trust.
- Inside a factory, three different teams do three different kinds of visual inspection:
  - **QC teams** → fabric/garment defects (stains, holes, stitching faults, misalignment)
  - **Safety teams** → workplace safety conditions
  - **Maintenance teams** → visible wear/damage on machinery
- Today, AI-based inspection is usually built as **one model per task** → new data collection, new labeling, new training, new maintenance overhead every time a new defect type or product line shows up.
- **Core research question:** *Can a shared visual AI framework achieve useful performance across multiple RMG inspection tasks with lower data and retraining requirements than building a separate model for each one?*

---

## 3. The Proposed Solution

A **Universal Anomaly Detection Framework** built around:

1. **One shared backbone** (AnomalyCLIP — a CLIP-based, object-agnostic, zero-/few-shot anomaly detection model) used across all three categories.
2. **A small "few-shot reference bank"** (3–5 "normal" images) per category, supplied at inference time — this is what lets one backbone adapt to fabric, safety, or machinery without retraining.
3. **A decoupled explanation layer** — when an anomaly is flagged, a vision-language model (VLM) generates a plain-language explanation of what looks wrong, so a human supervisor doesn't just see a number, they see *why*.
4. **A unified audit log** — every inference (any category) gets logged with category, score, verdict, and timestamp, producing one consolidated compliance/audit trail instead of three separate paper trails.
5. **Human-in-the-loop by design** — the system never takes autonomous action (no auto-rejecting a fabric roll, no auto-shutting-down a machine). It flags, scores, localizes, and explains; a human decides.

### The Three Prototype Categories
| Category | What it detects | End user | Data source for prototype |
|---|---|---|---|
| Garment/fabric quality | Stains, holes, stitching faults, surface defects | QC staff, line supervisors | Public fabric defect datasets |
| Visual safety screening | Recognizable safety-related conditions (e.g. PPE compliance) | Safety officers, compliance staff | Public construction-site PPE dataset |
| Machinery anomaly (proof of concept) | Visible wear/damage/abnormal appearance of equipment | Maintenance teams | MVTec-AD industrial anomaly benchmark (proxy — no RMG-specific dataset exists yet) |

> **Important, disclosed limitation:** none of the training/eval data was captured on a real Bangladeshi factory floor. The prototype is a proof of methodology. Real factory data collection is defined as the Phase 2 pilot step (see Workflow doc), and because the model is few-shot, adapting to real factory images later only requires a handful of new reference images per category — not a full retrain.

---

## 4. What Makes This Different From What Already Exists

| Existing option | Strength | Gap it has |
|---|---|---|
| Purpose-built fabric inspection hardware (Uster, Barco-class) | Very accurate | Expensive, foreign, single-purpose — can't extend to safety/machinery |
| Western construction-site PPE detection tools | Works for PPE | Not localized to RMG, single-purpose |
| Published zero-/few-shot research (WinCLIP, AnomalyCLIP, AnomalyGPT) | Category-agnostic methodology | Research code only — no product, no RMG adaptation, no explanation layer, no deployment |
| Typical student-project CV classifiers | Cheap to build | One model per category — reproduces the exact fragmentation problem this project solves |

**StitchAI's claimed gap** = the intersection of: (1) generalizes across anomaly categories, (2) produces human-readable explanations, (3) is actually adapted to the RMG domain, (4) is deployable on free-tier infrastructure. No existing option hits all four.

---

## 5. Scope

**In scope**
- One shared inference pipeline across the three categories above
- A few-shot reference bank per category
- A natural-language explanation layer over flagged anomalies
- A unified, timestamped inference log (audit trail)
- A working demo interface + public code repository

**Out of scope (deliberately, and disclosed as such)**
- Diagnosing internal mechanical/electrical faults from an external image
- Any autonomous action (rejection, shutdown, disciplinary action)
- Real-time multi-camera video analytics at full frame rate
- Any claim of validated accuracy on real Bangladeshi factory imagery (that needs the Phase 2 pilot)

---

## 6. Objectives

1. Demonstrate anomaly detection across at least 2, ideally all 3, RMG-relevant categories using one shared backbone.
2. Attach a plain-language explanation to every flagged anomaly.
3. Quantify the trade-off between the shared (few-shot) approach and conventional task-specific baselines, measured on: detection performance, false-negative rate, localization quality, inference time, and amount of task-specific data required.
4. Keep the entire prototype runnable on free/low-cost infrastructure.

---

## 7. Architecture Summary (Core — should not change)

```
Input Image (fabric / safety / machinery)
        │
        ▼
Few-Shot Reference Bank (3–5 "normal" images, per category)
        │
        ▼
Shared Visual-Language Backbone  ──►  Image-level anomaly score
   (AnomalyCLIP: CLIP ViT-L/14@336px,      +
    frozen visual encoder + learned,   Pixel-level anomaly heatmap
    object-agnostic prompts)
        │
        ▼
Threshold Decision (Normal vs. Anomalous — configurable)
        │
   ┌────┴─────┐
   ▼          ▼
Normal    Anomalous
(log it)      │
              ▼
      Explanation Layer (VLM — decoupled/optional)
              │
              ▼
   Unified Output: Verdict + Confidence Score +
        Heatmap + Plain-Language Explanation
              │
              ▼
     Audit / Compliance Log (category, score, verdict, timestamp)
```

**This backbone-plus-few-shot-reference-bank-plus-decoupled-explanation-layer design is the architectural core of the project and should be preserved even if specific tools (frontend framework, hosting, which VLM API, database choice, etc.) change.**

### System Layers
- **Client layer** — upload interface (image in → verdict/heatmap/explanation out)
- **Application layer** — backend API + an orchestrator that routes image → model → threshold check → (optionally) explanation service → log
- **Model-serving layer** — AnomalyCLIP inference (GPU) + explanation service (VLM API call, swappable for self-hosted later)
- **Data/audit layer** — reference image bank storage + inference log store

---

## 8. Flexible vs. Fixed Parts

**Fixed (core, keep as-is):**
- Shared backbone approach (one model, not one-per-category)
- Few-shot reference-image adaptation mechanism
- Score + heatmap + explanation output format
- Human-in-the-loop, non-autonomous decision model
- Unified audit logging across all categories
- Explanation layer decoupled from detection layer

**Flexible (swap freely to fit your timeline/skills/budget):**
- Exact backbone implementation (AnomalyCLIP vs. WinCLIP vs. another CLIP-based zero-/few-shot method)
- Explanation-layer model/provider (Gemini Flash vs. any other hosted or local VLM)
- Backend framework (FastAPI vs. Flask/Django/Node, etc.)
- Frontend (Streamlit vs. plain HTML/CSS/JS vs. React)
- Compute (Colab T4 vs. Kaggle GPU vs. any other free/low-cost GPU)
- Data store (SQLite vs. any lightweight DB)

---

## 9. Key Risks Already Identified (from the whitepaper — carry these into build decisions)

| Risk | Mitigation approach |
|---|---|
| Public datasets ≠ real factory conditions | Treat as proof of methodology; plan small real-image fine-tuning later (cheap, because it's few-shot) |
| No public RMG machinery-fault dataset | Use MVTec-AD explicitly as a disclosed proxy/proof-of-concept, not a validated result |
| Zero-/few-shot accuracy ceiling vs. fully supervised models | Frame as the intentional trade-off / value proposition (generalization over raw peak accuracy) |
| Limited compute/budget | Lightweight frozen-backbone approach + free-tier hosted API for explanations |
| False positives erode trust / false negatives are costly | Human-in-the-loop, configurable confidence threshold |
| Privacy/surveillance concerns (safety category) | Process on-premise where possible; retain only flagged frames + aggregate stats, not continuous footage |
| Explanation layer can hallucinate | Explanation is a secondary annotation only — score + heatmap remain the source of truth |
| Third-party API dependency during live demo | Core detection must run fully locally/independent of the explanation API; prepare cached fallback explanations for the demo |

---

## 10. Expected Deliverables

1. A working prototype (code + demo interface) covering at least 2–3 RMG categories through one shared pipeline.
2. Public code repository.
3. Reported per-category results (score, false-negative rate, localization quality, inference time, data required) — reported honestly, limitations included.
4. A clearly documented "next step" path from prototype → single-factory pilot, described as infrastructure substitutions rather than architectural redesign.
