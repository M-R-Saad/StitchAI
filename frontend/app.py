# -*- coding: utf-8 -*-
"""
StitchAI - Industrial Vision-Language Anomaly Detection Platform
Streamlit Frontend | Team HexaMind | BCOLBD 2026 AI Category
"""

import os
import glob
import io
import json
import time
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 0. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StitchAI — RMG Inspection Dashboard",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
BACKEND_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")
VERSION = "v0.2.0-PROD"

# -----------------------------------------------------------------------------
# 1. ADVANCED INDUSTRIAL DESIGN SYSTEM
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: #080c14;
    color: #e2e8f0;
}

.block-container {
    padding: 1.8rem 2.5rem 3rem !important;
    max-width: 100% !important;
}

/* Header & Navigation Bar Tweaks */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Dark Shell */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d121d 0%, #080c14 100%) !important;
    border-right: 1px solid #1e293b !important;
    padding-top: 1.25rem;
}

section[data-testid="stSidebar"] > div {
    padding: 0 1.25rem;
}

/* Sidebar Brand Header */
.brand-title {
    font-size: 1.7rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.15rem;
}
.brand-title span {
    background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-subtitle {
    font-size: 0.68rem;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
}

/* Status Indicator Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 1.25rem;
}
.status-online {
    background-color: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.status-offline {
    background-color: rgba(244, 63, 94, 0.12);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.status-online .status-dot {
    background-color: #34d399;
    box-shadow: 0 0 8px #34d399;
    animation: live-pulse 2s infinite ease-in-out;
}
.status-offline .status-dot {
    background-color: #fb7185;
}

@keyframes live-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.9); }
}

/* Sidebar Section Headers */
.sb-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.25rem 0 0.5rem;
}

/* Category Highlight Card in Sidebar */
.ready-card {
    background: rgba(99, 102, 241, 0.07);
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px -3px rgba(0,0,0,0.3);
}
.ready-badge {
    color: #34d399;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 0.4rem;
}
.ready-text {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.45;
}

/* Session Metrics Box in Sidebar */
.session-metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 0.5rem;
    background: #080c14;
    padding: 8px;
    border-radius: 10px;
    border: 1px solid #1e293b;
}
.session-metric-cell {
    text-align: center;
    padding: 6px 2px;
}
.session-metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
}
.session-metric-val.anom {
    color: #f43f5e;
}
.session-metric-lbl {
    font-size: 0.55rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 3px;
}

/* Sidebar Footer */
.sb-footer {
    font-size: 0.7rem;
    color: #475569;
    margin-top: 2rem;
    line-height: 1.6;
}

/* Main Dashboard Header */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.2rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #1e293b;
}
.main-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.main-icon {
    font-size: 2.3rem;
    line-height: 1;
}
.main-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.main-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.25rem;
    font-weight: 400;
}

.tech-badge {
    background: #111827;
    border: 1px solid #1f293b;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Section Bar Labels */
.section-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.65rem;
}

/* Industrial Card Container */
.industrial-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px -2px rgba(0,0,0,0.5);
}

/* Empty State Card */
.empty-preview-box {
    background: #0d1322;
    border: 1px dashed #1e293b;
    border-radius: 12px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin-bottom: 1.25rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.empty-icon {
    font-size: 3rem;
    opacity: 0.35;
    margin-bottom: 0.75rem;
}
.empty-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
}
.empty-subtitle {
    font-size: 0.8rem;
    color: #64748b;
}

/* Verdict Hero Banner */
.verdict-hero {
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px -5px rgba(0,0,0,0.5);
}
.verdict-anomalous {
    background: linear-gradient(135deg, rgba(159, 18, 57, 0.6) 0%, rgba(88, 28, 135, 0.45) 100%);
    border-left: 6px solid #f43f5e;
}
.verdict-normal {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.6) 0%, rgba(15, 23, 42, 0.45) 100%);
    border-left: 6px solid #10b981;
}

.verdict-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.badge-anom {
    background: rgba(244, 63, 94, 0.25);
    color: #fda4af;
    border: 1px solid rgba(244, 63, 94, 0.5);
}
.badge-norm {
    background: rgba(16, 185, 129, 0.25);
    color: #a7f3d0;
    border: 1px solid rgba(16, 185, 129, 0.5);
}

.verdict-heading {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem;
}
.verdict-anomalous .verdict-heading { color: #fecdd3; }
.verdict-normal .verdict-heading { color: #a7f3d0; }

.verdict-desc {
    font-size: 0.85rem;
    color: #cbd5e1;
}

.score-display {
    text-align: right;
    background: rgba(0,0,0,0.3);
    padding: 0.85rem 1.4rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.score-title {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 2px;
}
.score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}

/* AI Vision Diagnostic Console */
.ai-console {
    background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 14px;
    padding: 1.35rem;
    box-shadow: 0 8px 25px -4px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.ai-console::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #38bdf8, #34d399);
}
.ai-console-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.75rem;
    margin-bottom: 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}
.ai-console-title {
    font-size: 0.92rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ai-badge-live {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 9999px;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.ai-badge-live::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #818cf8;
    box-shadow: 0 0 6px #818cf8;
}

.ai-section {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 0.7rem;
    transition: all 0.2s ease;
}
.ai-section:hover {
    border-color: rgba(99, 102, 241, 0.35);
    background: rgba(17, 24, 39, 0.95);
    transform: translateY(-1px);
}
.ai-sec-finding { border-left: 4px solid #f43f5e; }
.ai-sec-cause   { border-left: 4px solid #f59e0b; }
.ai-sec-action  { border-left: 4px solid #10b981; }

.ai-section-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.ai-section-label {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-sec-finding .ai-section-label { color: #fda4af; }
.ai-sec-cause .ai-section-label   { color: #fde68a; }
.ai-sec-action .ai-section-label  { color: #6ee7b7; }

.ai-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #64748b;
    background: rgba(255, 255, 255, 0.04);
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.04em;
}
.ai-section-text {
    font-size: 0.88rem;
    color: #f1f5f9;
    line-height: 1.5;
}

/* Action Buttons Styling */
div.stButton > button {
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
}

/* Primary Action Button (Run Inspection) */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    border: none !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.55) !important;
}

/* Secondary Buttons (Quick Sample Chips & Tools) */
div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="baseButton-secondary"],
div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: rgba(30, 41, 59, 0.7) !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.76rem !important;
    padding: 0.35rem 0.65rem !important;
    min-height: 2.1rem !important;
    line-height: 1.2 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
div.stButton > button[kind="secondary"]:hover,
div.stButton > button[data-testid="baseButton-secondary"]:hover,
div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background: #1e293b !important;
    color: #38bdf8 !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* Audit KPI Summary Metric Cards */
.audit-kpi-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-top: 3px solid #6366f1;
    box-shadow: 0 4px 15px -2px rgba(0,0,0,0.4);
    margin-bottom: 0.5rem;
}
.audit-kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.audit-kpi-lbl {
    font-size: 0.68rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.audit-kpi-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Image Size & Aspect Ratio Constraints */
div[data-testid="stImage"] {
    text-align: center;
    display: flex;
    justify-content: center;
}
div[data-testid="stImage"] img {
    max-height: 380px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 8px !important;
    border: 1px solid #1e293b;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

/* Image Metadata Card */
.img-meta-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1.25rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.img-meta-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.img-meta-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.8rem;
}
.img-meta-key { color: #64748b; font-weight: 600; }
.img-meta-val { color: #cbd5e1; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* Streamlit Expander */
div[data-testid="stExpander"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    margin-top: 1rem;
}

/* Hide Streamlit Default Runner Spinner in Top Right */
div[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

/* AI Vision Processing Loader Card */
.ai-processing-card {
    background: linear-gradient(180deg, #0d1424 0%, #080c14 100%);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.25rem 0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}
.ai-processing-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.ai-processing-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ai-processing-step {
    font-size: 0.8rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-scan-bar {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.ai-scan-progress {
    width: 35%;
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #38bdf8, #34d399);
    border-radius: 4px;
    box-shadow: 0 0 14px #38bdf8;
    animation: ai-scan-anim 1.6s infinite ease-in-out;
}
@keyframes ai-scan-anim {
    0%   { transform: translateX(-100%); width: 25%; }
    50%  { width: 50%; }
    100% { transform: translateX(380%); width: 25%; }
}

/* Sample selector chip */
.sample-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0.6rem 0 1rem;
}
.sample-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CATEGORY CONFIGURATIONS & STATE
# -----------------------------------------------------------------------------
CATEGORIES = {
    "fabric": {
        "name": "Fabric Quality",
        "icon": "🧵",
        "desc": "Detect weaving defects, stains, holes, and texture anomalies in fabric rolls.",
        "ref_path": "data/reference_bank/fabric",
        "threshold": 0.35,
        "sample_names": ["Plain Weave", "Patterned Twill", "Knit Structure"],
    },
    "safety": {
        "name": "Worker Safety",
        "icon": "🪖",
        "desc": "Monitor PPE compliance — vests, helmets, and gloves on the factory floor.",
        "ref_path": "data/reference_bank/safety",
        "threshold": 0.40,
        "warning": "PPE tuning in progress — results may be experimental (see PROGRESS.md).",
        "sample_names": ["Safety Vest Frame", "Floor Inspector", "Worker Baseline"],
    },
    "machinery": {
        "name": "Machinery Wear",
        "icon": "⚙️",
        "desc": "Detect gear wear, cracks, and mechanical defects on sewing & cutting equipment.",
        "ref_path": "data/reference_bank/machinery",
        "threshold": 0.38,
        "info": "Proof of concept — using industrial proxy data (MVTec-AD).",
        "sample_names": ["Drive Gear Normal", "Shaft Bearing", "Motor Mount"],
    },
}

# Session state initialization
if "inspections_count" not in st.session_state:
    st.session_state.inspections_count = 0
if "anomalies_count" not in st.session_state:
    st.session_state.anomalies_count = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "current_image_bytes" not in st.session_state:
    st.session_state.current_image_bytes = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "last_inference_ms" not in st.session_state:
    st.session_state.last_inference_ms = None

# -----------------------------------------------------------------------------
# 3. BACKEND INTEGRATION FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5, show_spinner=False)
def check_backend_online() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False

def call_backend_infer(img_bytes: bytes, filename: str, category: str) -> dict:
    files = {"image": (filename, img_bytes, "image/png")}
    data = {"category": category}
    t0 = time.time()
    resp = requests.post(f"{BACKEND_URL}/infer", files=files, data=data, timeout=90)
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    res = resp.json()
    res["latency_ms"] = latency_ms
    return res

@st.cache_data(ttl=5, show_spinner=False)
def fetch_audit_logs() -> list:
    try:
        resp = requests.get(f"{BACKEND_URL}/logs", timeout=5)
        resp.raise_for_status()
        return resp.json().get("entries", [])
    except Exception:
        return []

def format_explanation(raw_text: str | None) -> dict:
    if not raw_text:
        return {
            "finding": "No abnormal features detected in visual feature map.",
            "cause": "Visual embedding is aligned with reference bank baseline cluster.",
            "action": "Pass batch to next production line stage.",
        }
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    return {
        "finding": lines[0] if len(lines) > 0 else raw_text,
        "cause": lines[1] if len(lines) > 1 else "Visual-semantic features deviate from standard baseline vectors.",
        "action": lines[2] if len(lines) > 2 else "Flag for supervisor review and physical batch verification.",
    }

is_backend_online = check_backend_online()

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & CATEGORY SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    # Brand
    st.markdown('<div class="brand-title">🧵 Stitch<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Industrial Anomaly Detection</div>', unsafe_allow_html=True)

    # Status indicator
    status_cls = "status-online" if is_backend_online else "status-offline"
    status_txt = "Backend Online" if is_backend_online else "Backend Offline"
    st.markdown(
        f'<div class="status-pill {status_cls}">'
        f'<div class="status-dot"></div>{status_txt}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Navigation Selection
    st.markdown('<div class="sb-section-title">Navigation</div>', unsafe_allow_html=True)
    active_nav = st.radio(
        "Navigation",
        options=["Run Inspection", "Audit Log"],
        index=0,
        label_visibility="collapsed",
    )

    # Category Selection
    st.markdown('<div class="sb-section-title">Inspection Category</div>', unsafe_allow_html=True)
    cat_keys = list(CATEGORIES.keys())
    selected_category_key = st.radio(
        "Inspection Category",
        options=cat_keys,
        format_func=lambda k: f"{CATEGORIES[k]['icon']} {CATEGORIES[k]['name']}",
        index=0,
        label_visibility="collapsed",
    )
    current_cat = CATEGORIES[selected_category_key]

    # Category Ready Card
    st.markdown(
        f'<div class="ready-card">'
        f'<div class="ready-badge">✦ Ready for Inspection</div>'
        f'<div class="ready-text">{current_cat["desc"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "warning" in current_cat:
        st.warning(current_cat["warning"], icon="⚠️")
    if "info" in current_cat:
        st.info(current_cat["info"], icon="ℹ️")

    # Session Stats
    st.markdown('<div class="sb-section-title">This Session</div>', unsafe_allow_html=True)
    total_insp = st.session_state.inspections_count
    anom_insp = st.session_state.anomalies_count
    rate_str = f"{(anom_insp / total_insp * 100):.0f}%" if total_insp > 0 else "—"

    st.markdown(
        f'<div class="session-metrics-grid">'
        f'<div class="session-metric-cell">'
        f'<div class="session-metric-val">{total_insp}</div>'
        f'<div class="session-metric-lbl">Inspections</div>'
        f'</div>'
        f'<div class="session-metric-cell">'
        f'<div class="session-metric-val anom">{anom_insp}</div>'
        f'<div class="session-metric-lbl">Anomalies</div>'
        f'</div>'
        f'<div class="session-metric-cell">'
        f'<div class="session-metric-val">{rate_str}</div>'
        f'<div class="session-metric-lbl">Rate</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Footer
    st.markdown(
        f'<div class="sb-footer">'
        f'Team <strong>HexaMind</strong> - UIU<br>'
        f'BCOLBD 2026 — AI Category<br>'
        f'{VERSION}'
        f'</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 5. VIEW 1: RUN INSPECTION
# -----------------------------------------------------------------------------
if active_nav == "Run Inspection":
    # Main Header
    st.markdown(
        f'<div class="main-header">'
        f'<div class="main-header-left">'
        f'<div class="main-icon">{current_cat["icon"]}</div>'
        f'<div>'
        f'<div class="main-title">{current_cat["name"]} Inspection</div>'
        f'<div class="main-subtitle">{current_cat["desc"]}</div>'
        f'</div>'
        f'</div>'
        f'<div class="tech-badge">⚡ Backbone: AnomalyCLIP-ViT-L/14@336</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Upload Section Header
    st.markdown(
        '<div class="section-bar"><span>📤 Upload Image for Inspection</span><span>MAX 200MB • JPG, PNG</span></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Upload 200MB per file • JPG, PNG",
    )

    if uploaded_file is not None:
        st.session_state.current_image_bytes = uploaded_file.getvalue()
        st.session_state.current_image_name = uploaded_file.name

    # Quick Sample Loader Bar
    ref_files = (
        glob.glob(os.path.join(current_cat["ref_path"], "*.png")) +
        glob.glob(os.path.join(current_cat["ref_path"], "*.jpg"))
    )
    if ref_files:
        st.markdown('<div class="sample-label">⚡ Or Quick Load a Sample from Reference Bank:</div>', unsafe_allow_html=True)
        sample_cols = st.columns(min(len(ref_files), 4))
        for idx, rf in enumerate(ref_files[:4]):
            with sample_cols[idx]:
                sample_name = os.path.basename(rf)
                if st.button(f"🔍 {sample_name}", key=f"quick_sample_{idx}", type="secondary", use_container_width=True):
                    with open(rf, "rb") as f:
                        st.session_state.current_image_bytes = f.read()
                        st.session_state.current_image_name = sample_name
                        st.session_state.last_result = None
                    st.rerun()

    # Inspection Action Button
    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        run_inspection = st.button(
            f"⚡ Run Inspection — {current_cat['name']}",
            type="primary",
            use_container_width=True,
        )

    # Perform Inference if clicked
    if run_inspection:
        if st.session_state.current_image_bytes is None:
            st.warning("Please upload an inspection image or select a sample before running.")
        elif not is_backend_online:
            st.error(f"Cannot connect to backend server at {BACKEND_URL}. Ensure it is running.")
        else:
            loader_box = st.empty()
            loader_box.markdown(
                f'<div class="ai-processing-card">'
                f'<div class="ai-processing-header">'
                f'<div class="ai-processing-title">⚡ AnomalyCLIP Vision Backbone</div>'
                f'<div class="ai-badge-live">INFERENCE IN PROGRESS</div>'
                f'</div>'
                f'<div class="ai-processing-step">🔬 Computing zero-shot feature distance maps & synthesizing anomaly heatmap overlay against {current_cat["name"]} baseline...</div>'
                f'<div class="ai-scan-bar"><div class="ai-scan-progress"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            try:
                res = call_backend_infer(
                    st.session_state.current_image_bytes,
                    st.session_state.current_image_name or "inspection.png",
                    selected_category_key,
                )
                loader_box.empty()
                st.session_state.last_result = res
                st.session_state.inspections_count += 1
                if res.get("verdict") == "anomalous":
                    st.session_state.anomalies_count += 1
                st.rerun()
            except Exception as e:
                loader_box.empty()
                st.error(f"Inference error: {e}")

    # Display Results or Empty State
    has_results = st.session_state.last_result is not None
    has_upload = st.session_state.current_image_bytes is not None

    if not has_upload and not has_results:
        # Initial empty state card
        st.markdown(
            f'<div class="empty-preview-box">'
            f'<div class="empty-icon">{current_cat["icon"]}</div>'
            f'<div class="empty-title">Upload an image to begin inspection</div>'
            f'<div class="empty-subtitle">Supported formats: JPG, JPEG, PNG • Category: {current_cat["name"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif has_upload and not has_results:
        # Pre-inspection preview
        st.markdown(
            '<div class="industrial-card">'
            f'<div class="section-bar"><span>📷 Captured Input Frame — {st.session_state.current_image_name}</span></div>',
            unsafe_allow_html=True,
        )
        c_prev_img, c_prev_meta = st.columns([1.1, 1], gap="medium")
        with c_prev_img:
            st.image(st.session_state.current_image_bytes, use_column_width=True)
        with c_prev_meta:
            file_kb = len(st.session_state.current_image_bytes) / 1024
            st.markdown(
                f'<div class="img-meta-card">'
                f'<div>'
                f'<div class="img-meta-title">📊 Frame Acquisition Data</div>'
                f'<div class="img-meta-item"><span class="img-meta-key">Filename</span><span class="img-meta-val">{st.session_state.current_image_name}</span></div>'
                f'<div class="img-meta-item"><span class="img-meta-key">File Size</span><span class="img-meta-val">{file_kb:.1f} KB</span></div>'
                f'<div class="img-meta-item"><span class="img-meta-key">Inspection Pipeline</span><span class="img-meta-val">{current_cat["name"]}</span></div>'
                f'<div class="img-meta-item"><span class="img-meta-key">Decision Threshold</span><span class="img-meta-val">{current_cat["threshold"]*100:.0f}%</span></div>'
                f'<div class="img-meta-item"><span class="img-meta-key">Status</span><span class="img-meta-val" style="color:#34d399;">Ready for Zero-Shot Inference</span></div>'
                f'</div>'
                f'<div style="font-size:0.75rem;color:#64748b;margin-top:1rem;line-height:1.4;">'
                f'Click the <strong>⚡ Run Inspection</strong> button above to process this capture through the AnomalyCLIP backbone.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    elif has_results:
        res = st.session_state.last_result
        verdict = res.get("verdict", "normal").lower()
        score = float(res.get("score", 0.0))
        is_anomaly = (verdict == "anomalous")
        latency = res.get("latency_ms", 0)

        hero_cls = "verdict-anomalous" if is_anomaly else "verdict-normal"
        badge_cls = "badge-anom" if is_anomaly else "badge-norm"
        badge_text = "DEFECT DETECTED" if is_anomaly else "QUALITY PASSED"
        hero_title = "ANOMALY DETECTED — HOLD BATCH" if is_anomaly else "NORMAL — CLEAR FOR PRODUCTION"
        hero_desc = (
            f"Confidence score {score*100:.1f}% exceeds category threshold ({current_cat['threshold']*100:.0f}%). Visual inspection recommended."
            if is_anomaly else
            f"Confidence score {score*100:.1f}% is within acceptable normal parameters (< {current_cat['threshold']*100:.0f}%)."
        )

        # Verdict Hero Banner
        st.markdown(
            f'<div class="verdict-hero {hero_cls}">'
            f'<div>'
            f'<div class="verdict-badge {badge_cls}">{badge_text}</div>'
            f'<div class="verdict-heading">{hero_title}</div>'
            f'<div class="verdict-desc">{hero_desc}</div>'
            f'<div style="font-size:0.7rem;color:#94a3b8;margin-top:0.4rem;font-family:JetBrains Mono,monospace;">⏱️ Latency: {latency} ms &nbsp;•&nbsp; Category: {selected_category_key.upper()}</div>'
            f'</div>'
            f'<div class="score-display">'
            f'<div class="score-title">Anomaly Score</div>'
            f'<div class="score-val">{score*100:.1f}%</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Visuals + Diagnostic Report Cards
        col_vis, col_diag = st.columns([1.25, 1], gap="medium")

        with col_vis:
            st.markdown(
                '<div class="industrial-card">'
                '<div class="section-bar"><span>🔬 Anomaly Localization Analysis</span></div>',
                unsafe_allow_html=True,
            )
            v_tab1, v_tab2 = st.tabs(["🖼️ Side-by-Side Comparison", "🌡️ Heatmap Only"])
            with v_tab1:
                v1, v2 = st.columns(2)
                with v1:
                    st.caption(f"Raw Input: {st.session_state.current_image_name}")
                    st.image(st.session_state.current_image_bytes, use_column_width=True)
                with v2:
                    st.caption("Anomaly Heatmap Overlay")
                    if res.get("heatmap_url"):
                        st.image(f"{BACKEND_URL}{res['heatmap_url']}", use_column_width=True)
                    else:
                        st.info("No heatmap layer needed for normal verdict.")
            with v_tab2:
                if res.get("heatmap_url"):
                    st.image(f"{BACKEND_URL}{res['heatmap_url']}", use_column_width=True)
                else:
                    st.info("Heatmap is clean. No localized anomalies detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_diag:
            exp = format_explanation(res.get("explanation"))
            severity_label = "CRITICAL DEFECT" if is_anomaly else "NOMINAL (PASS)"
            severity_color = "#fda4af" if is_anomaly else "#6ee7b7"
            
            st.markdown(
                f'<div class="ai-console">'
                f'<div class="ai-console-header">'
                f'<div class="ai-console-title">✦ AI Vision Diagnostic Engine</div>'
                f'<div class="ai-badge-live">VLM Reasoning Active</div>'
                f'</div>'
                f'<div class="ai-section ai-sec-finding">'
                f'<div class="ai-section-top">'
                f'<span class="ai-section-label">📌 Visual Observation</span>'
                f'<span class="ai-tag">FEATURE MAP ANALYSIS</span>'
                f'</div>'
                f'<div class="ai-section-text">{exp["finding"]}</div>'
                f'</div>'
                f'<div class="ai-section ai-sec-cause">'
                f'<div class="ai-section-top">'
                f'<span class="ai-section-label">🔍 Probable Root Cause</span>'
                f'<span class="ai-tag">ZERO-SHOT ATTRIBUTION</span>'
                f'</div>'
                f'<div class="ai-section-text">{exp["cause"]}</div>'
                f'</div>'
                f'<div class="ai-section ai-sec-action">'
                f'<div class="ai-section-top">'
                f'<span class="ai-section-label">⚡ Prescriptive Action</span>'
                f'<span class="ai-tag">OPERATIONAL DIRECTIVE</span>'
                f'</div>'
                f'<div class="ai-section-text">{exp["action"]}</div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.7rem;color:#64748b;font-family:JetBrains Mono,monospace;">'
                f'<span>STATUS: <strong style="color:{severity_color};">{severity_label}</strong></span>'
                f'<span>ENGINE: AnomalyCLIP</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            
            # Export Report Button
            report_dict = {
                "timestamp": res.get("timestamp", datetime.utcnow().isoformat()),
                "category": selected_category_key,
                "verdict": verdict,
                "score": score,
                "latency_ms": latency,
                "diagnostic": exp,
            }
            st.download_button(
                label="📥 Export AI Diagnostic JSON Report",
                data=json.dumps(report_dict, indent=2),
                file_name=f"stitchai_inspection_{int(time.time())}.json",
                mime="application/json",
                type="secondary",
                use_container_width=True,
            )

    # Reference Bank Section (collapsible expander at bottom)
    with st.expander(f"📁 Reference Bank ({len(ref_files)} standard baseline images)"):
        if ref_files:
            st.caption(f"Normal baseline standard images used for zero-shot inspection in category **{current_cat['name']}**:")
            cols = st.columns(min(len(ref_files), 5))
            for i, r_file in enumerate(ref_files[:5]):
                with cols[i]:
                    st.image(r_file, caption=os.path.basename(r_file), use_column_width=True)
        else:
            st.info(f"No baseline images found in {current_cat['ref_path']}.")

# -----------------------------------------------------------------------------
# 6. VIEW 2: AUDIT LOG
# -----------------------------------------------------------------------------
elif active_nav == "Audit Log":
    st.markdown(
        '<div class="main-header">'
        '<div class="main-header-left">'
        '<div class="main-icon">📊</div>'
        '<div>'
        '<div class="main-title">Inspection Audit Logs</div>'
        '<div class="main-subtitle">Centralized compliance history, threshold violations, and anomaly records across all factory lines.</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    logs = fetch_audit_logs()

    if not logs:
        st.markdown(
            '<div class="empty-preview-box">'
            '<div class="empty-icon">📋</div>'
            '<div class="empty-title">No Audit Logs Recorded</div>'
            '<div class="empty-subtitle">Run inspections in the workspace to record live inspection records.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(logs)
        total_logs = len(df)
        anomalies_logs = len(df[df["verdict"] == "anomalous"])
        pass_logs = total_logs - anomalies_logs
        defect_rate = f"{(anomalies_logs / total_logs * 100):.1f}%" if total_logs > 0 else "0.0%"
        avg_score = f"{df['score'].mean()*100:.1f}%" if total_logs > 0 else "0.0%"

        # 4 Responsive Metric Cards
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            st.markdown(
                f'<div class="audit-kpi-card" style="border-top-color:#6366f1;">'
                f'<div class="audit-kpi-val">{total_logs}</div>'
                f'<div class="audit-kpi-lbl">Total Inspections</div>'
                f'<div class="audit-kpi-sub">Across All Factory Lines</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kc2:
            st.markdown(
                f'<div class="audit-kpi-card" style="border-top-color:#f43f5e;">'
                f'<div class="audit-kpi-val" style="color:#fda4af;">{anomalies_logs}</div>'
                f'<div class="audit-kpi-lbl">Defects Flagged</div>'
                f'<div class="audit-kpi-sub" style="color:#fb7185;">{defect_rate} Defect Rate</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kc3:
            st.markdown(
                f'<div class="audit-kpi-card" style="border-top-color:#10b981;">'
                f'<div class="audit-kpi-val" style="color:#a7f3d0;">{pass_logs}</div>'
                f'<div class="audit-kpi-lbl">Quality Passed</div>'
                f'<div class="audit-kpi-sub" style="color:#34d399;">Within Baseline Tolerance</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kc4:
            st.markdown(
                f'<div class="audit-kpi-card" style="border-top-color:#38bdf8;">'
                f'<div class="audit-kpi-val" style="color:#7dd3fc;">{avg_score}</div>'
                f'<div class="audit-kpi-lbl">Mean Anomaly Score</div>'
                f'<div class="audit-kpi-sub">{len(df["category"].unique())} Active Categories</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Filter & Search Section
        st.markdown(
            '<div class="industrial-card" style="margin-top:0.75rem;">'
            '<div class="section-bar"><span>🔎 Filter & Compliance Search</span></div>',
            unsafe_allow_html=True,
        )
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.5], gap="small")
        with f1:
            cat_f = st.multiselect(
                "Category",
                options=list(CATEGORIES.keys()),
                default=list(CATEGORIES.keys()),
                format_func=lambda k: f"{CATEGORIES[k]['icon']} {CATEGORIES[k]['name']}",
            )
        with f2:
            verdict_f = st.multiselect("Verdict", options=["anomalous", "normal"], default=["anomalous", "normal"])
        with f3:
            min_score_f = st.slider("Min Anomaly Score", 0.0, 1.0, 0.0, 0.05)
        with f4:
            search_query = st.text_input("🔍 Search File / ID", placeholder="e.g. 001.png, tmp...", label_visibility="visible")
        st.markdown('</div>', unsafe_allow_html=True)

        # Filter dataframe
        filtered_df = df[
            (df["category"].isin(cat_f)) &
            (df["verdict"].isin(verdict_f)) &
            (df["score"] >= min_score_f)
        ].copy()

        if search_query:
            filtered_df = filtered_df[
                filtered_df["image_ref"].str.contains(search_query, case=False, na=False) |
                filtered_df["category"].str.contains(search_query, case=False, na=False) |
                filtered_df["verdict"].str.contains(search_query, case=False, na=False)
            ]

        # Action Toolbar
        col_ref, col_csv, col_count = st.columns([1.2, 1.6, 3])
        with col_ref:
            if st.button("🔄 Refresh Logs", type="secondary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_csv:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Filtered CSV",
                data=csv_data,
                file_name=f"stitchai_audit_logs_{int(time.time())}.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True,
            )
        with col_count:
            st.markdown(
                f'<div style="text-align:right;font-size:0.75rem;color:#64748b;padding-top:0.6rem;font-family:JetBrains Mono,monospace;">'
                f'SHOWING <strong style="color:#cbd5e1;">{len(filtered_df)}</strong> OF <strong style="color:#cbd5e1;">{total_logs}</strong> TOTAL RECORDS'
                f'</div>',
                unsafe_allow_html=True,
            )

        if not filtered_df.empty:
            display_df = filtered_df[["timestamp", "category", "verdict", "score", "image_ref"]].copy()
            
            # Format clean image filenames
            def clean_frame_name(p):
                if not p:
                    return "—"
                bname = os.path.basename(str(p))
                if bname.startswith("tmp") and len(bname) > 8:
                    return f"Frame_{bname[3:8]}.png"
                return bname

            display_df["image_ref"] = display_df["image_ref"].apply(clean_frame_name)
            
            # Format clean category badges
            cat_map = {"fabric": "🧵 Fabric Quality", "safety": "🪖 Worker Safety", "machinery": "⚙️ Machinery Wear"}
            display_df["category"] = display_df["category"].apply(lambda c: cat_map.get(c, c.title()))
            
            # Format clean verdict badges
            display_df["verdict"] = display_df["verdict"].apply(lambda v: "🚨 ANOMALOUS" if v == "anomalous" else "✅ NORMAL (PASS)")
            
            # Format score as clean percentage
            display_df["score"] = display_df["score"].apply(lambda s: f"{s*100:.1f}%")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)", format="YYYY-MM-DD HH:mm:ss"),
                    "category": st.column_config.TextColumn("Inspection Pipeline"),
                    "verdict": st.column_config.TextColumn("Verdict Status"),
                    "score": st.column_config.TextColumn("Anomaly Score"),
                    "image_ref": st.column_config.TextColumn("Captured Frame"),
                },
            )
        else:
            st.warning("No records match the current filter selection.")
