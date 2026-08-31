"""
Streamlit upload UI: upload -> verdict/heatmap/explanation display, plus a unified
audit log view spanning all categories (Phase 5).
"""
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="StitchAI", page_icon=":shirt:")
st.title("StitchAI — RMG Anomaly Detection")
st.caption("Team HexaMind | BCOLBD 2026 — AI Category")

# Phase 0 hello-world check
with st.expander("Backend connection status"):
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        st.success(f"Backend reachable: {r.json()}")
    except Exception as e:
        st.error(f"Backend unreachable at {BACKEND_URL}: {e}")

tab_inspect, tab_logs = st.tabs(["Run Inspection", "Audit Log"])

with tab_inspect:
    category = st.selectbox(
        "Inspection category",
        ["fabric", "safety", "machinery"],
        help='"machinery" is a proof-of-concept category using proxy (MVTec-AD) data — see README. "safety" is currently parked — see PROGRESS.md.',
    )
    if category == "machinery":
        st.info("Proof of concept — proxy industrial data (MVTec-AD), not RMG-specific.", icon="ℹ️")
    if category == "safety":
        st.warning(
            "This category is currently parked — the shared backbone doesn't yet "
            "reliably distinguish PPE compliance. See PROGRESS.md for details.",
            icon="⚠️",
        )

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

        if st.button("Run inspection"):
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"category": category}
            try:
                resp = requests.post(f"{BACKEND_URL}/infer", files=files, data=data, timeout=60)
                resp.raise_for_status()
                result = resp.json()

                verdict = result["verdict"]
                score = result["score"]

                if verdict == "anomalous":
                    st.error(f"Verdict: ANOMALOUS (confidence {score:.2f})")
                else:
                    st.success(f"Verdict: NORMAL (confidence {score:.2f})")

                if result.get("heatmap_url"):
                    st.image(f"{BACKEND_URL}{result['heatmap_url']}", caption="Anomaly heatmap")

                if result.get("explanation"):
                    st.write("**Explanation:**", result["explanation"])

            except requests.exceptions.HTTPError as e:
                st.warning(f"Backend returned an error: {e}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_logs:
    st.subheader("Unified inspection log")
    st.caption("One audit trail spanning every category — fabric, safety, and machinery alike.")

    if st.button("Refresh log"):
        st.rerun()

    try:
        resp = requests.get(f"{BACKEND_URL}/logs", timeout=10)
        resp.raise_for_status()
        entries = resp.json().get("entries", [])

        if not entries:
            st.info("No inspections logged yet — run one from the 'Run Inspection' tab.")
        else:
            df = pd.DataFrame(entries)
            df = df[["timestamp", "category", "verdict", "score", "image_ref"]]
            df["score"] = df["score"].round(4)
            st.dataframe(df, use_container_width=True, hide_index=True)

            counts = df["category"].value_counts()
            st.caption(
                "Entries by category: "
                + ", ".join(f"{cat}: {n}" for cat, n in counts.items())
            )

    except Exception as e:
        st.error(f"Could not load logs: {e}")
