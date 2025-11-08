# clean_pace_v2.py
# Streamlit cleaner for Run Style (with RS_Avg 5 & RS_Avg10), Official Ratings/Prize splitting,
# Speed Blocks parsing, and merged CSV output.
# - RS_Cat is derived ONLY from RS_Avg (5)
# - Box B is a single paste (it can contain just Speed Blocks; optional Conditions are auto-detected and ignored)
# - Robust splitter prevents "0/10, 0/11..." lines from the OR section polluting the Run Style table

import re
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="CleanPace v2", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v2 — Run Style (5 & 10), Speed Blocks, Joined CSV")

st.markdown(
    """
**Box A:** Paste the whole Page 1 block (Run Style Figure + Official Ratings + Race Class/Prize).  
**Box B:** Paste the Speed Blocks (11-line format). If your source includes a Race Conditions (W/P/T) table beneath, paste it too — the app will detect it and ignore it for now.
"""
)

# -------------------------
# Helpers
# -------------------------
def _read_table_guess(text: str) -> pd.DataFrame:
    """Try TSV first, then CSV."""
    text = text.strip()
    try:
        return pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(StringIO(text))


def split_sections(raw: str):
    """
    Split Box A into:
      rs_text  -> 'Run Style Figure' section (table we want)
      or_text  -> 'Official Ratings' block (not parsed here)
      prize_text -> 'Race Class - Today' block (not parsed here)
    If anchors are missing, returns (raw, "", "") as a permissive fallback.
    """
    text = raw.strip().replace("\r\n", "\n").replace("\r", "\n")

    def idx_of(pattern):
        m = re.search(pattern, text, flags=re.I)
        return m.start() if m else -1

    rs_idx = idx_of(r"\bRun\s*Style\s*Figure\b")
    or_idx = idx_of(r"\bOfficial\s*Ratings\b")
    rc_idx = idx_of(r"\bRace\s*Class\s*-\s*Today\b")

    # Fallback for looser headings if needed
    if or_idx == -1:
        or_idx = idx_of(r"\bOfficial\b.*\bRatings\b")

    if rs_idx == -1 or or_idx == -1 or rc_idx == -1:
        # Could be the user pasted only the Run Style table
        return raw.strip(), "", ""

    rs_text = text[rs_idx:or_idx].strip()
    or_text = text[or_idx:rc_idx].strip()
    prize_text = text[rc_idx:].strip()
    return rs_text, or_text, prize_text


def rs_category(avg, front=1.6, prom=2.4, mid=3.0):
    if pd.isna(avg):
        return None
    if avg < front:
        return "Front"
    if avg < prom:
        return "Prominent"
    if avg < mid:
        return "Mid"
    return "Hold-up"


def parse_run_style_table_from_section(rs_text: str) -> pd.DataFrame:
    """
    Parse the 'Run Style Figure' table only.
    Computes:
      - RS_Avg   : last 5 (Lto1..Lto5), ignoring 0s
      - RS_Avg10 : last 10 (Lto1..Lto10), ignoring 0s
      - RS_Cat   : derived from RS_Avg (5) ONLY
    """
    lines = rs_text.splitlines()
    # If the first line is the title, drop it
    table = "\n".join(lines[1:]) if lines and lines[0].strip().lower().startswith("run style figure") else rs_text

    df = _read_table_guess(table)

    # Identify columns
    lto_cols_5 = [c for c in [f"Lto{i}" for i in range(1, 6)] if c in df.columns]
    lto_cols_10 = [c for c in [f"Lto{i}" for i in range(1, 11)] if c in df.columns]

    # Force numeric for Lto columns
    for c in lto_cols_10:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def mean_ignore_zero(vals):
        vals = [v for v in vals if pd.notna(v) and float(v) > 0]
        return round(sum(vals) / len(vals), 2) if vals else None

    df["RS_Avg"] = df.apply(lambda r: mean_ignore_zero([r[c] for c in lto_cols_5]), axis=1)
    df["RS_Avg10"] = df.apply(lambda r: mean_ignore_zero([r[c] for c in lto_cols_10]), axis=1)
    df["RS_Cat"] = df["RS_Avg"].apply(rs_category)

    keep = (
        ["Horse"]
        + lto_cols_10
        + ["RS_Avg", "RS_Avg10", "RS_Cat", "Mode 5", "Mode 10", "Total", "Mode All", "Draw", "Dr.%"]
    )
    keep = [c for c in keep if c in df.columns]
    # Normalise horse names
    if "Horse" in df.columns:
        df["Horse"] = df["Horse"].astype(str).strip()
    return df[keep]


def extract_speed_block(block_lines):
    """Speed block = 11 lines per horse. Extract core numbers."""
    horse = block_lines[0].strip()
    if len(block_lines) < 4:
        return horse, None, None, None, None, None

    # Line 4 typically carries the series of speed figures
    line = block_lines[3]
    try:
        nums = list(map(int, re.findall(r"\d+", line)))
        if not nums:
            return horse, None, None, None, None, None
        last = nums[-1]
        high = max(nums)
        avg3 = round(sum(nums[-3:]) / min(3, len(nums)), 1)
        avga = round(sum(nums) / len(nums), 1)
        key = round((last + high + avg3) / 3, 1)
        return horse, last, high, avg3, avga, key
    except Exception:
        return horse, None, None, None, None, None


def parse_speed_blocks(raw: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines and "horse" in lines[0].lower():
        lines = lines[1:]
    rows = []
    for i in range(0, len(lines), 11):
        block = lines[i : i + 11]
        if not block:
            continue
        rows.append(extract_speed_block(block))

    df = pd.DataFrame(
        rows, columns=["Horse", "LastRace", "Highest", "Avg3", "AvgAll", "KeySpeedAvg"]
    )
    if "Horse" in df.columns:
        df["Horse"] = df["Horse"].astype(str).str.strip()
    return df


def split_box_b(raw: str):
    """
    Separate Speed Blocks from an optional Conditions (W/P/T) table.
    Returns (speed_text, conditions_text_or_None).
    """
    txt = raw.strip()
    cond_idx = None
    for i, line in enumerate(txt.splitlines()):
        ll = line.lower()
        # crude header detection for conditions tables
        if ("win %" in ll and "dist" in ll) or ("class %" in ll and "+/-1" in ll):
            cond_idx = i
            break
    if cond_idx is None:
        return txt, None

    lines = txt.splitlines()
    speed = "\n".join(lines[:cond_idx]).strip()
    cond = "\n".join(lines[cond_idx - 1 :]).strip()  # include header row above
    return speed, cond


def download_df(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label,
        df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# -------------------------
# UI
# -------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("📄 Box A — Page 1 block")
    st.caption("Paste the full block containing Run Style Figure + Official Ratings + Race Class/Prize.")
    a_text = st.text_area("Paste Box A content:", height=360)

with colB:
    st.subheader("⚡ Box B — Page 2 data (single paste)")
    st.caption("Paste your Speed Blocks (11-line format). If a W/P/T table is present underneath, paste it too — it will be detected.")
    b_text = st.text_area("Paste Box B content:", height=360)

if st.button("🚀 Process"):
    if not a_text.strip():
        st.warning("Please paste Box A.")
        st.stop()
    if not b_text.strip():
        st.warning("Please paste Box B.")
        st.stop()

    # --- Split Box A and parse ONLY the Run Style section
    rs_text, or_text, prize_text = split_sections(a_text)

    rs_df = parse_run_style_table_from_section(rs_text)
    st.success("Run Style parsed")
    st.dataframe(rs_df, use_container_width=True)
    download_df(rs_df, "cleanpace_rs.csv", "💾 Download Run Style CSV")

    with st.expander("Detected 'Official Ratings' section (preview)"):
        st.text(or_text[:1000] if or_text else "(not found)")
    with st.expander("Detected 'Race Class - Today' section (preview)"):
        st.text(prize_text[:1000] if prize_text else "(not found)")

    # --- Box B: split speed vs optional conditions
    speed_text, cond_text = split_box_b(b_text)
    sp_df = parse_speed_blocks(speed_text)
    st.success("Speed Blocks parsed")
    st.dataframe(sp_df, use_container_width=True)
    download_df(sp_df, "cleanpace_speed.csv", "💾 Download Speed CSV")

    if cond_text:
        with st.expander("Optional Conditions (W/P/T) detected (raw preview)"):
            st.text(cond_text[:1500])

    # --- Merge by Horse
    rs_norm = rs_df.copy()
    sp_norm = sp_df.copy()
    if "Horse" in rs_norm.columns:
        rs_norm["Horse"] = rs_norm["Horse"].astype(str).str.strip()
    if "Horse" in sp_norm.columns:
        sp_norm["Horse"] = sp_norm["Horse"].astype(str).str.strip()

    joined = pd.merge(rs_norm, sp_norm, on="Horse", how="outer")

    st.markdown("### 🧩 Joined Output (Run Style + Speed)")
    st.dataframe(joined, use_container_width=True)
    download_df(joined, "cleanpace_v2_joined.csv", "💾 Download Joined CSV")

st.caption(
    "Notes: RS_Avg = last 5 (Lto1–Lto5, ignoring zeros); RS_Avg10 = last 10 (display). RS_Cat derives only from RS_Avg (5). "
    "If you see '0/10' etc. in the Horse column, it means the Box A splitter didn't run — ensure the page contains the 'Run Style Figure' heading."
)
