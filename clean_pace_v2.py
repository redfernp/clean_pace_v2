# clean_pace_v2.py
# Combined cleaner for Run Style, Official Ratings, Race Class, and Speed Blocks (+ optional Conditions)
# v2.0 – RS_Avg uses last 5 runs only; RS_Avg10 shown for comparison

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO

st.set_page_config(page_title="CleanPace v2", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v2 — Unified Parser for Run Style & Speed Data")

st.markdown(
"""
### 📘 How to use
**Box A:** Paste your “Run Style + Official Ratings + Race Class” table.  
**Box B:** Paste your “Speed Figures” blocks.  
If your source page also includes the **Race Conditions (W/P/T)** table, just paste it underneath — the app will detect it automatically.
"""
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _read_table_guess(text: str) -> pd.DataFrame:
    """Try TSV → CSV autodetect."""
    try:
        return pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(StringIO(text))

def rs_category(avg, front=1.6, prom=2.4, mid=3.0):
    if pd.isna(avg): return None
    if avg < front: return "Front"
    if avg < prom:  return "Prominent"
    if avg < mid:   return "Mid"
    return "Hold-up"

def parse_run_style_table_from_section(rs_text: str) -> pd.DataFrame:
    """Extract Run Style section + compute RS_Avg (5) and RS_Avg10."""
    lines = rs_text.splitlines()
    table = "\n".join(lines[1:]) if lines and lines[0].strip().lower().startswith("run style figure") else rs_text
    df = _read_table_guess(table)

    lto_cols_5  = [c for c in [f"Lto{i}" for i in range(1,6)]  if c in df.columns]
    lto_cols_10 = [c for c in [f"Lto{i}" for i in range(1,11)] if c in df.columns]

    for c in lto_cols_10:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def mean_ignore_zero(vals):
        vals = [v for v in vals if pd.notna(v) and float(v) > 0]
        return round(sum(vals)/len(vals), 2) if vals else None

    df["RS_Avg"]   = df.apply(lambda r: mean_ignore_zero([r[c] for c in lto_cols_5]), axis=1)
    df["RS_Avg10"] = df.apply(lambda r: mean_ignore_zero([r[c] for c in lto_cols_10]), axis=1)
    df["RS_Cat"]   = df["RS_Avg"].apply(rs_category)

    keep = ["Horse"] + lto_cols_10 + ["RS_Avg","RS_Avg10","RS_Cat",
             "Mode 5","Mode 10","Total","Mode All","Draw","Dr.%"]
    return df[[c for c in keep if c in df.columns]]

def extract_speed_block(block):
    horse = block[0].strip()
    if len(block) < 4: return horse, None, None, None, None
    try:
        nums = list(map(int, re.findall(r"\d+", block[3])))
        if not nums: return horse, None, None, None, None
        last = nums[-1]
        high = max(nums)
        avg3 = round(sum(nums[-3:])/len(nums[-3:]),1)
        avgall = round(sum(nums)/len(nums),1)
        return horse, last, high, avg3, avgall
    except Exception:
        return horse, None, None, None, None

def parse_speed_blocks(raw: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if lines and "horse" in lines[0].lower(): lines = lines[1:]
    horses=[]
    for i in range(0,len(lines),11):
        block = lines[i:i+11]
        if block: horses.append(extract_speed_block(block))
    df = pd.DataFrame(horses,columns=["Horse","LastRace","Highest","Avg3","AvgAll"])
    df["KeySpeedAvg"]=df[["LastRace","Highest","Avg3"]].mean(axis=1)
    return df

def split_box_b(raw: str):
    """Auto-detect optional Conditions (W/P/T) section."""
    txt = raw.strip()
    cond_idx=None
    for i,l in enumerate(txt.splitlines()):
        ll=l.lower()
        if ("win %" in ll and "dist" in ll) or ("class %" in ll and "+/-1" in ll):
            cond_idx=i;break
    if cond_idx is None: return txt,None
    lines=txt.splitlines()
    return "\n".join(lines[:cond_idx]).strip(), "\n".join(lines[cond_idx-1:]).strip()

def download_df(df: pd.DataFrame, name: str, label: str):
    st.download_button(label, df.to_csv(index=False), file_name=name,
                       mime="text/csv", use_container_width=True)

# ---------------------------------------------------------------------
# UI – Two Boxes
# ---------------------------------------------------------------------
boxA, boxB = st.columns(2)

with boxA:
    st.subheader("📄 Box A — Run Style + Official Ratings + Race Class tables")
    a_text = st.text_area("Paste all Box A content here:", height=400)
with boxB:
    st.subheader("⚡ Box B — Speed Blocks (+ optional Race Conditions)")
    b_text = st.text_area("Paste Speed Blocks (and optional W/P/T table) here:", height=400)

if st.button("🚀 Process Both"):
    if not a_text.strip() or not b_text.strip():
        st.warning("Please paste content in both boxes.")
        st.stop()

    # --- Box A ---
    rs_df = parse_run_style_table_from_section(a_text)
    st.success("Parsed Run Style table")
    st.dataframe(rs_df, use_container_width=True)
    download_df(rs_df,"cleanpace_rs.csv","💾 Download Run Style CSV")

    # --- Box B ---
    sp_text, cond_text = split_box_b(b_text)
    sp_df = parse_speed_blocks(sp_text)
    st.success("Parsed Speed Blocks")
    st.dataframe(sp_df, use_container_width=True)
    download_df(sp_df,"cleanpace_speed.csv","💾 Download Speed CSV")

    if cond_text:
        st.info("Detected Race Conditions (W/P/T) section — parsed separately for reference.")
        cond_lines = cond_text.splitlines()
        # simple preview
        st.text("\n".join(cond_lines[:15]))
        with st.expander("Full Conditions Text"):
            st.text(cond_text)

    # --- Join by Horse name ---
    rs_norm = rs_df.rename(columns={"Horse":"Horse"})
    sp_norm = sp_df.rename(columns={"Horse":"Horse"})
    rs_norm["Horse"]=rs_norm["Horse"].astype(str).str.strip()
    sp_norm["Horse"]=sp_norm["Horse"].astype(str).str.strip()
    merged=pd.merge(rs_norm,sp_norm,on="Horse",how="outer")
    st.markdown("### 🧩 Joined Output (merged by Horse)")
    st.dataframe(merged,use_container_width=True)
    download_df(merged,"cleanpace_v2_joined.csv","💾 Download Joined CSV")

st.caption("CleanPace v2 — RS_Avg (5-run basis) drives RS_Cat; RS_Avg10 shown for comparison. Optional Conditions (W/P/T) auto-detected from Box B.")
