from __future__ import annotations
import re
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# App Setup
# =========================
st.set_page_config(
    page_title="CleanPace v3.1.1 — Backbone + Diagnostics",
    page_icon="🏇",
    layout="wide"
)
st.title("🏇 CleanPace v3.1.1 — Pace/Speed Backbone + Diagnostics")

# =========================
# Helpers
# =========================
def _read_table_guess(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(StringIO(text))

def _mean_ignore_zero(values):
    vals = [float(v) for v in values if pd.notna(v) and float(v) > 0]
    return round(sum(vals) / len(vals), 2) if vals else None

def _rs_category(avg):
    if avg is None or np.isnan(avg):
        return None
    if avg < 1.6: return "Front"
    if avg < 2.4: return "Prominent"
    if avg < 3.0: return "Mid"
    return "Hold-up"

# 🔑 FIX: ignore anything in brackets
def _numbers_excluding_brackets(s: str) -> List[float]:
    if s is None:
        return []
    s = str(s)
    s = re.sub(r"\([^)]*\)", "", s)  # remove (avg)
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]

# =========================
# Parse Run Style (Box A)
# =========================
def parse_run_style(text):
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("run style"):
        text = "\n".join(lines[1:])
    df = _read_table_guess(text)
    if "Horse" not in df.columns:
        raise ValueError("Run Style block must contain Horse column")

    df["Horse"] = df["Horse"].astype(str).str.strip()
    lto = [c for c in df.columns if c.lower().startswith("lto")]
    for c in lto:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["RS_Avg"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto[:5]]), axis=1)
    df["RS_Avg10"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto]), axis=1)
    df["Style"] = df["RS_Avg"].apply(_rs_category)

    return df[["Horse", "RS_Avg", "RS_Avg10", "Style"]]

# =========================
# Parse Box B (Speed + Conditions)
# =========================
BOX_ORDER = ["win", "form", "speed_series", "crs", "dist", "lhrh", "going", "cls", "runpm1", "trackstyle"]

def _is_name(lines, i):
    t = lines[i].strip()
    if not t or re.search(r"[0-9(),/£]", t): 
        return False
    return "(" in lines[i+1] if i+1 < len(lines) else False

def _parse_wpt(s):
    m = re.search(r"\((\d+)/(\d+)/(\d+)\)", str(s))
    if not m: return None
    w, p, t = map(int, m.groups())
    return None if t == 0 else round((w+p)/t, 3)

def parse_box_b(raw):
    lines = [l for l in raw.splitlines() if l.strip()]
    rows, i = [], 0
    while i < len(lines):
        if _is_name(lines, i):
            horse = lines[i].strip()
            vals = lines[i+1:i+11]
            rows.append(dict(zip(["Horse"]+BOX_ORDER, [horse]+vals)))
            i += 11
        else:
            i += 1
    return pd.DataFrame(rows)

def build_speed_conditions(df):
    out = df.copy()
    out["SpeedRunsRaw"] = out["speed_series"].fillna("").astype(str)

    runs_list, last, high, avg3, avgall, keyavg = [], [], [], [], [], []

    for s in out["SpeedRunsRaw"]:
        nums = _numbers_excluding_brackets(s)
        runs_list.append(", ".join(map(str, nums)))
        if nums:
            last.append(nums[-1])
            high.append(max(nums))
            avg3.append(round(np.mean(nums[-3:]), 1))
            avgall.append(round(np.mean(nums), 1))
            keyavg.append(round((nums[-1] + max(nums) + np.mean(nums[-3:])) / 3, 1))
        else:
            last.append(None); high.append(None)
            avg3.append(None); avgall.append(None); keyavg.append(None)

    out["SpeedRunsList"] = runs_list
    out["LastRace"] = last
    out["Highest"] = high
    out["Avg3"] = avg3
    out["AvgAll"] = avgall
    out["AdjSpeed"] = keyavg

    for c in ["crs","dist","lhrh","going","cls","runpm1","trackstyle"]:
        out[f"{c}_place"] = out[c].apply(_parse_wpt)

    return out[[
        "Horse","SpeedRunsRaw","SpeedRunsList","AdjSpeed",
        "crs_place","dist_place","lhrh_place","going_place",
        "cls_place","runpm1_place","trackstyle_place"
    ]]

# =========================
# Pace / Suitability
# =========================
PACEFIT = {
    "Slow":   {"Front":5,"Prominent":4,"Mid":3,"Hold-up":2},
    "Even":   {"Front":4,"Prominent":5,"Mid":4,"Hold-up":3},
    "Strong": {"Front":2,"Prominent":3,"Mid":4,"Hold-up":5},
}

def speed_fit(dvp):
    if dvp >= 2: return 5
    if dvp >= -1: return 4
    if dvp >= -4: return 3
    if dvp >= -8: return 2
    return 1

def compute_suit(df, pace_map, wp, ws):
    return df["Style"].map(pace_map).fillna(3)*wp + df["SpeedFit"]*ws

# =========================
# UI
# =========================
tab1, tab2 = st.tabs(["Inputs", "Backbone + Diagnostics"])

# ---------- TAB 1 ----------
with tab1:
    boxA = st.text_area("Box A — Run Style Figure", height=220)
    boxB = st.text_area("Box B — Speed + Conditions", height=300)

    if st.button("Process & Build CSV"):
        rs = parse_run_style(boxA)
        b_raw = parse_box_b(boxB)
        b = build_speed_conditions(b_raw)

        merged = rs.merge(b, on="Horse", how="outer")
        st.dataframe(merged, use_container_width=True)
        st.download_button("Download CSV", merged.to_csv(index=False), "cleanpace_v3.csv")

# ---------- TAB 2 ----------
with tab2:
    upl = st.file_uploader("Upload Combined CSV", type="csv")
    par = st.number_input("Par Speed", value=77.0)
    wp = st.slider("Pace weight", 0.4, 0.7, 0.5)
    ws = 1 - wp

    if upl:
        df = pd.read_csv(upl)
        df["ΔvsPar"] = df["AdjSpeed"] - par
        df["SpeedFit"] = df["ΔvsPar"].apply(speed_fit)

        df["Suit_Slow"] = compute_suit(df, PACEFIT["Slow"], wp, ws)
        df["Suit_Even"] = compute_suit(df, PACEFIT["Even"], wp, ws)
        df["Suit_Strong"] = compute_suit(df, PACEFIT["Strong"], wp, ws)
        df["Suitability"] = df["Suit_Even"]

        # =========================
        # Speed Consistency (FIXED)
        # =========================
        totals, atpar = [], []
        for s in df["SpeedRunsRaw"]:
            nums = _numbers_excluding_brackets(s)
            totals.append(len(nums))
            atpar.append(sum(1 for x in nums if x >= par))

        df["Speed_TotalRuns"] = totals
        df["Speed_RunsAtPar"] = atpar
        df["Speed_ParPct"] = df["Speed_RunsAtPar"] / df["Speed_TotalRuns"]

        df["SpeedConsistency"] = pd.cut(
            df["Speed_ParPct"],
            [-1,0.15,0.30,0.45,0.60,1],
            labels=[1,2,3,4,5]
        ).astype(float)

        st.subheader("Pace Sensitivity + Diagnostics")
        st.dataframe(
            df[[
                "Horse","Style","AdjSpeed","ΔvsPar",
                "Suit_Slow","Suit_Even","Suit_Strong",
                "Speed_TotalRuns","Speed_RunsAtPar","Speed_ParPct","SpeedConsistency"
            ]].sort_values("Suit_Even", ascending=False),
            use_container_width=True
        )
