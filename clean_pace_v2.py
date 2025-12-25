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
st.set_page_config(page_title="CleanPace v3 — Pace/Speed + Diagnostics", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v3 — Pace/Speed Backbone + Diagnostics (Conditions + Consistency)")

# =========================
# Helpers
# =========================
def _read_table_guess(text: str) -> pd.DataFrame:
    text = text.strip()
    try:
        return pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(StringIO(text))

def _mean_ignore_zero(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if pd.notna(v) and float(v) > 0]
    return round(sum(vals) / len(vals), 2) if vals else None

def _rs_category_from_value(avg: Optional[float], front_thr=1.6, prom_thr=2.4, mid_thr=3.0) -> Optional[str]:
    if avg is None or (isinstance(avg, float) and np.isnan(avg)):
        return None
    if avg < front_thr: return "Front"
    if avg < prom_thr:  return "Prominent"
    if avg < mid_thr:   return "Mid"
    return "Hold-up"

def _numbers(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", str(s))]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def map_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column whose normalized name matches a candidate."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None

# =========================
# Parse Run Style (Box A) — RS only (no OR/Class)
# =========================
def parse_run_style(rs_text: str, front_thr=1.6, prom_thr=2.4, mid_thr=3.0) -> pd.DataFrame:
    lines = rs_text.splitlines()
    if lines and lines[0].strip().lower().startswith("run style figure"):
        rs_text = "\n".join(lines[1:])
    df = _read_table_guess(rs_text)
    if "Horse" not in df.columns:
        raise ValueError("Run Style: missing 'Horse' column.")
    df["Horse"] = df["Horse"].astype(str).str.strip()

    lto5 = [c for c in [f"Lto{i}" for i in range(1, 6)] if c in df.columns]
    lto10 = [c for c in [f"Lto{i}" for i in range(1, 11)] if c in df.columns]
    for c in lto10:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "RS_Avg" not in df.columns:
        df["RS_Avg"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto5]), axis=1)
    if "RS_Avg10" not in df.columns:
        df["RS_Avg10"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto10]), axis=1)

    df["RS_Cat"] = df["RS_Avg"].apply(lambda x: _rs_category_from_value(x, front_thr, prom_thr, mid_thr))

    keep = ["Horse"] + lto10 + ["RS_Avg", "RS_Avg10", "RS_Cat"] + \
           [c for c in ["Mode 5", "Mode 10", "Total", "Mode All", "Draw", "Dr.%"] if c in df.columns]
    return df[keep]

# =========================
# Boxes 4 & 5 (Box B) parsing — keep raw speed series & raw condition lines
# =========================
BOX_ORDER = ["win_pct", "form_avg", "speed_series", "crs", "dist", "lhrh", "going", "cls", "runpm1", "trackstyle"]

def _is_win_line(s: str) -> bool:
    return bool(re.search(r"\(\s*\d+\s*/\s*\d+\s*/\s*\d+\s*\)", str(s)))

def _looks_like_name(lines: List[str], idx: int) -> bool:
    t = lines[idx].strip()
    if not t or t.lower().startswith("horse"):
        return False
    if re.search(r"[0-9(),/£]", t):
        return False
    j = idx + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    return _is_win_line(lines[j]) if j < len(lines) else False

def _parse_wpt_value(val: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    val = str(val).strip()
    m = re.search(r"\((\d+)\s*/\s*(\d+)\s*/\s*(\d+)\)", val)
    if not m:
        return None, None, None, None
    w = int(m.group(1)); p = int(m.group(2)); t = int(m.group(3))
    place = None if t == 0 else round((w + p) / t, 3)
    return place, w, p, t

def _parse_speed_series(s: str) -> List[float]:
    nums = _numbers(s)
    return nums

def parse_boxB_single(raw: str) -> pd.DataFrame:
    lines = [ln for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["Horse"])
    if lines[0].strip().lower().startswith("horse"):
        lines = lines[1:]

    i, n = 0, len(lines)
    recs: List[Dict[str, object]] = []
    while i < n:
        while i < n and not _looks_like_name(lines, i):
            i += 1
        if i >= n:
            break
        name = lines[i].strip()
        i += 1

        values: List[str] = []
        while i < n and len(values) < 10:
            if _looks_like_name(lines, i):
                break
            values.append(lines[i].strip())
            i += 1
        values += [None] * (10 - len(values)) if len(values) < 10 else []

        data = {"Horse": name}
        for key, val in zip(BOX_ORDER, values):
            data[key] = val
        recs.append(data)

    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["Horse"] = df["Horse"].astype(str).str.strip()
    return df

def build_speed_conditions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Produces:
      - SpeedRunsRaw (verbatim speed_series string)
      - SpeedRunsList (parsed list as a string for transparency)
      - KeySpeedAvg (your existing 'Adjusted Speed' proxy)
      - *_place columns for conditions
    """
    if df_raw.empty:
        return df_raw.copy()

    out = df_raw.copy()
    out["SpeedRunsRaw"] = out["speed_series"].fillna("").astype(str)

    speed_lists = []
    last_list, high_list, avg3_list, all_list, keyavg_list = [], [], [], [], []

    for s in out["SpeedRunsRaw"]:
        nums = _parse_speed_series(s)
        speed_lists.append(nums)

        if not nums:
            last_list.append(None); high_list.append(None); avg3_list.append(None); all_list.append(None); keyavg_list.append(None)
            continue

        last = nums[-1]
        highest = max(nums)
        avg3 = round(sum(nums[-3:]) / min(3, len(nums)), 1)
        avgall = round(sum(nums) / len(nums), 1)

        last_list.append(last)
        high_list.append(highest)
        avg3_list.append(avg3)
        all_list.append(avgall)
        keyavg_list.append(round((last + highest + avg3) / 3, 1) if (last is not None and highest is not None and avg3 is not None) else None)

    out["SpeedRunsList"] = [", ".join([str(int(x)) if float(x).is_integer() else str(x) for x in lst]) if lst else "" for lst in speed_lists]
    out["LastRace"] = last_list
    out["Highest"] = high_list
    out["Avg3"] = avg3_list
    out["AvgAll"] = all_list
    out["KeySpeedAvg"] = keyavg_list

    for col in ["crs", "dist", "lhrh", "going", "cls", "runpm1", "trackstyle"]:
        if col in out.columns:
            out[f"{col}_place"] = [_parse_wpt_value(v)[0] for v in out[col].fillna("")]

    keep = [
        "Horse",
        "SpeedRunsRaw", "SpeedRunsList",
        "LastRace", "Highest", "Avg3", "AvgAll", "KeySpeedAvg",
        "crs_place", "dist_place", "lhrh_place", "going_place", "cls_place", "runpm1_place", "trackstyle_place",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

# =========================
# Pace engine (unchanged backbone)
# =========================
@dataclass
class HorseRow:
    horse: str
    style_cat: Optional[str]          # Front/Prominent/Mid/Hold-up
    adj_speed: Optional[float]        # AdjSpeed (KeySpeedAvg)
    dvp: Optional[float]              # Δ vs Par
    lcp: str                          # High/Questionable/Unlikely/N/A

@dataclass
class Settings:
    class_par: float = 77.0
    distance_f: float = 7.0
    lcp_high: float = -3.0
    lcp_question_low: float = -8.0
    wp_even: float = 0.5
    wp_confident: float = 0.65

PACEFIT = {
    "Slow":   {"Front": 5, "Prominent": 4, "Mid": 3, "Hold-up": 2},
    "Even":   {"Front": 4, "Prominent": 5, "Mid": 4, "Hold-up": 3},
    "Strong": {"Front": 2, "Prominent": 3, "Mid": 4, "Hold-up": 5},
    "Very Strong": {"Front": 1, "Prominent": 2, "Mid": 4, "Hold-up": 5},
}
PACEFIT_EVEN_FC = {"Front": 4.5, "Prominent": 5.0, "Mid": 3.5, "Hold-up": 2.5}
PACEFIT_5F = {
    "Slow":   {"Front": 5.0, "Prominent": 4.5, "Mid": 3.5, "Hold-up": 2.0},
    "Even":   {"Front": 4.5, "Prominent": 5.0, "Mid": 3.5, "Hold-up": 2.5},
    "Strong": {"Front": 3.5, "Prominent": 4.5, "Mid": 4.0, "Hold-up": 3.0},
}
PACEFIT_6F = {
    "Slow":   {"Front": 5.0, "Prominent": 4.5, "Mid": 3.5, "Hold-up": 2.5},
    "Even":   {"Front": 4.0, "Prominent": 5.0, "Mid": 4.0, "Hold-up": 3.0},
    "Strong": {"Front": 3.0, "Prominent": 4.0, "Mid": 4.5, "Hold-up": 3.5},
}

def _dist_band(d: float) -> str:
    if d <= 5.5: return "5f"
    if d <= 6.5: return "6f"
    return "route"

def _lcp_from_dvp(style_cat: Optional[str], dvp: Optional[float], s: Settings) -> str:
    if style_cat not in ("Front", "Prominent") or dvp is None:
        return "N/A"
    if dvp >= s.lcp_high:
        return "High"
    if dvp >= s.lcp_question_low:
        return "Questionable"
    return "Unlikely"

def project_pace_from_rows(rows: List[HorseRow], s: Settings) -> Tuple[str, float, Dict[str, str], Dict[str, object]]:
    debug = {"rules_applied": []}
    if not rows:
        return "N/A", 0.0, {}, {"error": "no rows"}

    d = pd.DataFrame([{"horse": r.horse, "style": r.style_cat, "dvp": r.dvp, "lcp": r.lcp} for r in rows])

    front_all = d[d["style"] == "Front"]
    prom_all  = d[d["style"] == "Prominent"]
    front_high = d[(d["style"] == "Front") & (d["lcp"] == "High")]
    prom_high  = d[(d["style"] == "Prominent") & (d["lcp"] == "High")]
    front_q    = d[(d["style"] == "Front") & (d["lcp"] == "Questionable")]
    prom_q     = d[(d["style"] == "Prominent") & (d["lcp"] == "Questionable")]

    n_front, n_fh, n_ph, n_fq, n_pq = len(front_all), len(front_high), len(prom_high), len(front_q), len(prom_q)

    W_FH, W_PH, W_FQ, W_PQ = 2.0, 0.8, 0.5, 0.2
    energy = (W_FH * n_fh) + (W_PH * n_ph) + (W_FQ * n_fq) + (W_PQ * n_pq)

    band = _dist_band(s.distance_f)

    # Decision tree (as you had it)
    if n_fh >= 2:
        scenario, conf = "Strong", 0.65; debug["rules_applied"].append("≥2 High Front → Strong")
    elif (n_fh + n_ph) >= 3 and energy >= 3.2:
        scenario, conf = "Strong", 0.65; debug["rules_applied"].append("≥3 High early & energy≥3.2 → Strong")
    elif energy >= 3.2:
        scenario, conf = "Strong", 0.60; debug["rules_applied"].append("energy≥3.2 → Strong")
    elif (n_fh + n_ph) >= 2:
        scenario, conf = "Even", 0.60; debug["rules_applied"].append("≥2 High early → Even")
    elif (n_fh + n_ph) == 1 and (n_fq + n_pq) >= 1:
        scenario, conf = "Even", 0.55; debug["rules_applied"].append("1 High + Questionables → Even")
    elif (n_fh + n_ph) == 1:
        scenario, conf = "Even", 0.60; debug["rules_applied"].append("1 High → Even")
    elif (n_fq + n_pq) >= 1:
        scenario, conf = "Slow", 0.60; debug["rules_applied"].append("Only Questionables → Slow")
    else:
        scenario, conf = "Slow", 0.70; debug["rules_applied"].append("No credible early → Slow")

    if n_front == 0 or n_fh == 0:
        allow_strong = False
        if n_ph >= 3:
            try:
                allow_strong = float(prom_high["dvp"].mean()) >= -1.0
            except Exception:
                allow_strong = False
        if not allow_strong and scenario in ("Strong", "Very Strong"):
            scenario, conf = "Even", min(conf, 0.60); debug["rules_applied"].append("No-front cap → Even")

    if band in ("5f", "6f") and n_fh == 1 and n_ph <= 2:
        try:
            lf = float(front_high["dvp"].iloc[0])
        except Exception:
            lf = None
        dvp_ok = -0.5 if band == "5f" else -1.0
        energy_cap = 4.0 if band == "5f" else 3.6
        if (lf is not None) and (lf >= dvp_ok) and (energy < energy_cap) and scenario in ("Strong", "Very Strong"):
            scenario, conf = "Even", max(conf, 0.65); debug["rules_applied"].append("Sprint cap: Strong→Even")

    if band == "route" and scenario == "Even" and n_fh == 1 and n_ph <= 1:
        try:
            lf = float(front_high["dvp"].iloc[0])
        except Exception:
            lf = None
        if (lf is not None) and (lf >= -1.0):
            scenario = "Even (Front-Controlled)"
            debug["rules_applied"].append("Front-controlled tag")

    debug.update({
        "counts": {"Front_all": int(n_front), "Front_High": int(n_fh), "Prominent_High": int(n_ph)},
        "early_energy": float(energy),
        "distance_band": band
    })
    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario, conf, lcp_map, debug

# =========================
# Diagnostics (NEW, SEPARATE)
# =========================
COND_COLS = ["dist_place", "cls_place", "runpm1_place", "trackstyle_place", "crs_place", "lhrh_place", "going_place"]

def conditions_score_from_mean(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x >= 0.40: return 5.0
    if x >= 0.30: return 4.0
    if x >= 0.20: return 3.0
    if x >= 0.10: return 2.0
    return 1.0

def consistency_score_from_ratio(r: Optional[float]) -> Optional[float]:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return None
    if r >= 0.60: return 5.0
    if r >= 0.45: return 4.0
    if r >= 0.30: return 3.0
    if r >= 0.15: return 2.0
    return 1.0

# =========================
# Tabs UI
# =========================
TAB_MAIN, TAB_PACE = st.tabs(["Inputs (2 boxes) → Combined CSV", "Pace/Speed + Diagnostics (from Combined CSV)"])

# ---------- TAB 1 ----------
with TAB_MAIN:
    st.subheader("Box A — Run Style ONLY")
    st.caption("Paste ONLY the Run Style Figure table block. No Official Ratings. No Race Class block.")
    boxA = st.text_area("Box A paste (Run Style Figure)", height=260, key="boxA")

    st.subheader("Box B — Boxes 4 & 5 (single paste)")
    st.caption("Paste the block that includes Speed Series and condition lines with (W/P/T).")
    boxB = st.text_area("Box B paste", height=300, key="boxB")

    col_thr = st.columns(3)
    with col_thr[0]: front_thr = st.number_input("Front <", value=1.6, step=0.1)
    with col_thr[1]: prom_thr  = st.number_input("Prominent <", value=2.4, step=0.1)
    with col_thr[2]: mid_thr   = st.number_input("Mid <", value=3.0, step=0.1)

    if st.button("🚀 Process (A + B) → Combined CSV"):
        try:
            rs_df = parse_run_style(boxA, front_thr, prom_thr, mid_thr)

            b_raw = parse_boxB_single(boxB) if boxB.strip() else pd.DataFrame(columns=["Horse"])
            b_df = build_speed_conditions(b_raw) if not b_raw.empty else pd.DataFrame(columns=["Horse"])

            for d in (rs_df, b_df):
                if "Horse" in d.columns:
                    d["Horse"] = d["Horse"].astype(str).str.strip()

            merged = rs_df.merge(b_df, on="Horse", how="outer")

            st.success("Parsed ✓ (OR + Class removed)")

            a, b = st.tabs(["Run Style ✓", "Speed & Conditions (Box B) ✓"])
            with a:
                st.dataframe(rs_df, use_container_width=True)
            with b:
                st.dataframe(b_df, use_container_width=True)

            st.markdown("## 🧩 Combined Output (A + B)")
            st.dataframe(merged, use_container_width=True)

            st.download_button(
                "💾 Download Combined CSV",
                merged.to_csv(index=False),
                "cleanpace_v3_combined.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Failed: {e}")

# ---------- TAB 2 ----------
with TAB_PACE:
    st.subheader("Upload 🧩 Combined CSV")
    upl = st.file_uploader("Combined CSV", type=["csv"], key="pace_csv")

    c = st.columns(5)
    class_par  = c[0].number_input("Class Par (benchmark)", value=77.0, step=0.5)
    distance_f = c[1].number_input("Distance (f)", value=6.0, step=0.5, min_value=5.0)
    wp_even    = c[2].slider("wp (Even/Uncertain)", 0.3, 0.8, 0.50, 0.05)
    wp_conf    = c[3].slider("wp (Predictable Slow/Very Strong)", 0.3, 0.8, 0.65, 0.05)
    clip5      = c[4].checkbox("Clip Suitability to 1–5", value=True)

    pace_override = st.selectbox(
        "Pace override (optional)",
        options=["Auto", "Slow", "Even", "Strong", "Very Strong"],
        index=0,
        help="Leave Auto to use modelled pace, or force a scenario."
    )

    if upl is not None:
        try:
            df = pd.read_csv(upl)
            if "Horse" not in df.columns:
                st.error("Missing column: Horse"); st.stop()
            df["Horse"] = df["Horse"].astype(str).str.strip()

            # Map AdjSpeed & RS_Avg robustly
            adjspeed_aliases = ["AdjSpeed", "KeySpeedAvg", "Key Speed Factors Average", "Adjusted Speed", "Adjusted_Speed", "KeySpeed"]
            col_adjspeed = map_column(df, adjspeed_aliases)
            if col_adjspeed is None:
                st.error("Missing Adjusted Speed column (KeySpeedAvg)."); st.stop()
            df["AdjSpeed"] = pd.to_numeric(df[col_adjspeed], errors="coerce")

            rsavg_aliases = ["RS_Avg", "RS Avg", "RSAVG", "RS_Avg5", "RS_Avg (5)", "RS5_Avg", "RS(5)Avg"]
            col_rsavg = map_column(df, rsavg_aliases)
            if col_rsavg is None:
                st.error("Missing RS_Avg column."); st.stop()
            df["RS_Avg"] = pd.to_numeric(df[col_rsavg], errors="coerce")

            rs10_aliases = ["RS_Avg10", "RS Avg 10", "RSAVG10", "RS(10)Avg", "RS10_Avg"]
            col_rs10 = map_column(df, rs10_aliases)
            if col_rs10:
                df["RS_Avg10"] = pd.to_numeric(df[col_rs10], errors="coerce")
            else:
                df["RS_Avg10"] = np.nan

            # Backbone metrics
            df["ΔvsPar"] = df["AdjSpeed"] - float(class_par)
            df["Style"] = df["RS_Avg"].apply(lambda x: _rs_category_from_value(x, 1.6, 2.4, 3.0))

            s = Settings(class_par=class_par, distance_f=distance_f, wp_even=wp_even, wp_confident=wp_conf)
            df["LCP"] = df.apply(lambda r: _lcp_from_dvp(r["Style"], r["ΔvsPar"], s), axis=1)

            rows = [
                HorseRow(
                    horse=r.Horse,
                    style_cat=r.Style,
                    adj_speed=float(r.AdjSpeed) if pd.notna(r.AdjSpeed) else None,
                    dvp=float(r["ΔvsPar"]) if pd.notna(r["ΔvsPar"]) else None,
                    lcp=r.LCP,
                )
                for _, r in df.iterrows()
            ]
            scenario, conf_numeric, lcp_map, debug = project_pace_from_rows(rows, s)

            if pace_override != "Auto":
                debug["rules_applied"].append(f"Manual override applied: {pace_override}")
                debug["manual_override"] = pace_override
                scenario = pace_override
                conf_display = "Manual"
            else:
                conf_display = f"{conf_numeric:.2f}"

            # PaceFit map
            band = "5f" if distance_f <= 5.5 else ("6f" if distance_f <= 6.5 else "route")
            if scenario.startswith("Even (Front-Controlled)") and band == "route":
                pacefit_map = PACEFIT_EVEN_FC
            elif band == "5f":
                pacefit_map = PACEFIT_5F["Even"] if scenario.startswith("Even") else PACEFIT_5F.get(scenario, PACEFIT_5F["Even"])
            elif band == "6f":
                pacefit_map = PACEFIT_6F["Even"] if scenario.startswith("Even") else PACEFIT_6F.get(scenario, PACEFIT_6F["Even"])
            else:
                key = "Even" if scenario.startswith("Even") else scenario
                pacefit_map = PACEFIT.get(key, PACEFIT["Even"])

            wp = s.wp_confident if (scenario in ("Slow", "Very Strong") and conf_numeric >= 0.65) else s.wp_even
            if band == "5f": wp = max(wp, 0.60)
            elif band == "6f": wp = max(wp, 0.55)
            ws = 1 - wp

            df["PaceFit"] = df["Style"].map(pacefit_map).fillna(3.0)

            def _speedfit(v):
                if pd.isna(v): return 2.3
                v = float(v)
                if v >= 2: return 5.0
                if v >= -1: return 4.0
                if v >= -4: return 3.0
                if v >= -8: return 2.0
                return 1.0

            df["SpeedFit"] = df["ΔvsPar"].apply(_speedfit)
            df["Suitability_Base"] = df["PaceFit"] * wp + df["SpeedFit"] * ws
            df["wp"], df["ws"] = wp, ws
            df["Scenario"] = scenario
            df["Confidence"] = conf_display

            # =========================
            # Diagnostics (SEPARATE)
            # =========================

            # --- Conditions raw & scoring ---
            cond_cols_present = [c for c in COND_COLS if c in df.columns]
            # Conditions count & mean
            df["_cond_count"] = df[cond_cols_present].notna().sum(axis=1) if cond_cols_present else 0
            df["_cond_mean"] = df[cond_cols_present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True) if cond_cols_present else np.nan
            df["Conditions_Score"] = df["_cond_mean"].apply(lambda x: conditions_score_from_mean(float(x)) if pd.notna(x) else None)

            # --- Speed consistency (>= Par) ---
            # Use SpeedRunsRaw (preferred) or SpeedRunsList fallback
            sr_col = None
            for cand in ["SpeedRunsRaw", "speed_series", "SpeedRunsList"]:
                if cand in df.columns:
                    sr_col = cand
                    break
            if sr_col is None:
                df["Speed_TotalRuns"] = 0
                df["Speed_RunsAtPar"] = 0
                df["Speed_ParPct"] = np.nan
                df["SpeedConsistency_Score"] = None
            else:
                totals, atpar, pct = [], [], []
                for srs in df[sr_col].fillna("").astype(str):
                    nums = _numbers(srs)
                    n_total = len(nums)
                    n_par = sum(1 for x in nums if float(x) >= float(class_par))
                    totals.append(n_total)
                    atpar.append(n_par)
                    pct.append((n_par / n_total) if n_total > 0 else np.nan)

                df["Speed_TotalRuns"] = totals
                df["Speed_RunsAtPar"] = atpar
                df["Speed_ParPct"] = pct
                df["SpeedConsistency_Score"] = df["Speed_ParPct"].apply(lambda x: consistency_score_from_ratio(float(x)) if pd.notna(x) else None)

            # Optional clip for Suitability only (backbone)
            df["Suitability"] = df["Suitability_Base"]
            if clip5:
                df["Suitability"] = df["Suitability"].clip(1.0, 5.0)

            # =========================
            # OUTPUTS
            # =========================
            st.subheader(
                f"Projected Pace: {scenario} (confidence {conf_display}) — Band: {band}"
            )
            with st.expander("Why this pace? (Reason used)", expanded=True):
                counts = debug.get("counts", {})
                st.markdown(
                    f"- **Front (High):** {counts.get('Front_High', 0)} &nbsp;&nbsp; "
                    f"**Prominent (High):** {counts.get('Prominent_High', 0)}  \n"
                    f"- **Early energy:** {debug.get('early_energy', 0.0):.2f}"
                )
                rules = debug.get("rules_applied", [])
                if rules:
                    st.markdown("**Rules applied:**")
                    for r in rules:
                        st.write(f"• {r}")

            # ---- Backbone table (unchanged ranking) ----
            st.markdown("## Backbone — Pace/Speed Suitability (unchanged)")
            backbone_cols = [
                "Horse", "RS_Avg", "RS_Avg10", "Style",
                "AdjSpeed", "ΔvsPar", "LCP",
                "PaceFit", "SpeedFit", "wp", "ws",
                "Suitability_Base", "Suitability",
                "Scenario", "Confidence",
            ]
            backbone_cols = [c for c in backbone_cols if c in df.columns]
            out_backbone = df[backbone_cols].sort_values(["Suitability", "Suitability_Base", "SpeedFit"], ascending=False)
            st.dataframe(out_backbone, use_container_width=True)

            # ---- SUPPORTING TABLES: "parrot Box B" ----
            st.markdown("## Diagnostics — Supporting Tables (Box B → Checkable)")

            # Table 1: Raw Conditions Input
            st.markdown("### 1) Conditions — Raw Place Data (parroted from Box B)")
            raw_cond_cols = ["Horse"] + [c for c in ["crs_place","dist_place","going_place","cls_place","trackstyle_place","lhrh_place","runpm1_place"] if c in df.columns]
            st.dataframe(df[raw_cond_cols], use_container_width=True)

            # Table 2: Conditions Score Breakdown
            st.markdown("### 2) Conditions — Score Breakdown")
            cond_break_cols = ["Horse", "_cond_count", "_cond_mean", "Conditions_Score"]
            cond_break_cols = [c for c in cond_break_cols if c in df.columns]
            cond_break = df[cond_break_cols].rename(columns={
                "_cond_count": "Conditions_Count",
                "_cond_mean": "Conditions_Mean",
            })
            st.dataframe(cond_break, use_container_width=True)

            # Table 3: Raw Speed History (parrot speed series)
            st.markdown("### 3) Speed History — Raw (parroted from Box B)")
            speed_raw_cols = ["Horse"] + ([sr_col] if sr_col else [])
            if speed_raw_cols and (len(speed_raw_cols) > 1):
                st.dataframe(df[speed_raw_cols], use_container_width=True)
            else:
                st.info("No speed series column found in the combined CSV (expected SpeedRunsRaw / speed_series / SpeedRunsList).")

            # Table 4: Speed Consistency Breakdown
            st.markdown("### 4) Speed Consistency — Breakdown vs Par")
            cons_cols = ["Horse", "Speed_TotalRuns", "Speed_RunsAtPar", "Speed_ParPct", "SpeedConsistency_Score"]
            cons_cols = [c for c in cons_cols if c in df.columns]
            cons = df[cons_cols].copy()
            if "Speed_ParPct" in cons.columns:
                cons["Speed_ParPct"] = cons["Speed_ParPct"].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "")
            st.dataframe(cons, use_container_width=True)

            # Table 5: Diagnostics Summary (read-only)
            st.markdown("### 5) Diagnostics Summary (read-only overlay)")
            diag_cols = ["Horse", "Suitability", "Conditions_Score", "SpeedConsistency_Score"]
            diag_cols = [c for c in diag_cols if c in df.columns]
            st.dataframe(df[diag_cols].sort_values(["Suitability"], ascending=False), use_container_width=True)

            # Downloads
            st.download_button(
                "💾 Download Backbone (Suitability) CSV",
                out_backbone.to_csv(index=False),
                "cleanpace_v3_backbone.csv",
                mime="text/csv",
            )
            st.download_button(
                "💾 Download Full (Backbone + Diagnostics) CSV",
                df.to_csv(index=False),
                "cleanpace_v3_full.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
