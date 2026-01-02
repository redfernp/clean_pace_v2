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
st.set_page_config(page_title="CleanPace v3 — Backbone + Diagnostics", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v3 — Pace/Speed Backbone + Diagnostics (Conditions + Consistency)")

# =========================
# Small helpers
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
# Parse Run Style section (Box A — RS ONLY)
# =========================
def parse_run_style(rs_text: str, front_thr=1.6, prom_thr=2.4, mid_thr=3.0) -> pd.DataFrame:
    lines = rs_text.splitlines()
    if lines and lines[0].strip().lower().startswith("run style figure"):
        rs_text = "\n".join(lines[1:])
    df = _read_table_guess(rs_text)
    if "Horse" not in df.columns:
        raise ValueError("Run Style: missing 'Horse' column.")
    df["Horse"] = df["Horse"].astype(str).str.strip()

    lto5 = [c for c in [f"Lto{i}" for i in range(1,6)] if c in df.columns]
    lto10 = [c for c in [f"Lto{i}" for i in range(1,11)] if c in df.columns]
    for c in lto10: df[c] = pd.to_numeric(df[c], errors="coerce")

    if "RS_Avg" not in df.columns:
        df["RS_Avg"]   = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto5]), axis=1)
    if "RS_Avg10" not in df.columns:
        df["RS_Avg10"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto10]), axis=1)

    df["RS_Cat"] = df["RS_Avg"].apply(lambda x: _rs_category_from_value(x, front_thr, prom_thr, mid_thr))

    if "Dr.%" in df.columns:
        df["Dr.%"] = df["Dr.%"].astype(str).str.replace("%", "", regex=False)
        df["Dr.%"] = pd.to_numeric(df["Dr.%"], errors="coerce")

    keep = ["Horse"] + lto10 + ["RS_Avg","RS_Avg10","RS_Cat"] + \
           [c for c in ["Mode 5","Mode 10","Total","Mode All","Draw","Dr.%"] if c in df.columns]
    return df[keep]

# =========================
# Boxes 4 & 5 (single paste) — Robust name detection
# =========================
BOX_ORDER = ["win_pct", "form_avg", "speed_series", "crs", "dist", "lhrh", "going", "cls", "runpm1", "trackstyle"]

def _is_win_line(s: str) -> bool:
    """True if line looks like a Win % row such as '22 (2/2/9)'."""
    return bool(re.search(r"\(\s*\d+\s*/\s*\d+\s*/\s*\d+\s*\)", str(s)))

def _looks_like_name(line_or_list, idx: Optional[int] = None) -> bool:
    """
    Treat a line as a horse name only if:
      • it has no digits, commas, parentheses, slashes or £
      • the next non-empty line looks like a Win % line '(W/P/T)'
    """
    if isinstance(line_or_list, list):
        lines = line_or_list
        if idx is None or idx < 0 or idx >= len(lines):
            return False
        t = lines[idx].strip()
        if not t or t.lower().startswith("horse"):
            return False
        if re.search(r"[0-9(),/£]", t):
            return False
        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        return _is_win_line(lines[j]) if j < len(lines) else False
    else:
        t = str(line_or_list).strip()
        if not t or t.lower().startswith("horse"):
            return False
        if re.search(r"[0-9(),/£]", t):
            return False
        return True

def _parse_wpt_value(val: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    val = str(val).strip()
    m = re.search(r"\((\d+)\s*/\s*(\d+)\s*/\s*(\d+)\)", val)
    if not m: return None, None, None, None
    w = int(m.group(1)); p = int(m.group(2)); t = int(m.group(3))
    place = None if t == 0 else round((w+p)/t, 3)
    return place, w, p, t

def _parse_speed_series(s: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    nums = _numbers(s)
    if not nums: return None, None, None, None
    last = nums[-1]; highest = max(nums)
    avg3 = round(sum(nums[-3:])/min(3, len(nums)), 1)
    avgall = round(sum(nums)/len(nums), 1)
    return last, highest, avg3, avgall

def parse_box45_single(raw: str) -> pd.DataFrame:
    lines = [ln for ln in raw.replace("\r\n","\n").replace("\r","\n").split("\n") if ln.strip()]
    if not lines: return pd.DataFrame(columns=["Horse"])
    if lines[0].strip().lower().startswith("horse"):
        lines = lines[1:]

    i, n = 0, len(lines)
    recs: List[Dict[str, object]] = []
    while i < n:
        while i < n and not _looks_like_name(lines, i):
            i += 1
        if i >= n: break
        name = lines[i].strip(); i += 1

        values: List[str] = []
        while i < n and len(values) < 10:
            if _looks_like_name(lines, i):
                break
            values.append(lines[i].strip()); i += 1
        values += [None] * (10 - len(values)) if len(values) < 10 else []
        data = {"Horse": name}
        for key, val in zip(BOX_ORDER, values):
            data[key] = val
        recs.append(data)

    df = pd.DataFrame(recs)
    if df.empty: return df
    df["Horse"] = df["Horse"].astype(str).str.strip()
    return df

def build_speed_conditions(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty: return df_raw.copy()
    out = df_raw.copy()

    # keep raw series for transparency
    out["SpeedRunsRaw"] = out["speed_series"].fillna("").astype(str)

    last_list, high_list, avg3_list, all_list, keyavg_list, runs_list_str = [], [], [], [], [], []
    for s in out["SpeedRunsRaw"].fillna(""):
        nums = _numbers(s)
        runs_list_str.append(", ".join([str(int(x)) if float(x).is_integer() else str(x) for x in nums]) if nums else "")

        last, high, avg3, avgall = _parse_speed_series(s)
        last_list.append(last); high_list.append(high)
        avg3_list.append(avg3);  all_list.append(avgall)
        keyavg_list.append(None if (last is None or high is None or avg3 is None) else round((last+high+avg3)/3, 1))

    out["SpeedRunsList"] = runs_list_str
    out["LastRace"] = last_list
    out["Highest"]  = high_list
    out["Avg3"]     = avg3_list
    out["AvgAll"]   = all_list
    out["KeySpeedAvg"] = keyavg_list

    for col in ["crs","dist","lhrh","going","cls","runpm1","trackstyle"]:
        if col in out.columns:
            out[f"{col}_place"] = [ _parse_wpt_value(v)[0] for v in out[col].fillna("") ]

    keep = ["Horse",
            "SpeedRunsRaw","SpeedRunsList",
            "LastRace","Highest","Avg3","AvgAll","KeySpeedAvg",
            "crs_place","dist_place","lhrh_place","going_place","cls_place","runpm1_place","trackstyle_place"]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

# =========================
# Pace engine + Suitability (backbone unchanged)
# =========================
@dataclass
class HorseRow:
    horse: str
    style_cat: Optional[str]
    adj_speed: Optional[float]
    dvp: Optional[float]
    lcp: str

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
    if style_cat not in ("Front","Prominent") or dvp is None:
        return "N/A"
    if dvp >= s.lcp_high: return "High"
    if dvp >= s.lcp_question_low: return "Questionable"
    return "Unlikely"

# ========= HYBRID PACE HELPER =========
def should_use_hybrid(distance_f: float,
                      scenario: str,
                      confidence: float,
                      counts: dict) -> bool:
    """
    Determine whether to use a Hybrid Even/Strong suitability blend.
    Hybrid pace only useful at <= 8f and when scenario/confidence is borderline.
    """
    FH = counts.get("Front_High", 0)
    PH = counts.get("Prominent_High", 0)
    total_high = FH + PH

    if distance_f > 8.0:
        return False
    if scenario.startswith("Slow") or scenario.startswith("Very Strong"):
        return False
    if not (scenario.startswith("Even") or scenario == "Strong"):
        return False
    if confidence < 0.45 or confidence > 0.65:
        return False
    if total_high <= 1:
        return False
    if FH >= 3:
        return False
    if FH in (1, 2) and total_high >= 2:
        return True
    return False

# ========= LATE-STRONG (6f) WARNING TAG =========
def is_late_strong_6f(distance_f: float,
                      scenario: str,
                      confidence: float,
                      counts: dict,
                      energy: float) -> bool:
    """
    Display-only warning for 6f: race can *finish* Strong even if labelled Even/Hybrid.
    DOES NOT change scenario/hybrid/suitability; only drives one extra line of text.
    """
    # 6f band only
    if not (5.5 < float(distance_f) <= 6.5):
        return False

    FH = int(counts.get("Front_High", 0))
    PH = int(counts.get("Prominent_High", 0))
    total_high = FH + PH
    e = float(energy)
    c = float(confidence) if confidence is not None else 0.0

    # We only care about "Even–Strong type" races (Even / Strong / Hybrid)
    # NOTE: your UI scenario becomes "Hybrid (Even–Strong)" later, but at this point
    # scenario is still the raw scenario from the pace engine.
    if scenario.startswith("Slow") or scenario.startswith("Very Strong"):
        return False
    if not (scenario.startswith("Even") or scenario == "Strong"):
        return False

    # Needs credible early pressure (otherwise it's just Slow/Even)
    if total_high < 2:
        return False

    # If there are 3+ High fronts it's a true burn-up (Strong throughout), not "late-strong"
    if FH >= 3:
        return False

    # Key pattern you’re trying to capture:
    # - 6f
    # - multiple credible early (>=2)
    # - energy can be high (like 4.8)
    # - but not an extreme front-war (cap FH at 2)
    # - confidence moderate-to-high
    if FH in (1, 2) and total_high >= 2 and 0.55 <= c <= 0.80 and e >= 3.2:
        return True

    return False


# =========================
# Pace projection engine (advanced)
# =========================
def project_pace_from_rows(rows: List[HorseRow], s: Settings) -> Tuple[str,float,Dict[str,str],Dict[str,object]]:
    debug = {"rules_applied": []}
    if not rows: return "N/A", 0.0, {}, {"error": "no rows"}

    d = pd.DataFrame([{"horse":r.horse, "style":r.style_cat, "dvp":r.dvp, "lcp":r.lcp} for r in rows])

    front_all = d[d["style"]=="Front"]; prom_all = d[d["style"]=="Prominent"]
    front_high = d[(d["style"]=="Front") & (d["lcp"]=="High")]
    prom_high  = d[(d["style"]=="Prominent") & (d["lcp"]=="High")]
    front_q    = d[(d["style"]=="Front") & (d["lcp"]=="Questionable")]
    prom_q     = d[(d["style"]=="Prominent") & (d["lcp"]=="Questionable")]

    n_front, n_fh, n_ph, n_fq, n_pq = len(front_all), len(front_high), len(prom_high), len(front_q), len(prom_q)
    n_high_early = n_fh + n_ph

    W_FH, W_PH, W_FQ, W_PQ = 2.0, 0.8, 0.5, 0.2
    energy = (W_FH*n_fh) + (W_PH*n_ph) + (W_FQ*n_fq) + (W_PQ*n_pq)

    band = _dist_band(s.distance_f)
    locked_strong = False

    # --- 3-LCP guarantee rule (FRONT ONLY) ---
    if n_fh >= 3:
        scenario, conf = "Strong", 0.80
        locked_strong = True
        debug["rules_applied"].append("≥3 High-LCP Fronts → Strong (locked)")
    else:
        if n_fh >= 2:
            scenario, conf = "Strong", 0.65
            debug["rules_applied"].append("≥2 High Front → Strong")
        elif (n_fh+n_ph)>=3 and energy>=3.2:
            scenario, conf = "Strong", 0.65
            debug["rules_applied"].append("≥3 High early & energy≥3.2 → Strong")
        elif energy >= 3.2:
            scenario, conf = "Strong", 0.60
            debug["rules_applied"].append("energy≥3.2 → Strong")
        elif (n_fh+n_ph) >= 2:
            scenario, conf = "Even", 0.60
            debug["rules_applied"].append("≥2 High early → Even")
        elif (n_fh+n_ph) == 1 and (n_fq+n_pq) >= 1:
            scenario, conf = "Even", 0.55
            debug["rules_applied"].append("1 High + Questionables → Even")
        elif (n_fh+n_ph) == 1:
            scenario, conf = "Even", 0.60
            debug["rules_applied"].append("1 High → Even")
        elif (n_fq+n_pq) >= 1:
            scenario, conf = "Slow", 0.60
            debug["rules_applied"].append("Only Questionables → Slow")
        else:
            scenario, conf = "Slow", 0.70
            debug["rules_applied"].append("No credible early → Slow")

        if n_front == 0 or n_fh == 0:
            allow_strong = False
            if n_ph >= 3:
                try: allow_strong = float(prom_high["dvp"].mean()) >= -1.0
                except Exception: allow_strong = False
            if not allow_strong and scenario in ("Strong","Very Strong"):
                scenario, conf = "Even", min(conf,0.60)
                debug["rules_applied"].append("No-front cap → Even")

        if n_fh == 1 and n_ph <= 1:
            try: lf = float(front_high["dvp"].iloc[0])
            except Exception: lf = None
            if (lf is not None) and (lf >= 2.0) and scenario in ("Strong","Very Strong"):
                scenario, conf = "Even", max(conf,0.65)
                debug["rules_applied"].append("Dominant-front cap → Even")

        if n_fh == 1:
            try: lf2 = float(front_high["dvp"].iloc[0])
            except Exception: lf2 = None
            if (lf2 is None) or (lf2 <= 1.0):
                if scenario in ("Strong","Very Strong"):
                    scenario, conf = "Even", max(conf,0.60)
                    debug["rules_applied"].append("Single-front cap (≤+1) → Even")
            if (lf2 is not None) and (lf2 <= -2.0) and (n_ph <= 1) and scenario == "Even":
                scenario, conf = "Slow", max(conf,0.65)
                debug["rules_applied"].append("Single-front below par → Slow")

        if n_front == 1:
            try: anyf = float(front_all["dvp"].iloc[0])
            except Exception: anyf = None
            if (anyf is not None) and (anyf <= -8.0):
                idx = max(0, ["Slow","Even","Strong","Very Strong"].index(scenario)-1)
                scenario = ["Slow","Even","Strong","Very Strong"][idx]
                conf = max(conf,0.65)
                debug["rules_applied"].append("Weak solo leader → downgrade")

        if band in ("5f","6f") and n_fh == 1 and n_ph <= 2:
            try: lf = float(front_high["dvp"].iloc[0])
            except Exception: lf = None
            dvp_ok = -0.5 if band=="5f" else -1.0
            energy_cap = 4.0 if band=="5f" else 3.6
            if (lf is not None) and (lf >= dvp_ok) and (energy < energy_cap) and scenario in ("Strong","Very Strong"):
                scenario, conf = "Even", max(conf,0.65)
                debug["rules_applied"].append("Sprint cap: Strong→Even")

        if band == "route" and scenario == "Even" and n_fh == 1 and n_ph <= 1:
            try: lf = float(front_high["dvp"].iloc[0])
            except Exception: lf = None
            if (lf is not None) and (lf >= -1.0):
                scenario = "Even (Front-Controlled)"
                debug["rules_applied"].append("Front-controlled tag")

    debug.update({
        "counts":{
            "Front_all":int(n_front),
            "Front_High":int(n_fh),
            "Prominent_High":int(n_ph)
        },
        "early_energy": float(energy),
        "distance_band": band
    })
    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario, conf, lcp_map, debug

# =========================
# Diagnostics (NEW, separate; no effect on Suitability)
# =========================
COND_COLS = ["dist_place","cls_place","runpm1_place","trackstyle_place","crs_place","lhrh_place","going_place"]

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
TAB_MAIN, TAB_PACE = st.tabs(["All Inputs (2 boxes)", "Pace & Backbone + Diagnostics (from Combined CSV)"])

# ---------- TAB 1: Inputs ----------
with TAB_MAIN:
    st.subheader("Box A — Run Style ONLY")
    st.caption("Paste ONLY the Run Style Figure table. (No Official Ratings, no Race Class section.)")
    boxA = st.text_area("Box A paste", height=280, key="boxA")

    st.subheader("Box B — Boxes 4 & 5 (single paste)")
    st.caption("Paste the single block starting 'Horse Win % ...' that includes Speed Series and (W/P/T) lines.")
    boxB = st.text_area("Box B paste", height=320, key="boxB")

    col_thr = st.columns(3)
    with col_thr[0]: front_thr = st.number_input("Front <", value=1.6, step=0.1)
    with col_thr[1]: prom_thr  = st.number_input("Prominent <", value=2.4, step=0.1)
    with col_thr[2]: mid_thr   = st.number_input("Mid <", value=3.0, step=0.1)

    if st.button("🚀 Process All (A + B)"):
        try:
            rs_df = parse_run_style(boxA, front_thr, prom_thr, mid_thr)

            b_raw = parse_box45_single(boxB) if boxB.strip() else pd.DataFrame(columns=["Horse"])
            b_df = build_speed_conditions(b_raw) if not b_raw.empty else pd.DataFrame(columns=["Horse"])

            for d in (rs_df, b_df):
                if "Horse" in d.columns:
                    d["Horse"] = d["Horse"].astype(str).str.strip()

            merged = rs_df.merge(b_df, on="Horse", how="outer")

            st.success("Parsed ✓ (OR + Race Class removed)")

            a, b = st.tabs(["Run Style ✓", "Speed & Conditions (Box B) ✓"])
            with a: st.dataframe(rs_df, use_container_width=True)
            with b: st.dataframe(b_df, use_container_width=True)

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

# ---------- TAB 2: Backbone + Diagnostics ----------
with TAB_PACE:
    st.subheader("Upload 🧩 Combined Output (A + B) CSV")
    upl = st.file_uploader("Combined CSV", type=["csv"], key="pace_csv")

    c = st.columns(5)
    class_par = c[0].number_input("Class Par", value=77.0, step=0.5)
    distance_f = c[1].number_input("Distance (f)", value=6.0, step=0.5, min_value=5.0)
    wp_even    = c[2].slider("wp (Even/Uncertain)", 0.3, 0.8, 0.50, 0.05)
    wp_conf    = c[3].slider("wp (Predictable Slow/Very Strong)", 0.3, 0.8, 0.65, 0.05)
    clip5      = c[4].checkbox("Clip Suitability to 1–5", value=True)

    pace_override = st.selectbox(
        "Pace override (optional)",
        options=["Auto", "Slow", "Even", "Strong", "Very Strong"],
        index=0,
        help="Leave as 'Auto' to use the modelled pace, or select a scenario to override it."
    )

    if upl is not None:
        try:
            df = pd.read_csv(upl)
            if "Horse" not in df.columns:
                st.error("Missing column in CSV: Horse"); st.stop()
            df["Horse"] = df["Horse"].astype(str).str.strip()

            # ------- Robust AdjSpeed & RS_Avg mapping -------
            adjspeed_aliases = [
                "AdjSpeed","KeySpeedAvg","Key Speed Factors Average",
                "Key_Speed_Factors_Average","Adjusted Speed","Adjusted_Speed","KeySpeed"
            ]
            col_adjspeed = map_column(df, adjspeed_aliases)
            if col_adjspeed is None:
                st.error("Missing Adjusted Speed column. Expected one of: " + ", ".join(adjspeed_aliases))
                st.stop()
            if col_adjspeed != "AdjSpeed":
                df["AdjSpeed"] = pd.to_numeric(df[col_adjspeed], errors="coerce")
            else:
                df["AdjSpeed"] = pd.to_numeric(df["AdjSpeed"], errors="coerce")

            rsavg_aliases = ["RS_Avg","RS Avg","RSAVG","RS_Avg5","RS_Avg (5)","RS5_Avg","RS(5)Avg"]
            col_rsavg = map_column(df, rsavg_aliases)
            if col_rsavg is None:
                st.error("Missing RS_Avg (avg of last 5 run styles). Expected one of: " + ", ".join(rsavg_aliases))
                st.stop()
            if col_rsavg != "RS_Avg":
                df["RS_Avg"] = pd.to_numeric(df[col_rsavg], errors="coerce")
            else:
                df["RS_Avg"] = pd.to_numeric(df["RS_Avg"], errors="coerce")

            rs10_aliases = ["RS_Avg10","RS Avg 10","RSAVG10","RS(10)Avg","RS10_Avg"]
            col_rs10 = map_column(df, rs10_aliases)
            if col_rs10 and col_rs10 != "RS_Avg10":
                df["RS_Avg10"] = pd.to_numeric(df[col_rs10], errors="coerce")
            elif "RS_Avg10" in df.columns:
                df["RS_Avg10"] = pd.to_numeric(df["RS_Avg10"], errors="coerce")
            else:
                df["RS_Avg10"] = np.nan

            # Δ vs Par + Style
            df["ΔvsPar"] = pd.to_numeric(df["AdjSpeed"], errors="coerce") - float(class_par)
            df["Style"] = df["RS_Avg"].apply(lambda x: _rs_category_from_value(x, 1.6, 2.4, 3.0))

            # LCP from Δ vs Par only for Front/Prominent
            s = Settings(class_par=class_par, distance_f=distance_f, wp_even=wp_even, wp_confident=wp_conf)
            df["LCP"] = df.apply(lambda r: _lcp_from_dvp(r["Style"], r["ΔvsPar"], s), axis=1)

            # Build rows for pace projection
            rows = [HorseRow(
                horse=r.Horse,
                style_cat=r.Style,
                adj_speed=float(r.AdjSpeed) if pd.notna(r.AdjSpeed) else None,
                dvp=float(r["ΔvsPar"]) if pd.notna(r["ΔvsPar"]) else None,
                lcp=r.LCP
            ) for _, r in df.iterrows()]
            scenario, conf_numeric, lcp_map, debug = project_pace_from_rows(rows, s)

            # Apply manual override if selected
            if pace_override != "Auto":
                debug["rules_applied"].append(f"Manual override applied: {pace_override}")
                debug["manual_override"] = pace_override
                scenario = pace_override
                conf_display = "Manual"
            else:
                conf_display = f"{conf_numeric:.2f}"

            # ---- Late-Strong (6f) warning flag (no UI change yet; just compute) ----
            counts_for_warn = debug.get("counts", {})
            energy_for_warn = debug.get("early_energy", 0.0)
            late_strong_warn = is_late_strong_6f(
                distance_f=distance_f,
                scenario=scenario,
                confidence=conf_numeric,
                counts=counts_for_warn,
                energy=energy_for_warn
            )
            debug["late_strong_warn"] = late_strong_warn

            # PaceFit maps by band
            band = "5f" if distance_f <= 5.5 else ("6f" if distance_f <= 6.5 else "route")

            # Even & Strong maps (for Hybrid and scenario sensitivity)
            if band == "5f":
                even_map = PACEFIT_5F["Even"]
                strong_map = PACEFIT_5F["Strong"]
                slow_map = PACEFIT_5F["Slow"]
            elif band == "6f":
                even_map = PACEFIT_6F["Even"]
                strong_map = PACEFIT_6F["Strong"]
                slow_map = PACEFIT_6F["Slow"]
            else:
                even_map = PACEFIT_EVEN_FC if scenario.startswith("Even (Front-Controlled)") else PACEFIT["Even"]
                strong_map = PACEFIT["Strong"]
                slow_map = PACEFIT["Slow"]

            # Final scenario-specific map (for base Suitability)
            if scenario.startswith("Even (Front-Controlled)") and band == "route":
                pacefit_map = PACEFIT_EVEN_FC
            elif band == "5f":
                pacefit_map = PACEFIT_5F["Even"] if scenario.startswith("Even") else PACEFIT_5F.get(scenario, PACEFIT_5F["Even"])
            elif band == "6f":
                pacefit_map = PACEFIT_6F["Even"] if scenario.startswith("Even") else PACEFIT_6F.get(scenario, PACEFIT_6F["Even"])
            else:
                key = "Even" if scenario.startswith("Even") else scenario
                pacefit_map = PACEFIT.get(key, PACEFIT["Even"])

            # Pace/Speed weights (same backbone rules)
            wp = s.wp_confident if (scenario in ("Slow","Very Strong") and conf_numeric >= 0.65) else s.wp_even
            if band == "5f": wp = max(wp, 0.60)
            elif band == "6f": wp = max(wp, 0.55)
            ws = 1 - wp

            # SpeedFit
            def _speedfit(v):
                if pd.isna(v): return 2.3
                v = float(v)
                if v >= 2: return 5.0
                if v >= -1: return 4.0
                if v >= -4: return 3.0
                if v >= -8: return 2.0
                return 1.0
            df["SpeedFit"] = df["ΔvsPar"].apply(_speedfit)

            # Base PaceFit (scenario) + base Suitability
            df["PaceFit"] = df["Style"].map(pacefit_map).fillna(3.0)

            # Scenario sensitivity (requested): Suitability under Slow/Even/Strong
            df["PaceFit_Slow"]   = df["Style"].map(slow_map).fillna(3.0)
            df["PaceFit_Even"]   = df["Style"].map(even_map).fillna(3.0)
            df["PaceFit_Strong"] = df["Style"].map(strong_map).fillna(3.0)

            df["Suitability_Slow"]   = df["PaceFit_Slow"]   * wp + df["SpeedFit"] * ws
            df["Suitability_Even"]   = df["PaceFit_Even"]   * wp + df["SpeedFit"] * ws
            df["Suitability_Strong"] = df["PaceFit_Strong"] * wp + df["SpeedFit"] * ws

            # Base Even/Strong suitability bases (for Hybrid engine)
            df["Suitability_Base_Even"]   = df["Suitability_Even"]
            df["Suitability_Base_Strong"] = df["Suitability_Strong"]

            # Start with scenario-based base (non-hybrid)
            df["Suitability_Base"] = df["PaceFit"] * wp + df["SpeedFit"] * ws
            df["Suitability_Base_Hybrid"] = df["Suitability_Base_Even"]  # default

            # HYBRID PACE — only if Auto
            use_hyb = False
            if pace_override == "Auto":
                use_hyb = should_use_hybrid(
                    distance_f=distance_f,
                    scenario=scenario,
                    confidence=conf_numeric,
                    counts=debug.get("counts", {})
                )

            if use_hyb:
                debug["rules_applied"].append("Hybrid pace activated (Even + Strong, 60/40 weighted)")
                if scenario.startswith("Even"):
                    hybrid = 0.6 * df["Suitability_Base_Even"] + 0.4 * df["Suitability_Base_Strong"]
                elif scenario == "Strong":
                    hybrid = 0.6 * df["Suitability_Base_Strong"] + 0.4 * df["Suitability_Base_Even"]
                else:
                    hybrid = 0.5 * df["Suitability_Base_Even"] + 0.5 * df["Suitability_Base_Strong"]

                df["Suitability_Base_Hybrid"] = hybrid
                df["Suitability_Base"] = df["Suitability_Base_Hybrid"]
                df["Scenario"] = "Hybrid (Even–Strong)"
            else:
                df["Scenario"] = scenario

            df["wp"], df["ws"] = wp, ws
            df["Confidence"] = conf_display

            # FINAL Suitability (backbone only; no OR/Class/Cond additions)
            df["Suitability"] = df["Suitability_Base"]
            if clip5:
                df["Suitability"] = df["Suitability"].clip(1.0, 5.0)

            # =========================
            # Diagnostics (separate)
            # =========================
            # Conditions score
            cond_cols_present = [c for c in COND_COLS if c in df.columns]
            if cond_cols_present:
                df["Conditions_Count"] = df[cond_cols_present].notna().sum(axis=1)
                df["Conditions_Mean"] = df[cond_cols_present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
                df["Conditions_Score"] = df["Conditions_Mean"].apply(lambda x: conditions_score_from_mean(float(x)) if pd.notna(x) else None)
            else:
                df["Conditions_Count"] = 0
                df["Conditions_Mean"] = np.nan
                df["Conditions_Score"] = None

            # Speed consistency (runs >= Par)
            sr_col = None
            for cand in ["SpeedRunsRaw","speed_series","SpeedRunsList"]:
                if cand in df.columns:
                    sr_col = cand
                    break

            totals, atpar, pct = [], [], []
            if sr_col is not None:
                for srs in df[sr_col].fillna("").astype(str):
                    nums = _numbers(srs)
                    n_total = len(nums)
                    n_par = sum(1 for x in nums if float(x) >= float(class_par))
                    totals.append(n_total)
                    atpar.append(n_par)
                    pct.append((n_par / n_total) if n_total > 0 else np.nan)
            else:
                totals = [0]*len(df)
                atpar = [0]*len(df)
                pct = [np.nan]*len(df)

            df["Speed_TotalRuns"] = totals
            df["Speed_RunsAtPar"] = atpar
            df["Speed_ParPct"] = pct
            df["SpeedConsistency_Score"] = df["Speed_ParPct"].apply(lambda x: consistency_score_from_ratio(float(x)) if pd.notna(x) else None)

            # =========================
            # OUTPUTS
            # =========================
            st.subheader(
                f"Projected Pace: {df['Scenario'].iloc[0]} (confidence {conf_display}) — "
                f"Band: {band}"
            )

            with st.expander("Why this pace? (Reason used)", expanded=True):
                counts = debug.get("counts", {})
                st.markdown(
                    f"- **Front (High):** {counts.get('Front_High',0)} &nbsp;&nbsp; "
                    f"**Prominent (High):** {counts.get('Prominent_High',0)}  \n"
                    f"- **Early energy:** {debug.get('early_energy',0.0):.2f}"
                )
                rules = debug.get("rules_applied", [])
                if rules:
                    st.markdown("**Rules applied:**")
                    for r in rules: st.write(f"• {r}")

                # ---- UI CHANGE REQUESTED: add ONE extra line at the bottom (only when flagged) ----
                if debug.get("late_strong_warn", False):
                    st.markdown(
                        "\n⚠️ **Beware Late-Strong (6f):** pace may lift sharply late (~2f → line). "
                        "Prominent / first-wave attackers can be advantaged."
                    )

            st.markdown("## Backbone — Pace/Speed Suitability (unchanged)")
            show_cols = [
                "Horse","RS_Avg","RS_Avg10","Style","AdjSpeed","ΔvsPar","LCP",
                "PaceFit","SpeedFit","wp","ws",
                "Suitability_Slow","Suitability_Even","Suitability_Strong",
                "Suitability_Base","Suitability",
                "Scenario","Confidence"
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            out_backbone = df[show_cols].sort_values(
                ["Suitability","Suitability_Base","Suitability_Even","SpeedFit"],
                ascending=False
            )
            st.dataframe(out_backbone, use_container_width=True)

            st.markdown("## Diagnostics — Supporting Tables (Box B parrot + checks)")

            st.markdown("### 1) Conditions — Raw Place Data (parroted)")
            raw_cond_cols = ["Horse"] + [c for c in ["crs_place","dist_place","lhrh_place","going_place","cls_place","runpm1_place","trackstyle_place"] if c in df.columns]
            st.dataframe(df[raw_cond_cols], use_container_width=True)

            st.markdown("### 2) Conditions — Score Breakdown")
            cond_break = df[["Horse","Conditions_Count","Conditions_Mean","Conditions_Score"]].copy()
            st.dataframe(cond_break, use_container_width=True)

            st.markdown("### 3) Speed History — Raw (parroted)")
            if sr_col is not None:
                st.dataframe(df[["Horse", sr_col]], use_container_width=True)
            else:
                st.info("No speed series column found in the combined CSV (expected SpeedRunsRaw / speed_series / SpeedRunsList).")

            st.markdown("### 4) Speed Consistency — Breakdown vs Par")
            cons = df[["Horse","Speed_TotalRuns","Speed_RunsAtPar","Speed_ParPct","SpeedConsistency_Score"]].copy()
            cons["Speed_ParPct"] = cons["Speed_ParPct"].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "")
            st.dataframe(cons, use_container_width=True)

            st.markdown("### 5) Overlay — Backbone + Diagnostics (read-only)")
            overlay_cols = ["Horse","Suitability","Suitability_Even","Suitability_Strong","Suitability_Slow","Conditions_Score","SpeedConsistency_Score"]
            overlay_cols = [c for c in overlay_cols if c in df.columns]
            st.dataframe(df[overlay_cols].sort_values(["Suitability"], ascending=False), use_container_width=True)

            st.download_button(
                "💾 Download Backbone + Diagnostics CSV",
                df.to_csv(index=False),
                "cleanpace_v3_backbone_plus_diagnostics.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

