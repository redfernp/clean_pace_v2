# app.py
from __future__ import annotations
import re
from io import StringIO
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# UI CONFIG
# =============================================================================
st.set_page_config(page_title="Clean Pace v2 – Normalize & Suitability", page_icon="🏇", layout="wide")
st.title("🏇 Clean Pace v2")

# =============================================================================
# HELPERS — GENERAL
# =============================================================================

def _safe_read_table(raw: str) -> pd.DataFrame:
    """Try TSV then CSV; trim blank cols/rows."""
    txt = raw.strip()
    if not txt:
        return pd.DataFrame()
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(StringIO(txt), sep=sep)
            # drop fully-empty columns/rows
            df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
            return df
        except Exception:
            pass
    return pd.DataFrame()

def _num(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        s = str(x).strip().replace(",", "")
        # extract first signed float-like
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group()) if m else None
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    n = _num(x)
    return int(round(n)) if n is not None else None

def _pct_from_tuple(s: str) -> Optional[float]:
    """
    s like '22 (2/2/9)' -> place% = (W+P)/T.
    Returns None if not parseable or T == 0.
    """
    if not isinstance(s, str):
        s = str(s or "")
    m = re.search(r"\(\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\)", s)
    if not m:
        return None
    w, p, t = map(int, m.groups())
    if t <= 0:
        return None
    return round((w + p) / t, 3)  # proportion (0..1)

def _avg_last(values: List[int], k: int) -> Optional[float]:
    vals = [int(v) for v in values if int(v) != 0]
    if not vals:
        return None
    use = vals[:k] if len(vals) >= k else vals
    return round(float(np.mean(use)), 2)

def _avg_all(values: List[int]) -> Optional[float]:
    vals = [int(v) for v in values if int(v) != 0]
    return round(float(np.mean(vals)), 2) if vals else None

# =============================================================================
# BOX A — PARSERS (Run Style, Official Ratings, Race Class)
# =============================================================================

def split_box_a_sections(raw: str) -> Tuple[str, str, str, Dict[str, str]]:
    """
    Split the big paste into three sections using the markers:
    'Run Style Figure', 'Official Ratings', 'Race Class - Today:'
    Returns (rs_text, or_text, class_text, meta) where meta may contain 'today_class_value'.
    """
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")
    # find markers
    m_rs = re.search(r"(?is)run\s*style\s*figure\s*\n", txt)
    m_or = re.search(r"(?is)\nofficial\s*ratings\s*\n", txt)
    m_cl = re.search(r"(?is)\nrace\s*class\s*-\s*today:\s*class\s*\d+\s*£?[\d\.]+k?\s*\n", txt)

    if not m_rs or not m_or or not m_cl:
        return "", "", "", {}

    i_rs = m_rs.end()
    i_or = m_or.start()
    rs_text = txt[i_rs:i_or].strip()

    i_or_start = m_or.end()
    i_cl = m_cl.start()
    or_text = txt[i_or_start:i_cl].strip()

    i_cl_start = m_cl.end()
    class_text = txt[i_cl_start:].strip()

    # extract today class value in £ (optional)
    head = txt[m_cl.start():m_cl.end()]
    meta = {}
    m_val = re.search(r"today:\s*class\s*\d+\s*£?\s*([\d\.]+)\s*[kK]?", head, re.I)
    if m_val:
        val = m_val.group(1)
        # normalise £xK to numeric pounds if possible; assume '3K' -> 3000
        try:
            pounds = float(val) * 1000.0
            meta["today_class_value"] = int(round(pounds))
        except Exception:
            pass
    return rs_text, or_text, class_text, meta

def parse_runstyle_table(rs_text: str) -> pd.DataFrame:
    df = _safe_read_table(rs_text)
    if df.empty:
        return df

    # Standardise column names
    df.columns = [str(c).strip() for c in df.columns]
    if "Horse" not in df.columns:
        # try first col named like horse
        df.rename(columns={df.columns[0]: "Horse"}, inplace=True)

    # ensure Lto1..Lto10 exist
    for i in range(1, 11):
        c = f"Lto{i}"
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # RS_Avg = avg of last 5 (ignore zeros); RS_Avg10 = avg of last 10 (ignore zeros)
    arr5 = [df[f"Lto{i}"] for i in range(1, 6)]
    arr10 = [df[f"Lto{i}"] for i in range(1, 11)]
    def _avg_row(vals):
        v = [int(x) for x in vals if int(x) != 0]
        return round(float(np.mean(v)), 2) if v else np.nan

    df["RS_Avg"] = [ _avg_row([arr5[j][k] for j in range(5)]) for k in range(len(df)) ]
    df["RS_Avg10"] = [ _avg_row([arr10[j][k] for j in range(10)]) for k in range(len(df)) ]

    # RS_Cat from RS_Avg
    def _cat(x):
        if pd.isna(x): return "Unknown"
        if x < 1.6: return "Front"
        if x < 2.4: return "Prominent"
        if x < 3.0: return "Mid"
        return "Hold-up"
    df["RS_Cat"] = df["RS_Avg"].apply(_cat)

    # Keep useful columns in a stable order
    keep = ["Horse"] + [f"Lto{i}" for i in range(1, 11)] + ["RS_Avg", "RS_Avg10", "RS_Cat"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]

def parse_official_ratings(or_text: str) -> pd.DataFrame:
    """
    Expect columns like: Horse, To Last, Today, Last, Highest, Lto1..Lto10 (stacked pairs below each horse are OK)
    We'll keep: today_or, or_3back, max_or_10, hwin_or (mapped from 'Highest' if present)
    """
    df = _safe_read_table(or_text)
    if df.empty:
        return pd.DataFrame(columns=["Horse","today_or","or_3back","max_or_10","hwin_or"])

    # normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "horse" not in df.columns:
        df.rename(columns={df.columns[0]: "horse"}, inplace=True)

    out = pd.DataFrame()
    out["Horse"] = df["horse"].astype(str).str.strip()
    # map expected columns
    def grab(colnames):
        for c in colnames:
            if c in df.columns:
                return df[c]
        return pd.Series([None]*len(df))
    today = grab(["today", "or today", "or_today"])
    last3 = grab(["lto3","lto_3","or 3back","or_3back"])
    highest = grab(["highest", "highest win", "highest_win", "hwin_or"])

    # derive numeric fields
    out["today_or"] = today.apply(_to_int)
    # or_3back: if not explicit, try to parse last 10 stacked? here fallback None
    try:
        out["or_3back"] = last3.apply(lambda s: _to_int(str(s).split("/")[0]) if isinstance(s, str) and "/" in str(s) else _to_int(s))
    except Exception:
        out["or_3back"] = None
    out["max_or_10"] = None  # not reliable from this paste; leave empty unless present
    out["hwin_or"] = highest.apply(_to_int)

    return out

def parse_race_class(class_text: str, today_class_from_header: Optional[int]) -> pd.DataFrame:
    """
    Expect a table with Horse and per-run class/prize strings, plus Avg 3 / Avg 5 columns.
    We keep: avg3_prize, avg5_prize, today_class_value, cd, ccs_5
    """
    df = _safe_read_table(class_text)
    if df.empty:
        return pd.DataFrame(columns=["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"])

    # normalise
    cols = {c: str(c).strip().lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    if "horse" not in df.columns:
        df.rename(columns={df.columns[0]: "horse"}, inplace=True)

    out = pd.DataFrame()
    out["Horse"] = df["horse"].astype(str).str.strip()

    def pull_money(col_name):
        if col_name not in df.columns:
            return [None]*len(df)
        # values like '£3.52K' or '£4.27K'
        res = []
        for x in df[col_name]:
            s = str(x)
            m = re.search(r"£?\s*([\d\.]+)\s*[kK]?", s)
            if not m:
                res.append(None)
            else:
                try:
                    pounds = float(m.group(1)) * 1000.0
                    res.append(int(round(pounds)))
                except Exception:
                    res.append(None)
        return res

    out["avg3_prize"] = pull_money("avg 3")
    out["avg5_prize"] = pull_money("avg 5")

    out["today_class_value"] = today_class_from_header if today_class_from_header is not None else None

    # Optional CD/CCS if present
    # If your input already computed CD/CCS (0..1 or 1..5), map them; else None.
    out["cd"] = None
    out["ccs_5"] = None
    return out

# =============================================================================
# BOX B — single paste parser (Speed & Conditions) with NH safety
# =============================================================================

BOX_ORDER = [
    "Win %","Form Figures (Avg)","Speed Figures (Avg)","Crs %","Dist %","LHRH %","Going %","Class %","Run. +/-1 %","TrackStyle %"
]

def _is_win_line(s: str) -> bool:
    return bool(re.search(r"\(\s*\d+\s*/\s*\d+\s*/\s*\d+\s*\)", str(s)))

def _looks_like_name(lines_or_str, idx: Optional[int] = None) -> bool:
    """
    Treat line as a horse name only if:
      - does NOT contain (), /, £, commas, or digits
      - the next non-empty line looks like a Win% line '(W/P/T)'
    """
    if isinstance(lines_or_str, list):
        lines = lines_or_str
        if idx is None or idx < 0 or idx >= len(lines):
            return False
        t = lines[idx].strip()
        if not t or t.lower().startswith("horse"):
            return False
        if re.search(r"[(),/£]", t) or re.search(r"\d", t):
            return False
        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        return _is_win_line(lines[j]) if j < len(lines) else False
    else:
        t = str(lines_or_str).strip()
        if not t or t.lower().startswith("horse"):
            return False
        if re.search(r"[(),/£]", t) or re.search(r"\d", t):
            return False
        return True

def parse_box45_single(raw: str) -> pd.DataFrame:
    lines = [ln for ln in raw.replace("\r\n","\n").replace("\r","\n").split("\n") if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["Horse"])
    if lines[0].strip().lower().startswith("horse"):
        lines = lines[1:]

    i, n = 0, len(lines)
    recs = []
    while i < n:
        while i < n and not _looks_like_name(lines, i):
            i += 1
        if i >= n:
            break
        name = lines[i].strip()
        i += 1

        vals = []
        while i < n and len(vals) < 10:
            if _looks_like_name(lines, i):
                break
            vals.append(lines[i].strip())
            i += 1
        if len(vals) < 10:
            vals += [None]*(10-len(vals))

        rec = {"Horse": name}
        for k, v in zip(BOX_ORDER, vals):
            rec[k] = v
        recs.append(rec)

    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["Horse"] = df["Horse"].astype(str).str.strip()
    return df

def _extract_speed_list(s: str) -> List[int]:
    # line like "68, 73, 81, 74, ... (71)"
    nums = [int(x) for x in re.findall(r"\d+", str(s or ""))]
    # remove the last trailing average if it looks like " (71)" appended
    # Keep all; we’ll compute last/high/avg3 ourselves from the list
    return nums

def build_adj_speed_and_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute KeySpeedAvg and condition place rates."""
    if df.empty:
        return df

    out = pd.DataFrame()
    out["Horse"] = df["Horse"]

    # KeySpeedAvg = mean(Last, Highest, Avg of last3) from Speed Figures list
    def _keyspeed(row):
        arr = _extract_speed_list(row.get("Speed Figures (Avg)"))
        if not arr:
            return None
        last = arr[-2] if len(arr) >= 2 else arr[-1]  # handle the last number being the average in parentheses
        highest = max(arr) if arr else None
        avg3 = round(np.mean(arr[-3:]), 1) if len(arr) >= 3 else float(last)
        vals = [v for v in [last, highest, avg3] if v is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    out["KeySpeedAvg"] = df.apply(_keyspeed, axis=1)

    # Place rates from the W/P/T tuples
    mapping = {
        "Crs %": "crs_place",
        "Dist %": "dist_place",
        "LHRH %": "lhrh_place",
        "Going %": "going_place",
        "Class %": "cls_place",
        "Run. +/-1 %": "runpm1_place",
        "TrackStyle %": "trackstyle_place",
    }
    for src, dst in mapping.items():
        out[dst] = df[src].apply(_pct_from_tuple) if src in df.columns else None

    # Keep raw averages if desired
    out["LastRace"] = None
    out["Highest"] = None
    out["Avg3"] = None
    out["AvgAll"] = None
    # If you want to parse "Speed Figures (Avg)" further, you can add those here.

    return out

# =============================================================================
# PACE / SUITABILITY ENGINE (same as your pace_form_app logic, compacted)
# =============================================================================

@dataclass
class HorseRow:
    horse: str
    run_styles: List[int]  # we will simulate from RS_Avg
    adj_speed: Optional[float]
    or_today: Optional[int] = None
    or_highest_win: Optional[int] = None

@dataclass
class Settings:
    class_par: float = 77.0
    distance_f: float = 7.0
    front_thr: float = 1.6
    prom_thr: float = 2.4
    mid_thr: float = 3.0
    lcp_high: float = -3.0
    lcp_question_low: float = -8.0
    wp_even: float = 0.5
    wp_confident: float = 0.65

PACE_ORDER = ["Slow", "Even", "Strong", "Very Strong"]
PACEFIT = {
    "Slow":   {"Front": 5, "Prominent": 4, "Mid": 3, "Hold-up": 2},
    "Even":   {"Front": 4, "Prominent": 5, "Mid": 4, "Hold-up": 3},
    "Strong": {"Front": 2, "Prominent": 3, "Mid": 4, "Hold-up": 5},
    "Very Strong": {"Front": 1, "Prominent": 2, "Mid": 4, "Hold-up": 5},
}
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
PACEFIT_EVEN_FC = {"Front": 4.5, "Prominent": 5.0, "Mid": 3.5, "Hold-up": 2.5}

def _dist_band(d: float) -> str:
    if d <= 5.5: return "5f"
    if d <= 6.5: return "6f"
    return "route"

def avg_style(vals: List[int]) -> float:
    v = [int(x) for x in vals if int(x) != 0]
    return float(np.mean(v)) if v else np.nan

def classify_style(avg: float, s: Settings) -> str:
    if np.isnan(avg): return "Unknown"
    if avg < s.front_thr: return "Front"
    if avg < s.prom_thr: return "Prominent"
    if avg < s.mid_thr: return "Mid"
    return "Hold-up"

def delta_vs_par(adj_speed: Optional[float], class_par: float) -> Optional[float]:
    return None if adj_speed is None else round(float(adj_speed) - class_par, 1)

def lcp_from_delta(style: str, dvp: Optional[float], s: Settings) -> str:
    if style not in ("Front","Prominent") or dvp is None:
        return "N/A"
    if dvp >= s.lcp_high: return "High"
    if dvp >= s.lcp_question_low: return "Questionable"
    return "Unlikely"

def speed_score(dvp: Optional[float]) -> float:
    if dvp is None: return 2.3
    if dvp >= 2: return 5.0
    if dvp >= -1: return 4.0
    if dvp >= -4: return 3.0
    if dvp >= -8: return 2.0
    return 1.0

def project_pace(rows: List[HorseRow], s: Settings) -> Tuple[str, float, Dict[str,str], Dict[str,object]]:
    debug = {"rules_applied": []}
    recs = []
    for r in rows:
        avg = avg_style(r.run_styles)
        style = classify_style(avg, s)
        dvp = delta_vs_par(r.adj_speed, s.class_par)
        lcp  = lcp_from_delta(style, dvp, s)
        recs.append(dict(horse=r.horse, style=style, dvp=dvp, lcp=lcp))
    d = pd.DataFrame(recs)
    if d.empty:
        return "N/A", 0.0, {}, {"error":"no rows"}

    front_all = d[d["style"]=="Front"]
    prom_all  = d[d["style"]=="Prominent"]
    front_high= d[(d["style"]=="Front")&(d["lcp"]=="High")]
    prom_high = d[(d["style"]=="Prominent")&(d["lcp"]=="High")]
    front_q   = d[(d["style"]=="Front")&(d["lcp"]=="Questionable")]
    prom_q    = d[(d["style"]=="Prominent")&(d["lcp"]=="Questionable")]

    n_front_high = len(front_high); n_prom_high=len(prom_high)
    n_front_q = len(front_q); n_prom_q=len(prom_q)

    W_FH, W_PH, W_FQ, W_PQ = 2.0, 0.8, 0.5, 0.2
    early_energy = (W_FH*n_front_high)+(W_PH*n_prom_high)+(W_FQ*n_front_q)+(W_PQ*n_prom_q)

    if n_front_high >= 2:
        scenario, conf = "Strong", 0.65
        debug["rules_applied"].append("Base: ≥2 High Front → Strong")
    elif (n_front_high+n_prom_high)>=3 and early_energy>=3.2:
        scenario, conf = "Strong", 0.65
        debug["rules_applied"].append("Base: ≥3 High early & energy≥3.2 → Strong")
    elif early_energy>=3.2:
        scenario, conf = "Strong", 0.60
        debug["rules_applied"].append("Base: energy≥3.2 → Strong")
    elif (n_front_high+n_prom_high)>=2:
        scenario, conf = "Even", 0.60
    elif (n_front_high+n_prom_high)==1 and (n_front_q+n_prom_q)>=1:
        scenario, conf = "Even", 0.55
    elif (n_front_high+n_prom_high)==1:
        scenario, conf = "Even", 0.60
    elif (n_front_q+n_prom_q)>=1:
        scenario, conf = "Slow", 0.60
    else:
        scenario, conf = "Slow", 0.70

    # No-front cap
    if len(d[d["style"]=="Front"])==0 or n_front_high==0:
        allow_strong=False
        if n_prom_high>=3:
            try:
                allow_strong = float(prom_high["dvp"].mean())>=-1.0
            except Exception:
                allow_strong=False
        if not allow_strong and scenario in ("Strong","Very Strong"):
            scenario, conf = "Even", min(conf,0.60)
            debug["rules_applied"].append("No-front cap → Even")

    # Return maps
    debug.update({
        "counts":{
            "Front_all": int(len(d[d['style']=='Front'])),
            "Front_High": int(n_front_high),
            "Front_Questionable": int(n_front_q),
            "Prominent_High": int(n_prom_high),
            "Prominent_Questionable": int(n_prom_q),
        },
        "early_energy": float(early_energy),
    })
    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario, conf, lcp_map, debug

def suitability(rows: List[HorseRow], s: Settings) -> Tuple[pd.DataFrame, Dict[str,object]]:
    scenario, conf, lcp_map, debug = project_pace(rows, s)
    if scenario=="N/A": return pd.DataFrame(), debug
    band = _dist_band(s.distance_f)
    if band=="5f": pf = PACEFIT_5F
    elif band=="6f": pf = PACEFIT_6F
    else: pf = PACEFIT
    pacefit_map = pf.get(scenario, PACEFIT["Even"])
    wp = s.wp_confident if scenario in ("Slow","Very Strong") and conf>=0.65 else s.wp_even
    if band=="5f": wp=max(wp,0.60)
    elif band=="6f": wp=max(wp,0.55)
    ws=1-wp

    out=[]
    for r in rows:
        avg = avg_style(r.run_styles)
        style = classify_style(avg, s)
        dvp = delta_vs_par(r.adj_speed, s.class_par)
        pacefit = pacefit_map.get(style, 3)
        speedfit = speed_score(dvp)
        score = round(pacefit*wp + speedfit*ws, 1)
        out.append({
            "Horse": r.horse,
            "AvgStyle": round(avg,2),
            "Style": style,
            "ΔvsPar": dvp,
            "LCP": lcp_map.get(r.horse, "N/A"),
            "PaceFit": pacefit,
            "SpeedFit": speedfit,
            "wp": wp, "ws": ws,
            "Suitability": score,
            "Scenario": scenario, "Confidence": conf,
        })
    df = pd.DataFrame(out).sort_values(["Suitability","SpeedFit"], ascending=False)
    return df, debug

# =============================================================================
# UI — TABS
# =============================================================================

tabA, tabB = st.tabs(["A) Normalize & Merge (A + B)", "B) Suitability from Combined CSV"])

# ---------------------- TAB A ----------------------
with tabA:
    st.subheader("A) Normalize & Merge")
    c1, c2 = st.columns(2)
    with c1:
        raw_a = st.text_area("Box A — Paste **Run Style Figure + Official Ratings + Race Class - Today:**", height=320)
    with c2:
        raw_b = st.text_area("Box B — Paste **Speed & Conditions** (same paste used for boxes 4 & 5)", height=320)

    if st.button("Process A + B"):
        if not raw_a.strip() or not raw_b.strip():
            st.warning("Please paste both Box A and Box B.")
        else:
            try:
                rs_text, or_text, class_text, meta = split_box_a_sections(raw_a)

                rs_df = parse_runstyle_table(rs_text)
                or_df = parse_official_ratings(or_text)
                rc_df = parse_race_class(class_text, meta.get("today_class_value"))

                st.success("Box A parsed")
                st.write("Run Style", rs_df)
                st.write("Official Ratings (key fields)", or_df)
                st.write("Race Class (key fields)", rc_df)

                # Box B
                b_raw_df = parse_box45_single(raw_b)
                b_df = build_adj_speed_and_conditions(b_raw_df)
                st.success("Box B parsed")
                st.write("Speed & Conditions (KeySpeedAvg + place rates)", b_df)

                # Merge A pieces first (outer joins by Horse)
                a_merge = rs_df[["Horse","RS_Avg","RS_Avg10","RS_Cat"]].merge(
                    or_df, on="Horse", how="left"
                ).merge(
                    rc_df, on="Horse", how="left"
                )

                combined = a_merge.merge(b_df, on="Horse", how="left")

                st.markdown("### 🧩 Combined Output (A + B)")
                st.dataframe(combined, use_container_width=True, hide_index=True)

                st.download_button(
                    "💾 Download Combined CSV",
                    combined.to_csv(index=False).encode("utf-8"),
                    file_name="combined_A_B.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Failed: {e}")

# ---------------------- TAB B ----------------------
with tabB:
    st.subheader("B) Suitability from Combined CSV")
    st.caption("Upload the **Combined (A + B)** CSV you downloaded from Tab A.")
    f = st.file_uploader("Upload combined_A_B.csv", type=["csv"], key="combined_upload")

    st.sidebar.header("Race Settings")
    s = Settings()
    s.class_par = st.sidebar.number_input("Class Par", value=float(s.class_par))
    s.distance_f = st.sidebar.number_input("Distance (f)", value=float(s.distance_f))

    st.sidebar.header("Pace thresholds")
    s.front_thr = st.sidebar.number_input("Front threshold (<)", value=float(s.front_thr), step=0.1)
    s.prom_thr = st.sidebar.number_input("Prominent threshold (<)", value=float(s.prom_thr), step=0.1)
    s.mid_thr = st.sidebar.number_input("Mid threshold (<)", value=float(s.mid_thr), step=0.1)

    st.sidebar.header("LCP thresholds (Δ vs Par)")
    s.lcp_high = st.sidebar.number_input("High ≥", value=float(s.lcp_high), step=0.5)
    s.lcp_question_low = st.sidebar.number_input("Questionable lower bound", value=float(s.lcp_question_low), step=0.5)

    st.sidebar.header("Weights")
    s.wp_even = st.sidebar.slider("wp (Even/Uncertain)", 0.3, 0.8, float(s.wp_even), 0.05)
    s.wp_confident = st.sidebar.slider("wp (Predictable Slow/Very Strong)", 0.3, 0.8, float(s.wp_confident), 0.05)

    if not f:
        st.info("Upload the Combined CSV to run the pace & suitability model.")
    else:
        try:
            df = pd.read_csv(f)
            # normalise column names for mapping
            lower = {c: c for c in df.columns}  # keep original
            aliases = {str(c).strip().lower(): c for c in df.columns}

            def col(*cands):
                for c in cands:
                    if c in df.columns: return c
                # try lowercase aliases
                for c in cands:
                    if c.lower() in aliases: return aliases[c.lower()]
                return None

            # Required: Horse, RS_Avg, AdjSpeed (KeySpeedAvg)
            c_horse = col("Horse")
            c_rsavg = col("RS_Avg","RS_avg","rs_avg")
            c_adj   = col("AdjSpeed","KeySpeedAvg","keyspeedavg","key speed factors average")

            if not c_horse or not c_rsavg or not c_adj:
                st.error(f"Missing required columns. Found: {list(df.columns)}. Need: Horse, RS_Avg, and KeySpeedAvg/AdjSpeed.")
                st.stop()

            # Build HorseRows (simulate last-5 run styles from RS_Avg by repeating the nearest int)
            rows: List[HorseRow] = []
            for _, r in df.iterrows():
                name = str(r[c_horse]).strip()
                rsavg = r[c_rsavg]
                if pd.isna(rsavg):
                    style_list = [3,3,3,3,3]  # default mid if missing
                else:
                    v = int(round(float(rsavg)))
                    v = max(1, min(4, v))
                    style_list = [v]*5
                adj = None if pd.isna(r[c_adj]) else float(r[c_adj])
                # Optional OR fields if present
                c_today = col("today_or","or_today","today")
                c_hwin  = col("hwin_or","highest","highest win or")
                or_today = int(r[c_today]) if c_today and not pd.isna(r[c_today]) else None
                hwin_or  = int(r[c_hwin]) if c_hwin and not pd.isna(r[c_hwin]) else None

                rows.append(HorseRow(name, style_list, adj, or_today, hwin_or))

            res, debug = suitability(rows, s)
            if res.empty:
                st.warning("Could not compute suitability (no rows).")
            else:
                scenario = res.iloc[0]["Scenario"]
                conf = float(res.iloc[0]["Confidence"])
                st.subheader(f"Projected Pace: {scenario} (confidence {conf:.2f})")

                with st.expander("Why this pace? (Reason used)", expanded=True):
                    counts = debug.get("counts", {})
                    st.markdown(
                        f"- **Front (all):** {counts.get('Front_all', 0)} &nbsp;&nbsp; "
                        f"**Front (High):** {counts.get('Front_High', 0)} &nbsp;&nbsp; "
                        f"**Front (Questionable):** {counts.get('Front_Questionable', 0)}  \n"
                        f"- **Prominent (High):** {counts.get('Prominent_High', 0)} &nbsp;&nbsp; "
                        f"**Prominent (Questionable):** {counts.get('Prominent_Questionable', 0)}  \n"
                        f"- **Early energy:** {debug.get('early_energy', 0.0):.2f}"
                    )

                st.markdown("### Suitability Ratings (all runners)")
                show_cols = ["Horse","AvgStyle","Style","ΔvsPar","LCP","PaceFit","SpeedFit","wp","ws","Suitability"]
                show_cols = [c for c in show_cols if c in res.columns]
                st.dataframe(res[show_cols], use_container_width=True, hide_index=True)

                st.download_button(
                    "💾 Download Suitability CSV",
                    res.to_csv(index=False).encode("utf-8"),
                    file_name="suitability_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Failed to run suitability: {e}")
