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

def _numbers_ignore_brackets(s: str) -> List[float]:
    """
    Speed series helper: ignore anything in parentheses, e.g.
    '77, 85, 83 (82)' -> [77, 85, 83]
    because (82) is the average, not an extra run.
    """
    s2 = re.sub(r"\([^)]*\)", "", str(s))
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s2)]

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
    for c in lto10:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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
    if not m:
        return None, None, None, None
    w = int(m.group(1)); p = int(m.group(2)); t = int(m.group(3))
    place = None if t == 0 else round((w+p)/t, 3)
    return place, w, p, t

def _parse_speed_series(s: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    nums = _numbers(s)
    if not nums:
        return None, None, None, None
    last = nums[-1]; highest = max(nums)
    avg3 = round(sum(nums[-3:])/min(3, len(nums)), 1)
    avgall = round(sum(nums)/len(nums), 1)
    return last, highest, avg3, avgall

def parse_box45_single(raw: str) -> pd.DataFrame:
    lines = [ln for ln in raw.replace("\r\n","\n").replace("\r","\n").split("\n") if ln.strip()]
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
    if df.empty:
        return df
    df["Horse"] = df["Horse"].astype(str).str.strip()
    return df

def build_speed_conditions(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw.copy()
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

    # keep totals per condition too
    for col in ["crs","dist","lhrh","going","cls","runpm1","trackstyle"]:
        if col in out.columns:
            parsed = [_parse_wpt_value(v) for v in out[col].fillna("")]
            out[f"{col}_place"] = [p[0] for p in parsed]
            out[f"{col}_total"] = [p[3] for p in parsed]

    keep = ["Horse",
            "SpeedRunsRaw","SpeedRunsList",
            "LastRace","Highest","Avg3","AvgAll","KeySpeedAvg",
            "crs_place","dist_place","lhrh_place","going_place","cls_place","runpm1_place","trackstyle_place",
            "crs_total","dist_total","lhrh_total","going_total","cls_total","runpm1_total","trackstyle_total"]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

# =========================
# NEW: Betfair Exchange prices parser (Back all first price)
# =========================
def parse_betfair_backall(raw: str) -> pd.DataFrame:
    """
    Parse a Betfair Exchange 'Back all' ladder paste and extract:
      Horse name + best 'Back all' price (the first price shown after jockey line).
    Returns: DataFrame with columns ["Horse","MarketOdds"].
    """
    if not raw or not str(raw).strip():
        return pd.DataFrame(columns=["Horse", "MarketOdds"])

    lines = [ln.strip() for ln in str(raw).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()]

    def looks_like_horse_name(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        low = s.lower()
        bad = {"back all", "lay all", "bsp", "sp", "matched", "unmatched", "back", "lay"}
        if low in bad:
            return False
        # horse names: no digits, no €, no commas, no parentheses
        if "€" in s:
            return False
        if "," in s:
            return False
        if re.search(r"\d", s):
            return False
        if re.search(r"[\(\)]", s):
            return False
        return len(s) >= 3

    def looks_like_jockey_line(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if "€" in s:
            return False
        # allow (5) claims; disallow other digit patterns
        if re.search(r"\d", s) and not re.search(r"\(\s*\d+\s*\)", s):
            return False
        return bool(re.search(r"[A-Za-z]", s))

    def first_price_ahead(start_idx: int, lookahead: int = 14) -> Optional[float]:
        for j in range(start_idx, min(len(lines), start_idx + lookahead)):
            txt = lines[j]
            if "%" in txt:
                continue
            m = re.search(r"\b(\d+(?:\.\d+)?)\b", txt)
            if m:
                try:
                    val = float(m.group(1))
                    if val >= 1.01:
                        return val
                except Exception:
                    pass
        return None

    recs = []
    i = 0
    n = len(lines)
    while i < n:
        if looks_like_horse_name(lines[i]):
            horse = lines[i].strip()
            jline = lines[i + 1].strip() if i + 1 < n else ""
            if looks_like_jockey_line(jline):
                price = first_price_ahead(i + 2, lookahead=14)
                if price is not None:
                    recs.append({"Horse": horse, "MarketOdds": price})
                    i += 2
                    continue
        i += 1

    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["Horse", "MarketOdds"])

    df["Horse"] = df["Horse"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["Horse"], keep="first").reset_index(drop=True)
    return df

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

def should_use_hybrid(distance_f: float,
                      scenario: str,
                      confidence: float,
                      counts: dict) -> bool:
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

def late_strong_warn_hybrid_only_6f(distance_f: float,
                                   use_hybrid: bool,
                                   confidence: float,
                                   counts: dict,
                                   energy: float) -> bool:
    if not use_hybrid:
        return False
    if not (5.5 < float(distance_f) <= 6.5):
        return False

    FH = int(counts.get("Front_High", 0))
    PH = int(counts.get("Prominent_High", 0))
    total_high = FH + PH
    e = float(energy)
    c = float(confidence) if confidence is not None else 0.0

    if total_high < 2:
        return False
    if FH >= 3:
        return False
    if 0.55 <= c <= 0.85 and e >= 3.2:
        return True
    return False

# ==========================================================
# NEW: Pace Trigger Matrix helpers (logic-only, no UI change)
# ==========================================================
def _route_subband(distance_f: float) -> str:
    """
    Internal segmentation for route races.
    UI band stays 'route' (no UI changes).
    """
    d = float(distance_f)
    if d <= 7.5:
        return "7f"
    if d <= 9.5:
        return "R8-9"     # ~1m to 1m2
    if d <= 12.5:
        return "R10-12"   # ~1m2 to 1m4½
    return "R13+"         # ~1m5+

def _scenario_step_down(scn: str) -> str:
    """Downgrade one level: Very Strong -> Strong -> Even -> Slow (Even tags count as Even)."""
    if scn == "Very Strong":
        return "Strong"
    if scn == "Strong":
        return "Even"
    if str(scn).startswith("Even"):
        return "Slow"
    return scn

# ==========================================================
# NEW: Pace Trigger Matrix (distance-segmented)
# Returns scenario, confidence, debug rules (same interface)
# ==========================================================
def project_pace_from_rows(rows: List[HorseRow], s: Settings) -> Tuple[str,float,Dict[str,str],Dict[str,object]]:
    debug: Dict[str, object] = {"rules_applied": []}
    if not rows:
        return "N/A", 0.0, {}, {"error": "no rows", "rules_applied": []}

    d = pd.DataFrame([{"horse": r.horse, "style": r.style_cat, "dvp": r.dvp, "lcp": r.lcp} for r in rows])

    front_all = d[d["style"] == "Front"]
    prom_all  = d[d["style"] == "Prominent"]

    front_high = d[(d["style"] == "Front") & (d["lcp"] == "High")]
    prom_high  = d[(d["style"] == "Prominent") & (d["lcp"] == "High")]

    front_q = d[(d["style"] == "Front") & (d["lcp"] == "Questionable")]
    prom_q  = d[(d["style"] == "Prominent") & (d["lcp"] == "Questionable")]

    n_front = int(len(front_all))
    n_fh = int(len(front_high))
    n_ph = int(len(prom_high))
    n_fq = int(len(front_q))
    n_pq = int(len(prom_q))

    high_early = n_fh + n_ph
    q_early = n_fq + n_pq

    # Energy weights (kept consistent with your engine style)
    W_FH, W_PH, W_FQ, W_PQ = 2.0, 0.8, 0.5, 0.2
    energy = (W_FH * n_fh) + (W_PH * n_ph) + (W_FQ * n_fq) + (W_PQ * n_pq)

    band = _dist_band(s.distance_f)
    route_sb = _route_subband(s.distance_f) if band == "route" else None

    dvp_front_high: Optional[float] = None
    if n_fh == 1:
        try:
            dvp_front_high = float(front_high["dvp"].iloc[0])
        except Exception:
            dvp_front_high = None

    allow_strong_no_front = False
    if n_front == 0 and n_ph >= 4:
        try:
            allow_strong_no_front = float(prom_high["dvp"].mean()) >= -1.0
        except Exception:
            allow_strong_no_front = False

    scenario: str = "Slow"
    conf: float = 0.70

    # -------------------------
    # 5f
    # -------------------------
    if band == "5f":
        if n_fh >= 3:
            scenario, conf = "Very Strong", 0.80
            debug["rules_applied"].append("5f: ≥3 High-LCP Fronts → Very Strong (locked)")
        elif (n_fh >= 2 and high_early >= 4):
            scenario, conf = "Very Strong", 0.75
            debug["rules_applied"].append("5f: ≥2 High Front + HighEarly≥4 → Very Strong")
        elif n_fh == 2:
            scenario, conf = "Strong", 0.65
            debug["rules_applied"].append("5f: 2 High Front → Strong")
        elif (high_early >= 3 and energy >= 3.2):
            scenario, conf = "Strong", 0.65
            debug["rules_applied"].append("5f: HighEarly≥3 & energy≥3.2 → Strong")
        elif high_early == 2:
            scenario, conf = "Even", 0.60
            debug["rules_applied"].append("5f: HighEarly==2 → Even")
        elif (high_early == 1 and q_early >= 1):
            scenario, conf = "Even", 0.55
            debug["rules_applied"].append("5f: 1 High + Questionables → Even")
        elif (high_early == 1 and q_early == 0):
            scenario, conf = "Slow", 0.65
            debug["rules_applied"].append("5f: Single early only → Slow")
        else:
            scenario, conf = "Slow", 0.70
            debug["rules_applied"].append("5f: No credible early → Slow")

        # sprint cap: don't call it Slow if there are ≥2 credible pace angles
        if scenario == "Slow" and high_early >= 2:
            scenario, conf = "Even", max(conf, 0.60)
            debug["rules_applied"].append("5f cap: HighEarly≥2 → not Slow")

        # tag front-controlled for "single solid leader" structures
        if scenario in ("Even", "Slow"):
            if (n_fh == 1 and n_ph <= 1 and dvp_front_high is not None and dvp_front_high >= -1.0):
                scenario = "Even (Front-Controlled)"
                debug["rules_applied"].append("Front-controlled tag")

    # -------------------------
    # 6f
    # -------------------------
    elif band == "6f":
        if n_fh >= 3:
            scenario, conf = "Very Strong", 0.80
            debug["rules_applied"].append("6f: ≥3 High-LCP Fronts → Very Strong (locked)")
        elif n_fh >= 2:
            scenario, conf = "Strong", 0.65
            debug["rules_applied"].append("6f: ≥2 High Front → Strong")
        elif (high_early >= 3 and energy >= 3.2):
            scenario, conf = "Strong", 0.65
            debug["rules_applied"].append("6f: HighEarly≥3 & energy≥3.2 → Strong")
        elif energy >= 3.6:
            scenario, conf = "Strong", 0.60
            debug["rules_applied"].append("6f: energy≥3.6 → Strong")
        elif high_early == 2:
            scenario, conf = "Even", 0.60
            debug["rules_applied"].append("6f: HighEarly==2 → Even")
        elif (high_early == 1 and q_early >= 1):
            scenario, conf = "Even", 0.55
            debug["rules_applied"].append("6f: 1 High + Questionables → Even")
        elif (high_early == 1 and q_early == 0):
            scenario, conf = "Slow", 0.65
            debug["rules_applied"].append("6f: Single early only → Slow")
        else:
            scenario, conf = "Slow", 0.70
            debug["rules_applied"].append("6f: No credible early → Slow")

        if scenario == "Slow" and high_early >= 2:
            scenario, conf = "Even", max(conf, 0.60)
            debug["rules_applied"].append("6f cap: HighEarly≥2 → not Slow")

        if scenario in ("Even", "Slow"):
            if (n_fh == 1 and n_ph <= 1 and dvp_front_high is not None and dvp_front_high >= -1.0):
                scenario = "Even (Front-Controlled)"
                debug["rules_applied"].append("Front-controlled tag")

        debug["late_strong_risk"] = bool(high_early >= 3 and n_fh in (1, 2) and energy >= 3.2)
        if debug["late_strong_risk"]:
            debug["rules_applied"].append("6f: Late-Strong risk flag (HighEarly≥3 & FH∈{1,2} & energy≥3.2)")

    # -------------------------
    # Route (internally segmented)
    # -------------------------
    else:
        assert route_sb is not None

        if route_sb == "7f":
            if n_fh >= 3:
                scenario, conf = "Very Strong", 0.80
                debug["rules_applied"].append("7f: ≥3 High-LCP Fronts → Very Strong (locked)")
            elif n_fh >= 2:
                scenario, conf = "Strong", 0.65
                debug["rules_applied"].append("7f: ≥2 High Front → Strong")
            elif (high_early >= 3 and energy >= 3.2):
                scenario, conf = "Strong", 0.65
                debug["rules_applied"].append("7f: HighEarly≥3 & energy≥3.2 → Strong")
            elif (energy >= 3.2 and high_early >= 2):
                scenario, conf = "Strong", 0.60
                debug["rules_applied"].append("7f: energy≥3.2 & HighEarly≥2 → Strong")
            elif high_early == 2:
                scenario, conf = "Even", 0.60
                debug["rules_applied"].append("7f: HighEarly==2 → Even")
            elif (high_early == 1 and q_early >= 1):
                scenario, conf = "Even", 0.55
                debug["rules_applied"].append("7f: 1 High + Questionables → Even")
            elif (high_early == 1 and q_early == 0):
                if (dvp_front_high is not None) and (dvp_front_high >= -1.0):
                    scenario, conf = "Even (Front-Controlled)", 0.60
                    debug["rules_applied"].append("7f: Single solid leader → Even (Front-Controlled)")
                else:
                    scenario, conf = "Slow", 0.65
                    debug["rules_applied"].append("7f: Single leader not solid → Slow")
            else:
                scenario, conf = "Slow", 0.70
                debug["rules_applied"].append("7f: No credible early → Slow")

        elif route_sb == "R8-9":
            if n_fh >= 3:
                scenario, conf = "Very Strong", 0.80
                debug["rules_applied"].append("R8-9: ≥3 High-LCP Fronts → Very Strong (locked)")
            elif n_fh >= 2:
                scenario, conf = "Strong", 0.65
                debug["rules_applied"].append("R8-9: ≥2 High Front → Strong")
            elif (high_early >= 3 and energy >= 3.2):
                scenario, conf = "Strong", 0.65
                debug["rules_applied"].append("R8-9: HighEarly≥3 & energy≥3.2 → Strong")
            elif (energy >= 3.6 and high_early >= 3):
                scenario, conf = "Strong", 0.60
                debug["rules_applied"].append("R8-9: energy≥3.6 & HighEarly≥3 → Strong")
            elif high_early == 2:
                scenario, conf = "Even", 0.60
                debug["rules_applied"].append("R8-9: HighEarly==2 → Even")
            elif (high_early == 1 and q_early >= 1):
                scenario, conf = "Even", 0.55
                debug["rules_applied"].append("R8-9: 1 High + Questionables → Even")
            elif (high_early == 1 and q_early == 0):
                if (dvp_front_high is not None) and (dvp_front_high >= -1.0):
                    scenario, conf = "Even (Front-Controlled)", 0.60
                    debug["rules_applied"].append("R8-9: Single solid leader → Even (Front-Controlled)")
                else:
                    scenario, conf = "Slow", 0.65
                    debug["rules_applied"].append("R8-9: Single leader not solid → Slow")
            else:
                scenario, conf = "Slow", 0.70
                debug["rules_applied"].append("R8-9: No credible early → Slow")

        elif route_sb == "R10-12":
            if (n_fh >= 3 and high_early >= 4):
                scenario, conf = "Very Strong", 0.75
                debug["rules_applied"].append("R10-12: FH≥3 & HighEarly≥4 → Very Strong")
            elif n_fh >= 3:
                scenario, conf = "Strong", 0.65
                debug["rules_applied"].append("R10-12: FH≥3 → Strong")
            elif (n_fh >= 2 and high_early >= 3):
                scenario, conf = "Strong", 0.60
                debug["rules_applied"].append("R10-12: FH≥2 & HighEarly≥3 → Strong")
            elif (high_early >= 4 and energy >= 3.2):
                scenario, conf = "Strong", 0.60
                debug["rules_applied"].append("R10-12: HighEarly≥4 & energy≥3.2 → Strong")
            elif high_early in (2, 3):
                scenario, conf = "Even", 0.60
                debug["rules_applied"].append("R10-12: HighEarly in {2,3} → Even")
            elif (high_early == 1 and q_early >= 1):
                scenario, conf = "Even", 0.55
                debug["rules_applied"].append("R10-12: 1 High + Questionables → Even")
            elif (high_early == 1 and q_early == 0):
                if (dvp_front_high is not None) and (dvp_front_high >= -1.5):
                    scenario, conf = "Even (Front-Controlled)", 0.60
                    debug["rules_applied"].append("R10-12: Single solid-ish leader → Even (Front-Controlled)")
                else:
                    scenario, conf = "Slow", 0.65
                    debug["rules_applied"].append("R10-12: Single leader not solid → Slow")
            else:
                scenario, conf = "Slow", 0.70
                debug["rules_applied"].append("R10-12: No credible early → Slow")

        else:  # R13+
            if (n_fh >= 3 and high_early >= 4):
                scenario, conf = "Strong", 0.60
                debug["rules_applied"].append("R13+: FH≥3 & HighEarly≥4 → Strong")
            elif (high_early >= 5 and energy >= 3.2):
                scenario, conf = "Strong", 0.60
                debug["rules_applied"].append("R13+: HighEarly≥5 & energy≥3.2 → Strong")
            elif high_early in (2, 3, 4):
                scenario, conf = "Even", 0.60
                debug["rules_applied"].append("R13+: HighEarly in {2,3,4} → Even")
            else:
                scenario, conf = "Slow", 0.70
                debug["rules_applied"].append("R13+: HighEarly in {0,1} → Slow")

            if scenario in ("Even", "Slow"):
                if (n_fh == 1 and n_ph == 0 and dvp_front_high is not None and dvp_front_high >= -1.0):
                    scenario = "Even (Front-Controlled)"
                    debug["rules_applied"].append("R13+: optional Front-controlled tag")

        # Route-global caps/corrections
        if n_front == 0 and scenario in ("Strong", "Very Strong") and not allow_strong_no_front:
            scenario, conf = "Even", min(conf, 0.60)
            debug["rules_applied"].append("No-front cap → Even")

        if n_front == 1:
            try:
                any_front_dvp = float(front_all["dvp"].iloc[0])
            except Exception:
                any_front_dvp = None
            if (any_front_dvp is not None) and (any_front_dvp <= -8.0):
                prev = scenario
                scenario = _scenario_step_down(scenario)
                conf = max(conf, 0.65)
                debug["rules_applied"].append(f"Weak solo leader → downgrade ({prev}→{scenario})")

        if scenario in ("Strong", "Very Strong") and n_fh == 1:
            if not (n_fh >= 2 or high_early >= 3):
                prev = scenario
                scenario, conf = "Even", max(conf, 0.60)
                debug["rules_applied"].append(f"Route single-front cap → Even ({prev}→Even)")

    debug.update({
        "counts": {
            "Front_all": int(n_front),
            "Front_High": int(n_fh),
            "Prominent_High": int(n_ph),
            "Front_Questionable": int(n_fq),
            "Prominent_Questionable": int(n_pq),
            "HighEarly": int(high_early),
            "QEarly": int(q_early),
        },
        "early_energy": float(energy),
        "distance_band": band,
        "route_subband": route_sb,
        "dvp_front_high": dvp_front_high,
        "allow_strong_no_front": bool(allow_strong_no_front),
    })
    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario, float(conf), lcp_map, debug

# =========================
# Diagnostics (separate; no effect on Suitability)
# =========================
COND_COLS = ["dist_place","cls_place","runpm1_place","trackstyle_place","crs_place","lhrh_place","going_place"]
COND_TOTAL_COLS = ["dist_total","cls_total","runpm1_total","trackstyle_total","crs_total","lhrh_total","going_total"]

def conditions_score_from_mean(mean_val: Optional[float], count: int, one_run_placed: bool) -> Optional[float]:
    if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
        return None

    score = float(mean_val) * 10.0
    if int(count) == 6:
        score *= 0.86
    elif int(count) <= 5:
        score *= 0.72

    if one_run_placed:
        score = min(score, 8.0)

    score = max(0.0, min(10.0, score))
    return round(score, 1)

def consistency_score_from_ratio(r: Optional[float], total_runs: int, runs_at_par: int) -> Optional[float]:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return None

    total_runs = int(total_runs or 0)
    runs_at_par = int(runs_at_par or 0)

    if total_runs <= 0:
        return None

    if total_runs == 1 and runs_at_par >= 1:
        return 8.0

    rr = float(r)
    score = round(rr * 10)
    score = max(0, min(10, score))
    return float(score)

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

    # NEW: Betfair prices (optional)
    st.subheader("Box C — Betfair Exchange Prices (optional)")
    st.caption("Paste the Betfair Exchange market block. We extract the first price in the 'Back all' column per horse and save it as MarketOdds.")
    boxC = st.text_area("Box C paste (Betfair)", height=260, key="boxC")

    col_thr = st.columns(3)
    with col_thr[0]:
        front_thr = st.number_input("Front <", value=1.6, step=0.1)
    with col_thr[1]:
        prom_thr  = st.number_input("Prominent <", value=2.4, step=0.1)
    with col_thr[2]:
        mid_thr   = st.number_input("Mid <", value=3.0, step=0.1)

    if st.button("🚀 Process All (A + B + optional C)"):
        try:
            rs_df = parse_run_style(boxA, front_thr, prom_thr, mid_thr)

            b_raw = parse_box45_single(boxB) if boxB.strip() else pd.DataFrame(columns=["Horse"])
            b_df = build_speed_conditions(b_raw) if not b_raw.empty else pd.DataFrame(columns=["Horse"])

            bf_df = parse_betfair_backall(boxC) if boxC.strip() else pd.DataFrame(columns=["Horse","MarketOdds"])

            for ddf in (rs_df, b_df, bf_df):
                if "Horse" in ddf.columns:
                    ddf["Horse"] = ddf["Horse"].astype(str).str.strip()

            merged = rs_df.merge(b_df, on="Horse", how="outer")
            if not bf_df.empty:
                merged = merged.merge(bf_df[["Horse","MarketOdds"]], on="Horse", how="left")

            st.success("Parsed ✓ (Run Style + Speed/Conditions + optional Betfair prices)")

            a, b = st.tabs(["Run Style ✓", "Speed & Conditions (Box B) ✓"])
            with a:
                st.dataframe(rs_df, use_container_width=True)
            with b:
                st.dataframe(b_df, use_container_width=True)

            if not bf_df.empty:
                st.markdown("### Betfair Prices ✓ (parsed)")
                st.dataframe(bf_df, use_container_width=True)

            st.markdown("## 🧩 Combined Output (A + B + optional Betfair)")
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

            df["ΔvsPar"] = pd.to_numeric(df["AdjSpeed"], errors="coerce") - float(class_par)
            df["Style"] = df["RS_Avg"].apply(lambda x: _rs_category_from_value(x, 1.6, 2.4, 3.0))

            s = Settings(class_par=class_par, distance_f=distance_f, wp_even=wp_even, wp_confident=wp_conf)
            df["LCP"] = df.apply(lambda r: _lcp_from_dvp(r["Style"], r["ΔvsPar"], s), axis=1)

            rows = [HorseRow(
                horse=r.Horse,
                style_cat=r.Style,
                adj_speed=float(r.AdjSpeed) if pd.notna(r.AdjSpeed) else None,
                dvp=float(r["ΔvsPar"]) if pd.notna(r["ΔvsPar"]) else None,
                lcp=r.LCP
            ) for _, r in df.iterrows()]
            scenario, conf_numeric, lcp_map, debug = project_pace_from_rows(rows, s)

            if pace_override != "Auto":
                debug["rules_applied"].append(f"Manual override applied: {pace_override}")
                debug["manual_override"] = pace_override
                scenario = pace_override
                conf_display = "Manual"
            else:
                conf_display = f"{conf_numeric:.2f}"

            band = "5f" if distance_f <= 5.5 else ("6f" if distance_f <= 6.5 else "route")

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

            if scenario.startswith("Even (Front-Controlled)") and band == "route":
                pacefit_map = PACEFIT_EVEN_FC
            elif band == "5f":
                pacefit_map = PACEFIT_5F["Even"] if scenario.startswith("Even") else PACEFIT_5F.get(scenario, PACEFIT_5F["Even"])
            elif band == "6f":
                pacefit_map = PACEFIT_6F["Even"] if scenario.startswith("Even") else PACEFIT_6F.get(scenario, PACEFIT_6F["Even"])
            else:
                key = "Even" if scenario.startswith("Even") else scenario
                pacefit_map = PACEFIT.get(key, PACEFIT["Even"])

            wp = s.wp_confident if (scenario in ("Slow","Very Strong") and conf_numeric >= 0.65) else s.wp_even
            if band == "5f": wp = max(wp, 0.60)
            elif band == "6f": wp = max(wp, 0.55)
            ws = 1 - wp

            def _speedfit(v):
                if pd.isna(v): return 2.3
                v = float(v)
                if v >= 2: return 5.0
                if v >= -1: return 4.0
                if v >= -4: return 3.0
                if v >= -8: return 2.0
                return 1.0

            df["SpeedFit"] = df["ΔvsPar"].apply(_speedfit)

            df["PaceFit"] = df["Style"].map(pacefit_map).fillna(3.0)

            df["PaceFit_Slow"]   = df["Style"].map(slow_map).fillna(3.0)
            df["PaceFit_Even"]   = df["Style"].map(even_map).fillna(3.0)
            df["PaceFit_Strong"] = df["Style"].map(strong_map).fillna(3.0)

            df["Suitability_Slow"]   = df["PaceFit_Slow"]   * wp + df["SpeedFit"] * ws
            df["Suitability_Even"]   = df["PaceFit_Even"]   * wp + df["SpeedFit"] * ws
            df["Suitability_Strong"] = df["PaceFit_Strong"] * wp + df["SpeedFit"] * ws

            df["Suitability_Base_Even"]   = df["Suitability_Even"]
            df["Suitability_Base_Strong"] = df["Suitability_Strong"]

            df["Suitability_Base"] = df["PaceFit"] * wp + df["SpeedFit"] * ws
            df["Suitability_Base_Hybrid"] = df["Suitability_Base_Even"]

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

            late_strong_warn = late_strong_warn_hybrid_only_6f(
                distance_f=distance_f,
                use_hybrid=use_hyb,
                confidence=conf_numeric,
                counts=debug.get("counts", {}),
                energy=debug.get("early_energy", 0.0),
            )
            debug["late_strong_warn"] = bool(late_strong_warn)

            debug["late_strong_watch"] = []
            if debug["late_strong_warn"]:
                tmp = df.copy()
                tmp["LateStrongProfile"] = (tmp["Suitability_Slow"] - tmp["Suitability_Strong"])
                tmp = tmp[tmp["Style"].isin(["Front", "Prominent", "Mid"])].copy()
                tmp = tmp[(tmp["Suitability_Slow"] >= 3.6) & (tmp["Suitability_Even"] >= 3.1)].copy()
                tmp = tmp.sort_values(
                    ["LateStrongProfile", "Suitability_Slow", "Suitability_Base"],
                    ascending=False
                )
                watch = tmp[tmp["LateStrongProfile"] >= 0.6].head(2)["Horse"].tolist()
                if not watch and not tmp.empty:
                    watch = tmp.head(1)["Horse"].tolist()
                debug["late_strong_watch"] = watch

            df["wp"], df["ws"] = wp, ws
            df["Confidence"] = conf_display

            df["Suitability"] = df["Suitability_Base"]
            if clip5:
                df["Suitability"] = df["Suitability"].clip(1.0, 5.0)

            # =========================
            # Diagnostics
            # =========================
            cond_cols_present = [c for c in COND_COLS if c in df.columns]
            total_cols_present = [c for c in COND_TOTAL_COLS if c in df.columns]

            if cond_cols_present:
                df["Conditions_Count"] = df[cond_cols_present].notna().sum(axis=1)
                df["Conditions_Mean"] = df[cond_cols_present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

                df["Conditions_OneRunPlaced"] = False
                if total_cols_present:
                    flags = []
                    for _, r in df.iterrows():
                        flag = False
                        for base in ["crs","dist","lhrh","going","cls","runpm1","trackstyle"]:
                            pcol = f"{base}_place"
                            tcol = f"{base}_total"
                            if pcol in df.columns and tcol in df.columns:
                                t = r.get(tcol, None)
                                p = r.get(pcol, None)
                                if pd.notna(t) and int(t) == 1 and pd.notna(p) and float(p) > 0:
                                    flag = True
                                    break
                        flags.append(flag)
                    df["Conditions_OneRunPlaced"] = flags

                df["Conditions_Score"] = [
                    conditions_score_from_mean(m, int(c), bool(one))
                    for m, c, one in zip(df["Conditions_Mean"].tolist(),
                                         df["Conditions_Count"].tolist(),
                                         df["Conditions_OneRunPlaced"].tolist())
                ]
            else:
                df["Conditions_Count"] = 0
                df["Conditions_Mean"] = np.nan
                df["Conditions_OneRunPlaced"] = False
                df["Conditions_Score"] = None

            sr_col = None
            for cand in ["SpeedRunsRaw","speed_series","SpeedRunsList"]:
                if cand in df.columns:
                    sr_col = cand
                    break

            totals, atpar, pct = [], [], []
            if sr_col is not None:
                for srs in df[sr_col].fillna("").astype(str):
                    nums = _numbers_ignore_brackets(srs)
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

            df["SpeedConsistency_Score"] = [
                consistency_score_from_ratio(p, t, a)
                for p, t, a in zip(df["Speed_ParPct"].tolist(),
                                   df["Speed_TotalRuns"].tolist(),
                                   df["Speed_RunsAtPar"].tolist())
            ]

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
                    for r in rules:
                        st.write(f"• {r}")

                if debug.get("late_strong_warn", False):
                    watch = debug.get("late_strong_watch", [])
                    watch_txt = (f" **Watch:** {', '.join(watch)}" if watch else "")
                    st.markdown(
                        "\n⚠️ **Beware Late-Strong (6f):** Hybrid can understate late pressure — finishing speed can decide."
                        + watch_txt
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
            cond_break_cols = ["Horse","Conditions_Count","Conditions_Mean","Conditions_Score"]
            cond_break_cols = [c for c in cond_break_cols if c in df.columns]
            cond_break = df[cond_break_cols].copy()
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
            overlay_cols = ["Horse","Suitability","Suitability_Even","Suitability_Strong","Suitability_Slow","Conditions_Score","SpeedConsistency_Score","MarketOdds"]
            overlay_cols = [c for c in overlay_cols if c in df.columns]
            st.dataframe(df[overlay_cols].sort_values(["Suitability"], ascending=False), use_container_width=True)

            # =========================
            # NEW BOX: Tissue v2 (pricing layer only; does not affect backbone/diagnostics)
            # =========================
            st.markdown("## 💷 Tissue Prices v2 (Softmax + Evidence Control)")

            with st.expander("Tissue v2 settings (does not change backbone)", expanded=True):
                tc = st.columns(6)
                w_suit = tc[0].slider("Weight: Suitability", 0.0, 1.0, 0.50, 0.05)
                w_cond = tc[1].slider("Weight: Conditions", 0.0, 1.0, 0.20, 0.05)
                w_spc  = tc[2].slider("Weight: SpeedConsistency", 0.0, 1.0, 0.30, 0.05)
                temp   = tc[3].slider("Softmax temperature (lower = stronger favs)", 0.05, 0.50, 0.18, 0.01)
                k_shrk = tc[4].slider("Evidence shrink strength (k)", 1.0, 12.0, 6.0, 0.5)
                prior_mode = tc[5].selectbox("Missing-data prior", ["Neutral (0.50)", "Race mean"], index=1)

                oc = st.columns(4)
                use_overround = oc[0].checkbox("Apply overround (book %)", value=False)
                target_or = oc[1].slider("Target overround", 1.00, 1.30, 1.08, 0.01)
                or_shape  = oc[2].slider("Fav/longshot shape", 0.00, 0.60, 0.15, 0.05)
                show_value = oc[3].checkbox("Show value vs Market Odds (if column exists)", value=True)

            def _clip01(x: float) -> float:
                return float(np.clip(x, 0.0, 1.0))

            def _to01_0to10(x):
                if pd.isna(x):
                    return np.nan
                return _clip01(float(x) / 10.0)

            def _to01_suit_1to5(x):
                if pd.isna(x):
                    return np.nan
                return _clip01((float(x) - 1.0) / 4.0)

            def _shrink(x01, n, prior01=0.5, k=6.0):
                if pd.isna(x01):
                    x01 = prior01
                if pd.isna(n):
                    n = 0.0
                n = max(0.0, float(n))
                w = n / (n + float(k)) if (n + float(k)) > 0 else 0.0
                return _clip01(w * float(x01) + (1.0 - w) * float(prior01))

            def _softmax(series: pd.Series, temperature: float) -> pd.Series:
                t = max(1e-6, float(temperature))
                a = series.astype(float).values
                a = a - np.nanmax(a)
                ex = np.exp(a / t)
                ex = np.where(np.isfinite(ex), ex, 0.0)
                ssum = ex.sum()
                if ssum <= 0:
                    return pd.Series(np.ones(len(series)) / max(1, len(series)), index=series.index)
                return pd.Series(ex / ssum, index=series.index)

            def _apply_overround(p_fair: pd.Series, overround=1.00, shape=0.15) -> pd.Series:
                p = p_fair.clip(lower=1e-12).astype(float)
                sh = float(np.clip(shape, 0.0, 0.60))
                p2 = p ** (1.0 - sh)
                p2 = p2 / p2.sum()
                return p2 * float(overround)

            def _odds(p: pd.Series) -> pd.Series:
                return (1.0 / p.replace(0, np.nan)).astype(float)

            def _value_edge(fair_odds: pd.Series, mkt_odds: pd.Series) -> pd.Series:
                return (mkt_odds / fair_odds) - 1.0

            tissue = df.copy()

            tissue["T_Suit01"] = tissue["Suitability"].apply(_to01_suit_1to5)
            tissue["T_Cond01_raw"] = tissue["Conditions_Score"].apply(_to01_0to10) if "Conditions_Score" in tissue.columns else np.nan
            tissue["T_Spc01_raw"]  = tissue["SpeedConsistency_Score"].apply(_to01_0to10) if "SpeedConsistency_Score" in tissue.columns else np.nan

            cond_n = pd.to_numeric(tissue.get("Conditions_Count", 0), errors="coerce").fillna(0)
            spc_n  = pd.to_numeric(tissue.get("Speed_TotalRuns", 0), errors="coerce").fillna(0)

            if prior_mode == "Race mean":
                prior_s = float(np.nanmean(tissue["T_Suit01"])) if np.isfinite(np.nanmean(tissue["T_Suit01"])) else 0.5
                prior_c = float(np.nanmean(tissue["T_Cond01_raw"])) if np.isfinite(np.nanmean(tissue["T_Cond01_raw"])) else 0.5
                prior_k = float(np.nanmean(tissue["T_Spc01_raw"])) if np.isfinite(np.nanmean(tissue["T_Spc01_raw"])) else 0.5
            else:
                prior_s = prior_c = prior_k = 0.5

            tissue["T_Cond01"] = [
                _shrink(x, n, prior01=prior_c, k=k_shrk)
                for x, n in zip(tissue["T_Cond01_raw"].tolist(), cond_n.tolist())
            ]
            tissue["T_Spc01"] = [
                _shrink(x, n, prior01=prior_k, k=k_shrk)
                for x, n in zip(tissue["T_Spc01_raw"].tolist(), spc_n.tolist())
            ]

            rC = cond_n / (cond_n + float(k_shrk))
            rK = spc_n  / (spc_n  + float(k_shrk))

            wS = float(w_suit)
            wC = float(w_cond) * rC
            wK = float(w_spc)  * rK
            wSum = (wS + wC + wK).replace(0, np.nan)

            tissue["T_wS"] = wS / wSum
            tissue["T_wC"] = wC / wSum
            tissue["T_wK"] = wK / wSum

            tissue["T_Ability"] = (
                tissue["T_wS"] * tissue["T_Suit01"].fillna(prior_s)
                + tissue["T_wC"] * tissue["T_Cond01"].fillna(prior_c)
                + tissue["T_wK"] * tissue["T_Spc01"].fillna(prior_k)
            )

            tissue["T_Prob_Fair"] = _softmax(tissue["T_Ability"], temperature=temp)
            tissue["T_Odds_Fair"] = _odds(tissue["T_Prob_Fair"])

            if use_overround:
                tissue["T_Prob_Book"] = _apply_overround(tissue["T_Prob_Fair"], overround=target_or, shape=or_shape)
            else:
                tissue["T_Prob_Book"] = tissue["T_Prob_Fair"]

            tissue["T_Odds_Book"] = _odds(tissue["T_Prob_Book"])

            mkt_col = None
            if show_value:
                for cand in ["MarketOdds","Market Odds","Odds","BSP","SP","EarlyPrice","Price"]:
                    if cand in tissue.columns:
                        mkt_col = cand
                        break

            if show_value and mkt_col is not None:
                tissue["MarketOdds"] = pd.to_numeric(tissue[mkt_col], errors="coerce")
                tissue["Edge_Back"] = _value_edge(tissue["T_Odds_Fair"], tissue["MarketOdds"])
            else:
                tissue["MarketOdds"] = np.nan
                tissue["Edge_Back"] = np.nan

            tissue = tissue.sort_values(["T_Prob_Fair","T_Ability"], ascending=False).copy()
            tissue["T_Rank"] = range(1, len(tissue)+1)

            BACK_EDGE = 0.08
            LAY_EDGE  = -0.10
            tissue["Bet_Back"] = (tissue["T_Rank"] <= 3) & (tissue["Edge_Back"] >= BACK_EDGE)
            tissue["Bet_Lay"]  = (tissue["T_Rank"] >= 4) & (tissue["Edge_Back"] <= LAY_EDGE)

            out_cols = [
                "T_Rank","Horse",
                "Suitability","Conditions_Score","SpeedConsistency_Score",
                "Conditions_Count","Speed_TotalRuns",
                "T_wS","T_wC","T_wK",
                "T_Ability",
                "T_Prob_Fair","T_Odds_Fair",
                "T_Prob_Book","T_Odds_Book",
                "MarketOdds","Edge_Back",
                "Bet_Back","Bet_Lay"
            ]
            out_cols = [c for c in out_cols if c in tissue.columns]
            tissue_out = tissue[out_cols].copy()

            for ccol in ["T_wS","T_wC","T_wK","T_Prob_Fair","T_Prob_Book"]:
                if ccol in tissue_out.columns:
                    tissue_out[ccol] = tissue_out[ccol].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
            for ccol in ["T_Odds_Fair","T_Odds_Book","MarketOdds"]:
                if ccol in tissue_out.columns:
                    tissue_out[ccol] = tissue_out[ccol].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
            if "Edge_Back" in tissue_out.columns:
                tissue_out["Edge_Back"] = tissue_out["Edge_Back"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "")

            st.dataframe(tissue_out, use_container_width=True)

            # =========================
            # NEW BOX: Bet Guidance (Execution Layer) — DOES NOT CHANGE ANY MODEL OUTPUTS
            # =========================
            st.markdown("## 🧭 Bet Guidance (20/80 Win–Place) — Execution Layer (No model changes)")

            with st.expander("Bet guidance settings (does not change backbone/tissue)", expanded=True):
                gc = st.columns(6)
                min_rank_for_consider = gc[0].number_input("Only consider top-N tissue ranks", min_value=1, max_value=20, value=3, step=1)
                min_cond_wp = gc[1].slider("Min Conditions for Win–Place", 0.0, 10.0, 4.0, 0.5)
                min_spc_wp  = gc[2].slider("Min SpeedConsistency for Win–Place", 0.0, 10.0, 4.0, 0.5)
                min_edge_wp = gc[3].slider("Min value edge for Win–Place", 0.00, 0.50, 0.10, 0.01)
                min_edge_win = gc[4].slider("Min value edge for Win Only", 0.00, 0.50, 0.08, 0.01)
                show_table_all = gc[5].checkbox("Show guidance for entire field", value=True)

                rc = st.columns(4)
                block_wp_in_hybrid = rc[0].checkbox("Block Win–Place in Hybrid races", value=True)
                block_wp_5f = rc[1].checkbox("Block Win–Place at 5f", value=True)
                allow_wp_strong_route = rc[2].checkbox("Allow Win–Place in Strong (≥7f) for non-pace-exposed", value=True)
                pace_exposed_front_blocks = rc[3].checkbox("Treat Front/leader-likely as pace-exposed", value=True)

            scenario_label = str(df["Scenario"].iloc[0]) if "Scenario" in df.columns else str(scenario)
            band_label = str(band)
            counts = debug.get("counts", {}) if isinstance(debug, dict) else {}
            FH = int(counts.get("Front_High", 0))
            PH = int(counts.get("Prominent_High", 0))

            is_hybrid = bool(use_hyb) or scenario_label.startswith("Hybrid")
            is_5f = (band_label == "5f")
            is_slow = scenario_label.startswith("Slow")
            is_even = scenario_label.startswith("Even")
            is_strong = scenario_label.startswith("Strong")

            tactical_sensitive = (FH >= 2) or (FH >= 1 and PH >= 3) or (FH + PH >= 4)

            wp_allowed_by_pace = False
            if is_slow or is_even:
                wp_allowed_by_pace = True
            elif is_strong and allow_wp_strong_route and (float(distance_f) >= 7.0):
                wp_allowed_by_pace = True
            else:
                wp_allowed_by_pace = False

            if block_wp_in_hybrid and is_hybrid:
                wp_allowed_by_pace = False
            if block_wp_5f and is_5f:
                wp_allowed_by_pace = False
            if tactical_sensitive and not is_slow:
                wp_allowed_by_pace = False

            def _pace_exposed(style: str, lcp: str) -> bool:
                if not pace_exposed_front_blocks:
                    return False
                style = str(style or "")
                lcp = str(lcp or "")
                if style == "Front":
                    return True
                if style == "Prominent" and lcp in ("High", "Questionable") and (FH + PH) >= 3:
                    return True
                return False

            guide = tissue.copy()
            if "Edge_Back" not in guide.columns:
                guide["Edge_Back"] = np.nan

            guide["G_PaceExposed"] = [
                _pace_exposed(sty, lcp)
                for sty, lcp in zip(guide.get("Style", pd.Series([""]*len(guide))).tolist(),
                                    guide.get("LCP", pd.Series([""]*len(guide))).tolist())
            ]

            guide["G_DiagOK_WP"] = (
                (pd.to_numeric(guide.get("Conditions_Score", np.nan), errors="coerce") >= float(min_cond_wp))
                & (pd.to_numeric(guide.get("SpeedConsistency_Score", np.nan), errors="coerce") >= float(min_spc_wp))
            )

            guide["G_ValueOK_WP"] = (pd.to_numeric(guide["Edge_Back"], errors="coerce") >= float(min_edge_wp))
            guide["G_ValueOK_WIN"] = (pd.to_numeric(guide["Edge_Back"], errors="coerce") >= float(min_edge_win))

            guide["G_TopN"] = (pd.to_numeric(guide.get("T_Rank", np.nan), errors="coerce") <= int(min_rank_for_consider))

            has_any_market = pd.to_numeric(guide.get("MarketOdds", np.nan), errors="coerce").notna().any()

            def _bet_guidance_row(r) -> str:
                topn = bool(r.get("G_TopN", False))
                if not topn:
                    return "🔴 No Bet"

                edge_wp = bool(r.get("G_ValueOK_WP", False))
                edge_win = bool(r.get("G_ValueOK_WIN", False))
                diag_ok = bool(r.get("G_DiagOK_WP", False))
                pace_ex = bool(r.get("G_PaceExposed", False))

                # If we don't have market prices, provide conditional guidance
                if not has_any_market:
                    if wp_allowed_by_pace and diag_ok and (not pace_ex):
                        return "🟢 Win–Place OK (IF value)"
                    return "🟡 Win Only (IF value)"

                if wp_allowed_by_pace and diag_ok and (not pace_ex) and edge_wp:
                    return "🟢 Win–Place OK (20/80)"
                if edge_win:
                    return "🟡 Win Only"
                return "🔴 No Bet"

            guide["Bet_Guidance"] = guide.apply(_bet_guidance_row, axis=1)

            # --- Race-level headline (FIXED & SAFE)
            topN_view = guide[guide["G_TopN"]].copy()

            if not topN_view.empty:
                if (topN_view["Bet_Guidance"].astype(str).str.startswith("🟢")).any():
                    headline = "✅ Win–Place (20/80) is allowed on qualifying value selections." if has_any_market else "🟢 Win–Place structurally allowed (apply ONLY if market value exists)."
                    level = "success"
                elif (topN_view["Bet_Guidance"].astype(str).str.startswith("🟡")).any():
                    headline = "⚠️ Win–Place NOT advised. If betting, keep to WIN ONLY on value." if has_any_market else "🟡 Win Only structurally preferred (requires market value)."
                    level = "warning"
                else:
                    headline = "⛔ No bet suggested (top-N lacks value/robustness)."
                    level = "info"
            else:
                headline = "⛔ No bet suggested (no top-N candidates)."
                level = "info"

            if level == "success":
                st.success(headline)
            elif level == "warning":
                st.warning(headline)
            else:
                st.info(headline)

            reasons = []
            reasons.append(f"**Scenario:** {scenario_label}  |  **Band:** {band_label}  |  **Distance:** {distance_f}f")
            reasons.append(f"**Credible pace (High LCP):** Front {FH}, Prominent {PH}")
            reasons.append(f"**Hybrid active:** {is_hybrid}  |  **Tactical sensitivity:** {tactical_sensitive}")
            reasons.append(f"**Win–Place allowed by pace filter:** {wp_allowed_by_pace}")
            reasons.append(f"**Market odds provided:** {has_any_market}")
            st.markdown("- " + "\n- ".join(reasons))

            show_cols = [
                "T_Rank","Horse","Style","LCP",
                "Suitability","Conditions_Score","SpeedConsistency_Score",
                "MarketOdds","Edge_Back",
                "Bet_Guidance"
            ]
            show_cols = [c for c in show_cols if c in guide.columns]
            guide_view = guide[show_cols].copy().sort_values("T_Rank")

            if "Edge_Back" in guide_view.columns:
                guide_view["Edge_Back"] = pd.to_numeric(guide_view["Edge_Back"], errors="coerce").apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
                )
            if "MarketOdds" in guide_view.columns:
                guide_view["MarketOdds"] = pd.to_numeric(guide_view["MarketOdds"], errors="coerce").apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )

            topN_only = guide_view[guide_view["T_Rank"] <= int(min_rank_for_consider)].copy()
            st.markdown(f"### Top {int(min_rank_for_consider)} candidates — bet type guidance")
            st.dataframe(topN_only, use_container_width=True)

            if show_table_all:
                st.markdown("### Full field — bet type guidance")
                st.dataframe(guide_view, use_container_width=True)

            st.download_button(
                "💾 Download Backbone + Diagnostics CSV",
                df.to_csv(index=False),
                "cleanpace_v3_backbone_plus_diagnostics.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
