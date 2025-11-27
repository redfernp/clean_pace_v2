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
st.set_page_config(page_title="CleanPace v2 — All-in-One", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v2 — All-in-One Normaliser + Suitability")

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

def _to_money(val: str) -> Optional[float]:
    if val is None: return None
    s = str(val).strip().replace(",", "")
    m = re.search(r"£\s*([0-9]*\.?[0-9]+)\s*([KkMm])?", s)
    if not m: return None
    num = float(m.group(1)); suf = (m.group(2) or "").upper()
    if suf == "K": num *= 1000
    if suf == "M": num *= 1_000_000
    return num

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
# Split Box A paste into 3 sections
# =========================
def split_three_sections(raw: str) -> Tuple[str, str, str]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    def find_idx(pat: str) -> int:
        m = re.search(pat, text, flags=re.I)
        return m.start() if m else -1

    i_rs = find_idx(r"\bRun\s*Style\s*Figure\b")
    i_or = find_idx(r"\bOfficial\s*Ratings\b")
    i_rc = find_idx(r"\bRace\s*Class\s*-\s*Today\b")
    if min(i_rs, i_or, i_rc) == -1:
        raise ValueError("Anchors not found. Need 'Run Style Figure', 'Official Ratings', 'Race Class - Today: ...'.")
    if not (i_rs < i_or < i_rc):
        raise ValueError("Anchors out of order. Expected: Run Style → Official Ratings → Race Class.")

    return text[i_rs:i_or].strip(), text[i_or:i_rc].strip(), text[i_rc:].strip()

# =========================
# Parse Run Style section
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
# Parse Official Ratings section
# =========================
def _parse_or_line(s: str) -> Optional[int]:
    s = str(s).strip()
    if not s: return None
    left = s.split("/")[0].strip()
    m = re.search(r"-?\d+", left)
    return int(m.group(0)) if m else None

def parse_official_ratings(or_text: str) -> pd.DataFrame:
    lines = [l for l in or_text.splitlines() if l.strip()]
    if lines and lines[0].strip().lower().startswith("official ratings"):
        lines = lines[1:]

    rows = []; i = 0
    while i < len(lines):
        parts = lines[i].split("\t")
        if i == 0 and parts and parts[0].strip().lower().startswith("horse"):
            i += 1; continue
        if len(parts) >= 5:
            horse = parts[0].strip()
            today_str = parts[2].strip()
            highest_header_str = parts[4].strip()
            i += 1
            ratings: List[Optional[int]] = []
            j = i
            while j < len(lines):
                nxt = lines[j].split("\t")
                if len(nxt) >= 5: break
                ratings.append(_parse_or_line(lines[j])); j += 1
            i = j

            today_or = int(re.search(r"-?\d+", today_str).group(0)) if re.search(r"-?\d+", today_str) else None
            valid = [x for x in ratings if x is not None]
            max_or_10 = max(valid) if valid else None
            or_3back  = valid[2] if len(valid) >= 3 else None

            re_lb = (max_or_10 - today_or) if (max_or_10 is not None and today_or is not None) else None
            d_or_trend = (today_or - or_3back) if (today_or is not None and or_3back is not None) else None

            ts_5 = None
            if re_lb is not None:
                ts_5 = 5 if re_lb >= 8 else 4 if re_lb >= 4 else 3 if re_lb >= 0 else 2 if re_lb >= -4 else 1
                if d_or_trend is not None:
                    if d_or_trend >= 2:   ts_5 = min(5, ts_5 + 0.5)
                    elif d_or_trend <= -2: ts_5 = max(1, ts_5 - 0.5)

            hwin_or = None
            mH = re.search(r"-?\d+", highest_header_str)
            if mH: hwin_or = int(mH.group(0))

            handicap_runs = sum(1 for r in valid if (r is not None and r > 0))

            delta_hwin_code = (hwin_or - today_or) if (hwin_or is not None and today_or is not None) else None
            tau = 6 if handicap_runs < 3 else 4 if handicap_runs <= 5 else 2 if handicap_runs <= 9 else 0
            if (hwin_or is not None) and (today_or is not None) and (today_or > hwin_or):
                delta_hwin_adj = (hwin_or + tau) - today_or
            else:
                delta_hwin_adj = delta_hwin_code

            def _bucket(x: Optional[float]) -> Optional[int]:
                if x is None: return None
                if x >= 8:  return 5
                if x >= 4:  return 4
                if x >= 0:  return 3
                if x >= -4: return 2
                return 1

            rating_fit_code_5     = _bucket(delta_hwin_code)
            rating_fit_code_adj_5 = _bucket(delta_hwin_adj)

            ts_5_adj = ts_5
            if (handicap_runs < 6) and (today_or is not None) and (max_or_10 is not None) and (today_or > max_or_10):
                ts_5_adj = min(5, (ts_5 or 0) + 0.5)

            rci = None
            if (ts_5_adj is not None) and (rating_fit_code_adj_5 is not None):
                rci = round(0.7 * ts_5_adj + 0.3 * rating_fit_code_adj_5, 2)

            rows.append({
                "Horse": horse,
                "today_or": today_or,
                "or_3back": or_3back,
                "max_or_10": max_or_10,
                "re_lb": re_lb,
                "d_or_trend": d_or_trend,
                "ts_5": ts_5,
                "handicap_runs": handicap_runs,
                "hwin_or": hwin_or,
                "delta_hwin_code": delta_hwin_code,
                "delta_hwin_adj": delta_hwin_adj,
                "rating_fit_code_5": rating_fit_code_5,
                "rating_fit_code_adj_5": rating_fit_code_adj_5,
                "ts_5_adj": ts_5_adj,
                "rating_context_index": rci
            })
        else:
            i += 1

    return pd.DataFrame(rows)

# =========================
# Parse Race Class section
# =========================
def parse_race_class(rc_text: str) -> Tuple[pd.DataFrame, Optional[float]]:
    lines = rc_text.splitlines()
    today_val = None
    if lines and lines[0].strip().lower().startswith("race class - today"):
        m = re.search(r"£\s*[0-9]*\.?[0-9]+\s*[KkMm]?", lines[0])
        if m: today_val = _to_money(m.group(0))
        rc_text_body = "\n".join(lines[1:])
    else:
        rc_text_body = rc_text

    def _try_tabular(text: str) -> Optional[pd.DataFrame]:
        try:
            df = _read_table_guess(text)
        except Exception:
            return None
        if "Horse" not in df.columns:
            return None
        df["Horse"] = df["Horse"].astype(str).str.strip()
        df["avg3_prize"] = df.get("Avg 3", np.nan).apply(_to_money) if "Avg 3" in df.columns else np.nan
        df["avg5_prize"] = df.get("Avg 5", np.nan).apply(_to_money) if "Avg 5" in df.columns else np.nan
        return df[["Horse","avg3_prize","avg5_prize"]]

    df_tab = _try_tabular(rc_text_body)
    if df_tab is not None and (df_tab["avg3_prize"].notna().any() or df_tab["avg5_prize"].notna().any()):
        df_out = df_tab.copy()
        df_out["today_class_value"] = today_val
        def ccs(cd):
            if pd.isna(cd): return None
            if cd >= 0.40: return 5
            if cd >= 0.15: return 4
            if cd >= -0.15: return 3
            if cd >= -0.40: return 2
            return 1
        df_out["cd"] = (df_out["avg3_prize"] / df_out["today_class_value"]) - 1 if today_val else np.nan
        df_out["ccs_5"] = df_out["cd"].apply(ccs)
        return df_out[["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]], today_val

    # Fallback (block)
    body_lines = [ln for ln in rc_text_body.splitlines() if ln.strip()]
    if body_lines and body_lines[0].strip().lower().startswith("horse"):
        body_lines = body_lines[1:]

    def is_horse_line(s: str) -> bool:
        t = s.strip()
        if not t: return False
        if t.lower().startswith("cl"): return False
        if t.startswith("£"): return False
        if "avg" in t.lower(): return False
        return True

    horses = []; i = 0; n = len(body_lines)
    while i < n:
        line = body_lines[i].strip()
        if is_horse_line(line):
            horse = line; i += 1
            block = []
            while i < n and not is_horse_line(body_lines[i]):
                block.append(body_lines[i].strip()); i += 1
            money_vals = [_to_money(x) for x in block if "£" in x]
            avg3, avg5 = (money_vals[-2], money_vals[-1]) if len(money_vals) >= 2 else (None, None)
            horses.append({"Horse": horse, "avg3_prize": avg3, "avg5_prize": avg5})
        else:
            i += 1

    df_blk = pd.DataFrame(horses)
    if df_blk.empty:
        return pd.DataFrame(columns=["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]), today_val

    df_blk["today_class_value"] = today_val
    def ccs(cd):
        if pd.isna(cd): return None
        if cd >= 0.40: return 5
        if cd >= 0.15: return 4
        if cd >= -0.15: return 3
        if cd >= -0.40: return 2
        return 1
    df_blk["cd"] = (df_blk["avg3_prize"] / df_blk["today_class_value"]) - 1 if today_val else np.nan
    df_blk["ccs_5"] = df_blk["cd"].apply(ccs)
    return df_blk[["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]], today_val

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
    This prevents lines like '1, B, 2, 2, 2, 8 (4)' being tagged as names.
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
        # seek next real horse name (robust)
        while i < n and not _looks_like_name(lines, i):
            i += 1
        if i >= n: break
        name = lines[i].strip(); i += 1

        # read up to 10 metric lines until next real name
        values: List[str] = []
        while i < n and len(values) < 10:
            if _looks_like_name(lines, i):  # next horse found
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

    last_list, high_list, avg3_list, all_list, keyavg_list = [], [], [], [], []
    for s in out["speed_series"].fillna(""):
        last, high, avg3, avgall = _parse_speed_series(s)
        last_list.append(last); high_list.append(high)
        avg3_list.append(avg3);  all_list.append(avgall)
        keyavg_list.append(None if (last is None or high is None or avg3 is None) else round((last+high+avg3)/3, 1))

    out["LastRace"] = last_list
    out["Highest"]  = high_list
    out["Avg3"]     = avg3_list
    out["AvgAll"]   = all_list
    out["KeySpeedAvg"] = keyavg_list

    for col in ["crs","dist","lhrh","going","cls","runpm1","trackstyle"]:
        if col in out.columns:
            out[f"{col}_place"] = [ _parse_wpt_value(v)[0] for v in out[col].fillna("") ]

    keep = ["Horse","LastRace","Highest","Avg3","AvgAll","KeySpeedAvg",
            "crs_place","dist_place","lhrh_place","going_place","cls_place","runpm1_place","trackstyle_place"]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

# =========================
# Pace engine + Suitability (from RS_Avg classification)
# =========================
@dataclass
class HorseRow:
    horse: str
    style_cat: Optional[str]          # Front/Prominent/Mid/Hold-up (from RS_Avg)
    adj_speed: Optional[float]        # AdjSpeed (KeySpeedAvg)
    dvp: Optional[float]              # Δ vs Par (computed in tab)
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
    if style_cat not in ("Front","Prominent") or dvp is None:
        return "N/A"
    if dvp >= s.lcp_high: return "High"
    if dvp >= s.lcp_question_low: return "Questionable"
    return "Unlikely"

def _speed_score(dvp: Optional[float]) -> float:
    if dvp is None: return 2.3
    if dvp >= 2:    return 5.0
    if dvp >= -1:   return 4.0
    if dvp >= -4:   return 3.0
    if dvp >= -8:   return 2.0
    return 1.0

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

    # A) Distance gate
    if distance_f > 8.0:
        return False

    # B) Scenario gate (borderline Even/Strong only)
    if scenario.startswith("Slow") or scenario.startswith("Very Strong"):
        return False
    if not (scenario.startswith("Even") or scenario == "Strong"):
        return False

    # C) Confidence gate (borderline zone)
    if confidence < 0.45 or confidence > 0.65:
        return False

    # D) Leader structure gate
    if total_high <= 1:
        return False
    if FH >= 3:
        return False

    # Best Hybrid zone: 1–2 High front, plus at least 2 early-high combined
    if FH in (1, 2) and total_high >= 2:
        return True

    return False

# =========================
# Pace projection engine
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

    # --- 3-LCP guarantee rule ---
    if n_high_early >= 3:
        scenario, conf = "Strong", 0.80
        locked_strong = True
        debug["rules_applied"].append("≥3 High-LCP early (Front/Prominent) → Strong (locked)")
    else:
        # Original decision tree
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

        # Caps and adjustments only if not locked
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

        if n_fh == 2 and scenario in ("Strong","Very Strong"):
            debug["rules_applied"].append("Two High Fronts (kept)")

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
# Tabs UI
# =========================
TAB_MAIN, TAB_PACE = st.tabs(["All Inputs (2 boxes)", "Pace & Suitability (from Combined CSV)"])

# ---------- TAB 1: All Inputs (2 boxes) ----------
with TAB_MAIN:
    st.subheader("Box A — RS/OR/Class (single paste)")
    st.caption("Paste: Run Style Figure → Official Ratings → Race Class - Today: Class N £X")
    boxA = st.text_area("Box A paste", height=300, key="boxA")

    st.subheader("Box B — Boxes 4 & 5 (single paste)")
    st.caption("Paste the single block that starts with: 'Horse Win % Form Figures (Avg) ...'")
    boxB = st.text_area("Box B paste", height=300, key="boxB")

    col_thr = st.columns(3)
    with col_thr[0]: front_thr = st.number_input("Front <", value=1.6, step=0.1)
    with col_thr[1]: prom_thr  = st.number_input("Prominent <", value=2.4, step=0.1)
    with col_thr[2]: mid_thr   = st.number_input("Mid <", value=3.0, step=0.1)

    if st.button("🚀 Process All (A + B)"):
        try:
            rs_txt, or_txt, rc_txt = split_three_sections(boxA)
            rs_df = parse_run_style(rs_txt, front_thr, prom_thr, mid_thr)
            or_df = parse_official_ratings(or_txt)
            rc_df, today_cls_val = parse_race_class(rc_txt)

            b_raw = parse_box45_single(boxB) if boxB.strip() else pd.DataFrame(columns=["Horse"])
            b_df = build_speed_conditions(b_raw) if not b_raw.empty else pd.DataFrame(columns=["Horse"])

            for d in (rs_df, or_df, rc_df, b_df):
                if "Horse" in d.columns: d["Horse"] = d["Horse"].astype(str).str.strip()

            merged = rs_df.merge(or_df, on="Horse", how="outer") \
                          .merge(rc_df, on="Horse", how="outer") \
                          .merge(b_df, on="Horse", how="outer")

            st.success("Parsed all sections ✓")
            a, b, c, dtab = st.tabs(["Run Style ✓","Official Ratings ✓","Race Class ✓","Speed & Conditions ✓"])
            with a: st.dataframe(rs_df, use_container_width=True)
            with b: st.dataframe(or_df, use_container_width=True)
            with c:
                st.write(f"Today Class £: **{today_cls_val if today_cls_val else 'n/a'}**")
                st.dataframe(rc_df, use_container_width=True)
            with dtab: st.dataframe(b_df, use_container_width=True)

            st.markdown("## 🧩 Combined Output (A + B)")
            st.dataframe(merged, use_container_width=True)
            st.download_button("💾 Download Combined CSV",
                               merged.to_csv(index=False),
                               "cleanpace_allinone_combined.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"Failed: {e}")

# ---------- TAB 2: Pace & Suitability ----------
with TAB_PACE:
    st.subheader("Upload 🧩 Combined Output (A + B) CSV")
    upl = st.file_uploader("Combined CSV", type=["csv"], key="pace_csv")

    c = st.columns(5)
    class_par = c[0].number_input("Class Par", value=77.0, step=0.5)
    distance_f = c[1].number_input("Distance (f)", value=6.0, step=0.5, min_value=5.0)
    wp_even    = c[2].slider("wp (Even/Uncertain)", 0.3, 0.8, 0.50, 0.05)
    wp_conf    = c[3].slider("wp (Predictable Slow/Very Strong)", 0.3, 0.8, 0.65, 0.05)
    clip5      = c[4].checkbox("Clip Suitability to 1–5", value=True)

    c2 = st.columns(3)
    w_or   = c2[0].slider("OR context weight", 0.0, 0.6, 0.20, 0.05)
    w_cls  = c2[1].slider("Class compat (ccs_5) weight", 0.0, 0.4, 0.10, 0.05)
    w_cond = c2[2].slider("Conditions (place rates) weight", 0.0, 0.4, 0.10, 0.05)

    # --- Manual pace override switch ---
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

            rsavg_aliases = ["RS_Avg","RS Avg","RSAVG","RS_Avg5","RS_Avg (5)","RS5_Avg","RS(5)Avg"]
            col_rsavg = map_column(df, rsavg_aliases)
            if col_rsavg is None:
                st.error("Missing RS_Avg (avg of last 5 run styles). Expected one of: " + ", ".join(rsavg_aliases))
                st.stop()
            if col_rsavg != "RS_Avg":
                df["RS_Avg"] = pd.to_numeric(df[col_rsavg], errors="coerce")

            rs10_aliases = ["RS_Avg10","RS Avg 10","RSAVG10","RS(10)Avg","RS10_Avg"]
            col_rs10 = map_column(df, rs10_aliases)
            if col_rs10 and col_rs10 != "RS_Avg10":
                df["RS_Avg10"] = pd.to_numeric(df[col_rs10], errors="coerce")

            # Δ vs Par
            df["ΔvsPar"] = pd.to_numeric(df["AdjSpeed"], errors="coerce") - float(class_par)

            # Style category strictly from RS_Avg
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

            # PaceFit maps
            band = "5f" if distance_f <= 5.5 else ("6f" if distance_f <= 6.5 else "route")

            # Even & Strong maps (for Hybrid & display)
            if band == "5f":
                even_map = PACEFIT_5F["Even"]
                strong_map = PACEFIT_5F["Strong"]
            elif band == "6f":
                even_map = PACEFIT_6F["Even"]
                strong_map = PACEFIT_6F["Strong"]
            else:
                # Route distances – use standard Even & Strong, with special Even (FC) if tagged
                if scenario.startswith("Even (Front-Controlled)"):
                    even_map = PACEFIT_EVEN_FC
                else:
                    even_map = PACEFIT["Even"]
                strong_map = PACEFIT["Strong"]

            # Final scenario-specific map (for non-hybrid use)
            if scenario.startswith("Even (Front-Controlled)") and band == "route":
                pacefit_map = PACEFIT_EVEN_FC
            elif band == "5f":
                pacefit_map = PACEFIT_5F["Even"] if scenario.startswith("Even") else PACEFIT_5F.get(scenario, PACEFIT_5F["Even"])
            elif band == "6f":
                pacefit_map = PACEFIT_6F["Even"] if scenario.startswith("Even") else PACEFIT_6F.get(scenario, PACEFIT_6F["Even"])
            else:
                key = "Even" if scenario.startswith("Even") else scenario
                pacefit_map = PACEFIT.get(key, PACEFIT["Even"])

            # Pace/Speed weights (use numeric confidence internally)
            wp = s.wp_confident if (scenario in ("Slow","Very Strong") and conf_numeric >= 0.65) else s.wp_even
            if band == "5f": wp = max(wp, 0.60)
            elif band == "6f": wp = max(wp, 0.55)
            ws = 1 - wp

            # Base PaceFit & SpeedFit
            df["PaceFit_Even"]  = df["Style"].map(even_map).fillna(3.0)
            df["PaceFit_Strong"] = df["Style"].map(strong_map).fillna(3.0)
            df["PaceFit"]       = df["Style"].map(pacefit_map).fillna(3.0)

            def _speedfit(v):
                if pd.isna(v): return 2.3
                v = float(v)
                if v >= 2: return 5.0
                if v >= -1: return 4.0
                if v >= -4: return 3.0
                if v >= -8: return 2.0
                return 1.0
            df["SpeedFit"] = df["ΔvsPar"].apply(_speedfit)

            # Even & Strong suitability bases
            df["Suitability_Base_Even"]   = df["PaceFit_Even"]   * wp + df["SpeedFit"] * ws
            df["Suitability_Base_Strong"] = df["PaceFit_Strong"] * wp + df["SpeedFit"] * ws

            # Start with scenario-based base (non-hybrid)
            df["Suitability_Base"] = df["PaceFit"] * wp + df["SpeedFit"] * ws
            df["Suitability_Base_Hybrid"] = df["Suitability_Base_Even"]  # default

            # HYBRID PACE — automatic when conditions met, and ONLY if no manual override
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
                df["Scenario"] = scenario  # keep original or override

            df["wp"], df["ws"] = wp, ws
            df["Confidence"] = conf_display

            # OR context bonus
            if "rating_context_index" in df.columns:
                rci = pd.to_numeric(df["rating_context_index"], errors="coerce")
            elif {"today_or","max_or_10"}.issubset(df.columns):
                proxy = (pd.to_numeric(df["max_or_10"], errors="coerce") - pd.to_numeric(df["today_or"], errors="coerce")).clip(-12, 12) / 12.0 * 5
                rci = proxy.fillna(3.0)
            else:
                rci = pd.Series([3.0]*len(df))
            df["OR_Bonus"] = (rci - 3.0) / 2.0 * w_or

            # Progressive tolerance
            if {"handicap_runs","today_or","max_or_10"}.issubset(df.columns):
                mask_prog = (pd.to_numeric(df["handicap_runs"], errors="coerce").fillna(99) < 6) & \
                            (pd.to_numeric(df["today_or"], errors="coerce") > pd.to_numeric(df["max_or_10"], errors="coerce"))
                df.loc[mask_prog, "OR_Bonus"] = df.loc[mask_prog, "OR_Bonus"] + 0.05

            # Class compatibility bonus
            if "ccs_5" in df.columns:
                df["Class_Bonus"] = ((pd.to_numeric(df["ccs_5"], errors="coerce") - 3.0) / 2.0) * w_cls
            else:
                df["Class_Bonus"] = 0.0

            # Conditions bonus
            cond_cols = [c for c in ["dist_place","cls_place","runpm1_place","trackstyle_place","crs_place","lhrh_place","going_place"] if c in df.columns]
            if cond_cols:
                df["_cond_mean"] = df[cond_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
                df["Cond_Bonus"] = ((df["_cond_mean"] - 0.2) * 0.5).fillna(0.0) * (w_cond / 0.5)
            else:
                df["Cond_Bonus"] = 0.0

            # Final Suitability (using Hybrid-adjusted Suitability_Base)
            df["Suitability"] = df["Suitability_Base"] + df["OR_Bonus"] + df["Class_Bonus"] + df["Cond_Bonus"]
            if clip5:
                df["Suitability"] = df["Suitability"].clip(1.0, 5.0)

            # Show ALL runners
            show_cols = [
                "Horse","RS_Avg","RS_Avg10","Style","AdjSpeed","ΔvsPar","LCP",
                "PaceFit_Even","PaceFit_Strong","PaceFit",
                "SpeedFit","wp","ws",
                "Suitability_Base_Even","Suitability_Base_Strong","Suitability_Base_Hybrid","Suitability_Base",
                "OR_Bonus","Class_Bonus","Cond_Bonus","Suitability",
                "Scenario","Confidence"
            ]
            extra_cols = [c for c in ["today_or","max_or_10","handicap_runs","ccs_5",
                                      "dist_place","cls_place","runpm1_place","trackstyle_place",
                                      "crs_place","lhrh_place","going_place"] if c in df.columns]
            out = df[show_cols + extra_cols].sort_values(
                ["Suitability","Suitability_Base","SpeedFit"], ascending=False
            )

            st.subheader(
                f"Projected Pace: {df['Scenario'].iloc[0]} (confidence {conf_display}) — "
                f"Band: {'5f' if distance_f<=5.5 else ('6f' if distance_f<=6.5 else 'route')}"
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

            st.markdown("### Suitability Ratings (all runners)")
            st.dataframe(out, use_container_width=True)

            st.download_button(
                "💾 Download Suitability CSV",
                out.to_csv(index=False),
                "paceform_suitability.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
