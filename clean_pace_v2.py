# clean_pace_v2_allinone.py
# One main tab with TWO paste boxes:
#  Box A: Run Style + Official Ratings + Race Class (single paste; auto-split)
#  Box B: Boxes 4 & 5 combined (speed + conditions in one paste)
# Adds a minimal Suitability Score preview (pace scenario + weights + class par).

from __future__ import annotations

import re
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------
# App Meta
# -------------------------------------------------
st.set_page_config(page_title="CleanPace v2 — All-in-One", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v2 — All-in-One Normaliser")
st.caption("Box A: RS/OR/Class (single paste). Box B: Boxes 4&5 (Speed + Conditions). One click → combined CSV + Suitability preview.")

# -------------------------------------------------
# Generic helpers
# -------------------------------------------------
def _read_table_guess(text: str) -> pd.DataFrame:
    text = text.strip()
    try:
        return pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(StringIO(text))

def _mean_ignore_zero(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if pd.notna(v) and float(v) > 0]
    return round(sum(vals) / len(vals), 2) if vals else None

def _rs_category(avg: Optional[float], f=1.6, p=2.4, m=3.0) -> Optional[str]:
    if avg is None or (isinstance(avg, float) and np.isnan(avg)):
        return None
    if avg < f: return "Front"
    if avg < p: return "Prominent"
    if avg < m: return "Mid"
    return "Hold-up"

def _to_float_money(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace(" ", "")
    if not s:
        return None
    m = re.search(r"£\s*([0-9]*\.?[0-9]+)\s*([KkMm])?", s)
    if not m:
        return None
    num = float(m.group(1))
    suf = (m.group(2) or "").upper()
    if suf == "K": num *= 1000
    if suf == "M": num *= 1_000_000
    return float(num)

def _numbers(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]

# -------------------------------------------------
# Splitter for the 3-in-1 paste (Box A)
# -------------------------------------------------
def split_three_sections(raw: str) -> Tuple[str, str, str]:
    """
    Expect headings (case-insensitive) in this order:
      Run Style Figure
      Official Ratings
      Race Class - Today
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    def find_idx(pattern: str) -> int:
        m = re.search(pattern, text, flags=re.I)
        return m.start() if m else -1

    i_rs = find_idx(r"\bRun\s*Style\s*Figure\b")
    i_or = find_idx(r"\bOfficial\s*Ratings\b")
    i_rc = find_idx(r"\bRace\s*Class\s*-\s*Today\b")

    if min(i_rs, i_or, i_rc) == -1:
        raise ValueError("Anchors not found. Need 'Run Style Figure', 'Official Ratings', 'Race Class - Today: ...'.")
    if not (i_rs < i_or < i_rc):
        raise ValueError("Anchors out of order. Expected: Run Style → Official Ratings → Race Class.")

    rs_text = text[i_rs:i_or].strip()
    or_text = text[i_or:i_rc].strip()
    rc_text = text[i_rc:].strip()
    return rs_text, or_text, rc_text

# -------------------------------------------------
# Section parsers for Box A
# -------------------------------------------------
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

    df["RS_Avg"]   = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto5]), axis=1)
    df["RS_Avg10"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto10]), axis=1)
    df["RS_Cat"]   = df["RS_Avg"].apply(lambda x: _rs_category(x, front_thr, prom_thr, mid_thr))

    if "Dr.%" in df.columns:
        df["Dr.%"] = df["Dr.%"].astype(str).str.replace("%", "", regex=False)
        df["Dr.%"] = pd.to_numeric(df["Dr.%"], errors="coerce")

    keep = ["Horse"] + lto10 + ["RS_Avg","RS_Avg10","RS_Cat"] + \
           [c for c in ["Mode 5","Mode 10","Total","Mode All","Draw","Dr.%"] if c in df.columns]
    return df[keep]

def _parse_or_rating_line(s: str) -> Optional[int]:
    s = str(s).strip()
    if not s:
        return None
    left = s.split("/")[0].strip()
    m = re.search(r"-?\d+", left)
    return int(m.group(0)) if m else None

def parse_official_ratings(or_text: str) -> pd.DataFrame:
    """
    Produces:
      today_or, or_3back, max_or_10, re_lb, d_or_trend, ts_5,
      handicap_runs, hwin_or, delta_hwin_code, delta_hwin_adj,
      rating_fit_code_5, rating_fit_code_adj_5, ts_5_adj,
      rating_context_index
    """
    lines = [l for l in or_text.splitlines() if l.strip()]
    if lines and lines[0].strip().lower().startswith("official ratings"):
        lines = lines[1:]

    rows = []
    i = 0
    while i < len(lines):
        parts = lines[i].split("\t")
        if i == 0 and parts and parts[0].strip().lower().startswith("horse"):
            i += 1
            continue

        if len(parts) >= 5:
            horse = parts[0].strip()
            today_str = parts[2].strip()
            highest_header_str = parts[4].strip()

            i += 1
            ratings: List[Optional[int]] = []
            j = i
            while j < len(lines):
                nxt = lines[j].split("\t")
                if len(nxt) >= 5:
                    break
                ratings.append(_parse_or_rating_line(lines[j]))
                j += 1
            i = j

            today_or = int(re.search(r"-?\d+", today_str).group(0)) if re.search(r"-?\d+", today_str) else None
            valid_ratings = [x for x in ratings if x is not None]
            max_or_10 = max(valid_ratings) if valid_ratings else None
            or_3back  = valid_ratings[2] if len(valid_ratings) >= 3 else None

            re_lb = (max_or_10 - today_or) if (max_or_10 is not None and today_or is not None) else None
            d_or_trend = (today_or - or_3back) if (today_or is not None and or_3back is not None) else None

            ts_5 = None
            if re_lb is not None:
                ts_5 = 5 if re_lb >= 8 else 4 if re_lb >= 4 else 3 if re_lb >= 0 else 2 if re_lb >= -4 else 1
                if d_or_trend is not None:
                    if d_or_trend >= 2:   ts_5 = min(5, ts_5 + 0.5)
                    elif d_or_trend <= -2: ts_5 = max(1, ts_5 - 0.5)

            handicap_runs = sum(1 for r in valid_ratings if (r is not None and r > 0))

            hwin_or = None
            mH = re.search(r"-?\d+", highest_header_str)
            if mH: hwin_or = int(mH.group(0))

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

def parse_race_class(rc_text: str) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Supports table or block. Returns Horse + (avg3_prize, avg5_prize, today_class_value, cd, ccs_5)
    """
    lines = rc_text.splitlines()
    today_class_value = None
    if lines and lines[0].strip().lower().startswith("race class - today"):
        m = re.search(r"£\s*[0-9]*\.?[0-9]+\s*[KkMm]?", lines[0])
        if m:
            today_class_value = _to_float_money(m.group(0))
        rc_text_body = "\n".join(lines[1:])
    else:
        rc_text_body = rc_text

    # try tabular
    def _try_tabular(text: str) -> Optional[pd.DataFrame]:
        try:
            df = _read_table_guess(text)
        except Exception:
            return None
        if "Horse" not in df.columns:
            return None
        df["Horse"] = df["Horse"].astype(str).str.strip()
        df["avg3_prize"] = df.get("Avg 3", np.nan).apply(_to_float_money) if "Avg 3" in df.columns else np.nan
        df["avg5_prize"] = df.get("Avg 5", np.nan).apply(_to_float_money) if "Avg 5" in df.columns else np.nan
        return df[["Horse","avg3_prize","avg5_prize"]]

    df_tab = _try_tabular(rc_text_body)
    if df_tab is not None and (df_tab["avg3_prize"].notna().any() or df_tab["avg5_prize"].notna().any()):
        df_out = df_tab.copy()
        df_out["today_class_value"] = today_class_value
        def ccs_from_cd(cd: Optional[float]) -> Optional[int]:
            if cd is None or (isinstance(cd, float) and np.isnan(cd)): return None
            if cd >= 0.40: return 5
            if cd >= 0.15: return 4
            if cd >= -0.15: return 3
            if cd >= -0.40: return 2
            return 1
        df_out["cd"]    = (df_out["avg3_prize"] / df_out["today_class_value"]) - 1 if today_class_value else np.nan
        df_out["ccs_5"] = df_out["cd"].apply(ccs_from_cd)
        return df_out[["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]], today_class_value

    # fallback block
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

    horses = []
    i = 0
    n = len(body_lines)
    while i < n:
        line = body_lines[i].strip()
        if is_horse_line(line):
            horse = line
            i += 1
            block = []
            while i < n and not is_horse_line(body_lines[i]):
                block.append(body_lines[i].strip())
                i += 1
            money_vals = [_to_float_money(x) for x in block if "£" in x]
            avg3, avg5 = (money_vals[-2], money_vals[-1]) if len(money_vals) >= 2 else (None, None)
            horses.append({"Horse": horse, "avg3_prize": avg3, "avg5_prize": avg5})
        else:
            i += 1

    df_blk = pd.DataFrame(horses)
    if df_blk.empty:
        return pd.DataFrame(columns=["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]), today_class_value

    df_blk["today_class_value"] = today_class_value
    def ccs_from_cd(cd: Optional[float]) -> Optional[int]:
        if cd is None or (isinstance(cd, float) and np.isnan(cd)): return None
        if cd >= 0.40: return 5
        if cd >= 0.15: return 4
        if cd >= -0.15: return 3
        if cd >= -0.40: return 2
        return 1
    df_blk["cd"]    = (df_blk["avg3_prize"] / df_blk["today_class_value"]) - 1 if today_class_value else np.nan
    df_blk["ccs_5"] = df_blk["cd"].apply(ccs_from_cd)
    return df_blk[["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]], today_class_value

# -------------------------------------------------
# Box B — Boxes 4 & 5 (single paste) parsing
# -------------------------------------------------
BOX_ORDER = [
    "win_pct", "form_avg", "speed_series",
    "crs", "dist", "lhrh", "going", "cls", "runpm1", "trackstyle"
]

def _looks_like_name(line: str) -> bool:
    t = line.strip()
    if not t: return False
    if t.lower().startswith("horse"): return False
    if re.fullmatch(r"[\d\W_]+", t): return False
    if t.startswith("(") or t.startswith("£"): return False
    return True

def _parse_wpt_value(val: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    val = str(val).strip()
    m = re.search(r"\((\d+)\s*/\s*(\d+)\s*/\s*(\d+)\)", val)
    if not m:
        return None, None, None, None
    w = int(m.group(1)); p = int(m.group(2)); t = int(m.group(3))
    place = None if t == 0 else round((w + p) / t, 3)
    return place, w, p, t

def _parse_speed_series(s: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    nums = _numbers(s)
    if not nums:
        return None, None, None, None
    last = nums[-1]; highest = max(nums)
    avg_last_3 = round(sum(nums[-3:])/min(3, len(nums)), 1)
    avg_all    = round(sum(nums)/len(nums), 1)
    return last, highest, avg_last_3, avg_all

def parse_box45_single(raw: str) -> pd.DataFrame:
    lines = [ln for ln in raw.replace("\r\n","\n").replace("\r","\n").split("\n") if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["Horse"])
    if lines[0].strip().lower().startswith("horse"):
        lines = lines[1:]

    i, n = 0, len(lines)
    recs: List[Dict[str, object]] = []
    while i < n:
        while i < n and not _looks_like_name(lines[i]):
            i += 1
        if i >= n: break
        name = lines[i].strip(); i += 1

        values: List[str] = []
        while i < n and len(values) < 10:
            if _looks_like_name(lines[i]): break
            values.append(lines[i].strip()); i += 1
        if len(values) < 10: values += [None] * (10 - len(values))
        else: values = values[:10]

        data = {"Horse": name}
        for key, val in zip(BOX_ORDER, values):
            data[key] = val
        recs.append(data)

    df = pd.DataFrame(recs)
    if df.empty: return df
    df["Horse"] = df["Horse"].astype(str).str.strip()
    return df

def build_speed_conditions(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    last_list, high_list, avg3_list, all_list, keyavg_list = [], [], [], [], []
    for s in df_raw["speed_series"].fillna(""):
        last, high, avg3, avgall = _parse_speed_series(s)
        last_list.append(last); high_list.append(high)
        avg3_list.append(avg3);  all_list.append(avgall)
        keyavg_list.append(None if (last is None or high is None or avg3 is None) else round((last+high+avg3)/3, 1))

    out = df_raw.copy()
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

# -------------------------------------------------
# Suitability Preview Helpers
# -------------------------------------------------
PACEFIT = {
    "Slow":   {"Front": 5, "Prominent": 4, "Mid": 3, "Hold-up": 2},
    "Even":   {"Front": 4, "Prominent": 5, "Mid": 4, "Hold-up": 3},
    "Strong": {"Front": 2, "Prominent": 3, "Mid": 4, "Hold-up": 5},
    "Very Strong": {"Front": 1, "Prominent": 2, "Mid": 4, "Hold-up": 5},
}

def _speed_score(dvp: Optional[float]) -> float:
    # Light penalty for missing KeySpeedAvg (no Δ vs Par)
    if dvp is None or pd.isna(dvp):
        return 2.3
    if dvp >= 2:
        return 5.0
    if dvp >= -1:
        return 4.0
    if dvp >= -4:
        return 3.0
    if dvp >= -8:
        return 2.0
    return 1.0

def build_suitability_preview(merged: pd.DataFrame, scenario: str, wp: float, class_par: float) -> pd.DataFrame:
    """
    Minimal suitability using:
      - RS_Cat from RS_Avg (5) only
      - Δ vs Par from KeySpeedAvg - class_par
      - PaceFit from selected scenario
      - Suitability = PaceFit*wp + SpeedFit*(1-wp)
    """
    if merged.empty:
        return pd.DataFrame()

    df = merged.copy()
    df["RS_Cat_used"] = df.get("RS_Cat")
    df["ΔvsPar"] = df.get("KeySpeedAvg") - float(class_par)

    pace_map = PACEFIT.get(scenario, PACEFIT["Even"])
    df["PaceFit"] = df["RS_Cat_used"].map(pace_map).fillna(3).astype(float)
    df["SpeedFit"] = df["ΔvsPar"].apply(_speed_score)

    ws = 1 - float(wp)
    df["wp"] = float(wp)
    df["ws"] = ws
    df["Suitability"] = (df["PaceFit"] * df["wp"] + df["SpeedFit"] * df["ws"]).round(1)

    show_cols = ["Horse","RS_Avg","RS_Avg10","RS_Cat_used","KeySpeedAvg","ΔvsPar","PaceFit","SpeedFit","wp","ws","Suitability"]
    show_cols = [c for c in show_cols if c in df.columns]
    out = df[show_cols].sort_values(["Suitability","SpeedFit"], ascending=False)
    return out

# -------------------------------------------------
# UI — Tabs
# -------------------------------------------------
TAB_MAIN, TAB_DEBUG = st.tabs(["All Inputs (2 boxes)", "Debug"])

with TAB_MAIN:
    st.subheader("Box A — RS/OR/Class (single paste)")
    st.caption("Paste the full block containing: Run Style Figure → Official Ratings → Race Class - Today: Class N £X.")
    big = st.text_area("Box A paste", height=320, key="boxA")

    st.subheader("Box B — Boxes 4 & 5 (single paste)")
    st.caption("Paste the single block that starts with the header row 'Horse Win % Form Figures (Avg) ...'.")
    box45_raw = st.text_area("Box B paste", height=320, key="boxB")

    col_thr = st.columns(3)
    with col_thr[0]: front_thr = st.number_input("Front <", value=1.6, step=0.1)
    with col_thr[1]: prom_thr  = st.number_input("Prominent <", value=2.4, step=0.1)
    with col_thr[2]: mid_thr   = st.number_input("Mid <", value=3.0, step=0.1)

    if st.button("🚀 Process All (A + B)"):
        try:
            # --- Box A ---
            rs_text, or_text, rc_text = split_three_sections(big)
            rs_df = parse_run_style(rs_text, front_thr, prom_thr, mid_thr)
            or_df = parse_official_ratings(or_text)
            rc_df, today_cls_val = parse_race_class(rc_text)

            a1, a2, a3 = st.tabs(["RS ✓", "OR ✓", "Class ✓"])
            with a1: st.dataframe(rs_df, use_container_width=True)
            with a2: st.dataframe(or_df, use_container_width=True)
            with a3:
                st.write(f"Today Class £: **{today_cls_val if today_cls_val else 'n/a'}**")
                st.dataframe(rc_df, use_container_width=True)

            # --- Box B ---
            base_b = parse_box45_single(box45_raw) if box45_raw.strip() else pd.DataFrame(columns=["Horse"])
            if base_b.empty:
                st.warning("Box B: No data parsed. Check the header and that each horse has 10 lines.")
                b_out = pd.DataFrame(columns=["Horse"])
            else:
                b_out = build_speed_conditions(base_b)
                st.markdown("### Boxes 4 & 5 → Speed & Conditions")
                st.dataframe(b_out, use_container_width=True)

            # --- Merge everything
            for dfm in (rs_df, or_df, rc_df, b_out):
                if "Horse" in dfm.columns:
                    dfm["Horse"] = dfm["Horse"].astype(str).str.strip()
            merged = rs_df.merge(or_df, on="Horse", how="outer") \
                          .merge(rc_df, on="Horse", how="outer") \
                          .merge(b_out, on="Horse", how="outer")

            st.markdown("## 🧩 Combined Output (A + B)")
            st.dataframe(merged, use_container_width=True)
            st.download_button(
                "💾 Download Combined CSV",
                merged.to_csv(index=False),
                "cleanpace_allinone_combined.csv",
                mime="text/csv"
            )

            # -------- Suitability Preview --------
            st.markdown("## ⭐ Suitability Score — Minimal Preview")
            with st.expander("Configure preview", expanded=True):
                c = st.columns(4)
                scenario = c[0].selectbox("Pace Scenario", ["Slow","Even","Strong","Very Strong"], index=1)
                wp = c[1].slider("Pace weight (wp)", 0.3, 0.8, 0.50, 0.05)
                class_par = c[2].number_input("Class Par (for Δ vs Par)", value=77.0, step=0.5)
                show_top_n = int(c[3].number_input("Show Top N", value=5, step=1, min_value=1))

            suit_df = build_suitability_preview(merged, scenario, wp, class_par)
            if suit_df.empty:
                st.info("Suitability preview needs RS_Cat (from Box A) and KeySpeedAvg (from Box B).")
            else:
                st.dataframe(suit_df, use_container_width=True)
                topN = suit_df.head(show_top_n)[["Horse","Suitability","RS_Cat_used","ΔvsPar"]]
                st.markdown("### Top Picks")
                for i, (_, r) in enumerate(topN.iterrows()):
                    medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i] if i < 5 else f"{i+1}."
                    st.write(f"{medal} **{r['Horse']}** — Score **{r['Suitability']}** | {r['RS_Cat_used']} | ΔvsPar {r['ΔvsPar']:.1f}")

        except Exception as e:
            st.error(f"Failed: {e}")

with TAB_DEBUG:
    st.subheader("Quick Parsers (use for troubleshooting)")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Run Style Figure (only)")
        rs_only = st.text_area("RS only", height=180, key="dbg_rs")
        if st.button("Parse RS", key="dbg_prs"):
            try:
                st.dataframe(parse_run_style(rs_only), use_container_width=True)
            except Exception as e:
                st.error(e)

        st.caption("Official Ratings (only)")
        or_only = st.text_area("OR only", height=180, key="dbg_or")
        if st.button("Parse OR", key="dbg_por"):
            try:
                st.dataframe(parse_official_ratings(or_only), use_container_width=True)
            except Exception as e:
                st.error(e)

    with c2:
        st.caption("Race Class - Today: ... (only)")
        rc_only = st.text_area("Class only", height=180, key="dbg_rc")
        if st.button("Parse Class", key="dbg_prc"):
            try:
                dfc, val = parse_race_class(rc_only)
                st.write(f"Today Class £: {val}")
                st.dataframe(dfc, use_container_width=True)
            except Exception as e:
                st.error(e)

        st.caption("Boxes 4 & 5 (single paste)")
        b_only = st.text_area("Box 4/5 only", height=180, key="dbg_b45")
        if st.button("Parse Box 4/5", key="dbg_pb45"):
            try:
                base_dbg = parse_box45_single(b_only)
                st.markdown("**Parsed (name + 10 fields):**")
                st.dataframe(base_dbg, use_container_width=True)
                st.markdown("**Computed Speed & Conditions:**")
                st.dataframe(build_speed_conditions(base_dbg), use_container_width=True)
            except Exception as e:
                st.error(e)

st.markdown("---")
st.caption("Suitability = PaceFit × wp + SpeedFit × (1−wp). RS_Cat comes from RS_Avg (5 only). Δ vs Par uses KeySpeedAvg − Class Par.")
