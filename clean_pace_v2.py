# clean_pace_v2_threeinone.py
# One paste → split into:
#   1) Run Style Figure
#   2) Official Ratings (with unexposed logic + highest-winning OR by code)
#   3) Race Class - Today: Class N £X
# Then preview and export a combined CSV.
#
# Tab 2 lets you parse each section separately (debug).

from __future__ import annotations

import re
from io import StringIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------
# App Meta
# -------------------------------------------------
st.set_page_config(page_title="CleanPace v2 — 3-in-1 Splitter", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v2 — Single Paste (Run Style + Official Ratings + Race Class)")
st.caption("Paste the whole page (Run Style → Official Ratings → Race Class - Today: Class N £X), click Process, and download a combined CSV.")

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

# -------------------------------------------------
# Splitter for the 3-in-1 paste
# -------------------------------------------------
def split_three_sections(raw: str) -> Tuple[str, str, str]:
    """
    Split big pasted text into (run_style, official_ratings, race_class) sections.
    Anchors (case-insensitive):
      - 'Run Style Figure'
      - 'Official Ratings'
      - 'Race Class - Today'
    Must appear in that order.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    def find_idx(pattern: str) -> int:
        m = re.search(pattern, text, flags=re.I)
        return m.start() if m else -1

    i_rs = find_idx(r"\bRun\s*Style\s*Figure\b")
    i_or = find_idx(r"\bOfficial\s*Ratings\b")
    i_rc = find_idx(r"\bRace\s*Class\s*-\s*Today\b")

    if min(i_rs, i_or, i_rc) == -1:
        raise ValueError("Anchors not found. Expect headings: 'Run Style Figure', 'Official Ratings', 'Race Class - Today: ...'.")
    if not (i_rs < i_or < i_rc):
        raise ValueError("Anchors out of order. Expected: Run Style → Official Ratings → Race Class.")

    rs_text = text[i_rs:i_or].strip()
    or_text = text[i_or:i_rc].strip()
    rc_text = text[i_rc:].strip()
    return rs_text, or_text, rc_text

# -------------------------------------------------
# Section parsers
# -------------------------------------------------
# 1) Run Style Figure
def parse_run_style(rs_text: str, front_thr=1.6, prom_thr=2.4, mid_thr=3.0) -> pd.DataFrame:
    lines = rs_text.splitlines()
    # remove title line if present
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

    df["RS_Avg"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto5]), axis=1)
    df["RS_Avg10"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto10]), axis=1)
    df["RS_Cat"] = df["RS_Avg"].apply(lambda x: _rs_category(x, front_thr, prom_thr, mid_thr))

    if "Dr.%" in df.columns:
        df["Dr.%"] = df["Dr.%"].astype(str).str.replace("%", "", regex=False)
        df["Dr.%"] = pd.to_numeric(df["Dr.%"], errors="coerce")

    keep = ["Horse"] + lto10 + ["RS_Avg","RS_Avg10","RS_Cat"] + \
           [c for c in ["Mode 5","Mode 10","Total","Mode All","Draw","Dr.%"] if c in df.columns]
    return df[keep]

# 2) Official Ratings
def _parse_or_rating_line(s: str) -> Optional[int]:
    s = str(s).strip()
    if not s:
        return None
    left = s.split("/")[0].strip()
    m = re.search(r"-?\d+", left)
    return int(m.group(0)) if m else None

def parse_official_ratings(or_text: str) -> pd.DataFrame:
    """
    Parses the 'Official Ratings' block and computes:
      - today_or, max_or_10, or_3back, re_lb, d_or_trend, ts_5
      - handicap_runs (count of valid trailing ORs; proxy for exposure)
      - hwin_or (highest winning OR for THIS race type if supplied via 'Highest' col)
      - delta_hwin_code (hwin_or - today_or)
      - delta_hwin_adj (exposure-tolerant delta when today_or > hwin_or)
      - rating_fit_code_5 (1–5 from delta_hwin_code)
      - rating_fit_code_adj_5 (1–5 from delta_hwin_adj)
      - ts_5_adj (ts_5 +0.5 nudge for unexposed over max_or_10)
      - rating_context_index = 0.7*ts_5_adj + 0.3*rating_fit_code_adj_5
    """
    lines = [l for l in or_text.splitlines() if l.strip()]
    # strip title line
    if lines and lines[0].strip().lower().startswith("official ratings"):
        lines = lines[1:]

    rows = []
    i = 0

    def _bucket5(x: Optional[float]) -> Optional[int]:
        if x is None: return None
        if x >= 8:  return 5
        if x >= 4:  return 4
        if x >= 0:  return 3
        if x >= -4: return 2
        return 1

    while i < len(lines):
        parts = lines[i].split("\t")

        # header row ("Horse  To Last  Today  Last  Highest ...")
        if i == 0 and parts and parts[0].strip().lower().startswith("horse"):
            i += 1
            continue

        # a new horse header row has at least 5 tabbed fields in your format
        if len(parts) >= 5:
            horse = parts[0].strip()
            today_str = parts[2].strip()
            highest_header_str = parts[4].strip()  # treated as highest winning OR for this race code if that's how you feed it

            # advance to rating lines for this horse
            i += 1
            ratings: List[Optional[int]] = []
            j = i
            while j < len(lines):
                nxt = lines[j].split("\t")
                if len(nxt) >= 5:  # next horse header encountered
                    break
                r = _parse_or_rating_line(lines[j])
                ratings.append(r)
                j += 1
            i = j

            # today_or (accept "+60", "60", etc.)
            m = re.search(r"-?\d+", today_str)
            today_or = int(m.group(0)) if m else None

            valid_ratings = [x for x in ratings if x is not None]
            max_or_10 = max(valid_ratings) if valid_ratings else None
            or_3back = valid_ratings[2] if len(valid_ratings) >= 3 else None

            # baseline relief & trend
            re_lb = (max_or_10 - today_or) if (max_or_10 is not None and today_or is not None) else None
            d_or_trend = (today_or - or_3back) if (today_or is not None and or_3back is not None) else None

            # ts_5 from re_lb with trend nudge
            ts_5 = None
            if re_lb is not None:
                ts_5 = 5 if re_lb >= 8 else 4 if re_lb >= 4 else 3 if re_lb >= 0 else 2 if re_lb >= -4 else 1
                if d_or_trend is not None:
                    if d_or_trend >= 2:   ts_5 = min(5, ts_5 + 0.5)
                    elif d_or_trend <= -2: ts_5 = max(1, ts_5 - 0.5)

            # exposure: count of valid trailing OR entries (proxy for handicap runs)
            handicap_runs = sum(1 for r in valid_ratings if (r is not None and r > 0))

            # try to extract highest-winning OR for this code (if fed via 'Highest' column on header)
            hwin_or = None
            try:
                mH = re.search(r"-?\d+", highest_header_str)
                hwin_or = int(mH.group(0)) if mH else None
            except Exception:
                hwin_or = None

            # raw delta vs highest-win mark (if available)
            delta_hwin_code = (hwin_or - today_or) if (hwin_or is not None and today_or is not None) else None

            # exposure-tolerant adjustment τ when today_or > hwin_or
            if handicap_runs < 3:
                tau = 6
            elif handicap_runs <= 5:
                tau = 4
            elif handicap_runs <= 9:
                tau = 2
            else:
                tau = 0

            if (hwin_or is not None) and (today_or is not None) and (today_or > hwin_or):
                delta_hwin_adj = (hwin_or + tau) - today_or
            else:
                delta_hwin_adj = delta_hwin_code  # already ≥ 0 or not applicable

            rating_fit_code_5 = _bucket5(delta_hwin_code)
            rating_fit_code_adj_5 = _bucket5(delta_hwin_adj)

            # nudge ts_5 for unexposed types stepping above prior max_or_10
            ts_5_adj = ts_5
            if (handicap_runs < 6) and (today_or is not None) and (max_or_10 is not None) and (today_or > max_or_10):
                ts_5_adj = min(5, (ts_5 or 0) + 0.5)

            rating_context_index = None
            if (ts_5_adj is not None) and (rating_fit_code_adj_5 is not None):
                rating_context_index = round(0.7 * ts_5_adj + 0.3 * rating_fit_code_adj_5, 2)

            rows.append({
                "Horse": horse,
                "today_or": today_or,
                "or_3back": or_3back,
                "max_or_10": max_or_10,
                "re_lb": re_lb,
                "d_or_trend": d_or_trend,
                "ts_5": ts_5,
                "handicap_runs": handicap_runs,                 # NEW
                "hwin_or": hwin_or,                             # NEW
                "delta_hwin_code": delta_hwin_code,             # NEW
                "delta_hwin_adj": delta_hwin_adj,               # NEW
                "rating_fit_code_5": rating_fit_code_5,         # NEW
                "rating_fit_code_adj_5": rating_fit_code_adj_5, # NEW
                "ts_5_adj": ts_5_adj,                           # NEW
                "rating_context_index": rating_context_index    # NEW
            })
        else:
            i += 1

    return pd.DataFrame(rows)

# 3) Race Class / Prize
def parse_race_class(rc_text: str) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Parse 'Race Class - Today: Class X £Y' section.
    Supports two formats:
      A) Tabular: columns include 'Horse', 'Avg 3', 'Avg 5'
      B) Block:  Horse on its own line, followed by several 'Cl.. £..' lines,
                 then two lines with money only (Avg 3 then Avg 5).
    Returns (df, today_class_value)
    """
    lines = rc_text.splitlines()

    # --- extract today's class value from header line
    today_class_value = None
    if lines and lines[0].strip().lower().startswith("race class - today"):
        m = re.search(r"£\s*[0-9]*\.?[0-9]+\s*[KkMm]?", lines[0])
        if m:
            today_class_value = _to_float_money(m.group(0))
        # drop header
        rc_text_body = "\n".join(lines[1:])
    else:
        rc_text_body = rc_text

    # -------- Attempt A: tabular parse
    def _try_tabular(text: str) -> Optional[pd.DataFrame]:
        try:
            df = _read_table_guess(text)
        except Exception:
            return None
        if "Horse" not in df.columns:
            return None
        df["Horse"] = df["Horse"].astype(str).str.strip()
        # parse money if present
        if "Avg 3" in df.columns:
            df["avg3_prize"] = df["Avg 3"].apply(_to_float_money)
        else:
            df["avg3_prize"] = np.nan
        if "Avg 5" in df.columns:
            df["avg5_prize"] = df["Avg 5"].apply(_to_float_money)
        else:
            df["avg5_prize"] = np.nan
        keep = ["Horse", "avg3_prize", "avg5_prize"]
        return df[keep]

    df_tab = _try_tabular(rc_text_body)

    # If tabular worked AND at least some avg values parsed, use it
    if df_tab is not None and (
        df_tab["avg3_prize"].notna().any() or df_tab["avg5_prize"].notna().any()
    ):
        df_out = df_tab.copy()
        df_out["today_class_value"] = today_class_value
        def ccs_from_cd(cd: Optional[float]) -> Optional[int]:
            if cd is None or (isinstance(cd, float) and np.isnan(cd)): return None
            if cd >= 0.40: return 5
            if cd >= 0.15: return 4
            if cd >= -0.15: return 3
            if cd >= -0.40: return 2
            return 1
        df_out["cd"] = (df_out["avg3_prize"] / df_out["today_class_value"]) - 1 if today_class_value else np.nan
        df_out["ccs_5"] = df_out["cd"].apply(ccs_from_cd)
        return df_out[["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]], today_class_value

    # -------- Attempt B: block parser
    # Clean out the table header line if present (Horse Lto1 Lto2 ...)
    body_lines = [ln for ln in rc_text_body.splitlines() if ln.strip()]
    if body_lines and body_lines[0].strip().lower().startswith("horse"):
        body_lines = body_lines[1:]

    # A horse line is a non-empty line that does NOT start with 'Cl' and does NOT start with '£'
    def is_horse_line(s: str) -> bool:
        t = s.strip()
        if not t: return False
        if t.lower().startswith("cl"): return False
        if t.startswith("£"): return False
        # avoid stray 'Avg' labels if ever present
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
            # from the block, take the last two money lines as Avg3, Avg5 (order in your sample)
            money_vals = [ _to_float_money(x) for x in block if "£" in x ]
            avg3, avg5 = (money_vals[-2], money_vals[-1]) if len(money_vals) >= 2 else (None, None)
            horses.append({"Horse": horse, "avg3_prize": avg3, "avg5_prize": avg5})
        else:
            i += 1

    df_blk = pd.DataFrame(horses)
    if df_blk.empty:
        # fallback: at least return an empty shell with today_class_value
        return pd.DataFrame(columns=["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]), today_class_value

    df_blk["today_class_value"] = today_class_value

    def ccs_from_cd(cd: Optional[float]) -> Optional[int]:
        if cd is None or (isinstance(cd, float) and np.isnan(cd)): return None
        if cd >= 0.40: return 5
        if cd >= 0.15: return 4
        if cd >= -0.15: return 3
        if cd >= -0.40: return 2
        return 1

    df_blk["cd"] = (df_blk["avg3_prize"] / df_blk["today_class_value"]) - 1 if today_class_value else np.nan
    df_blk["ccs_5"] = df_blk["cd"].apply(ccs_from_cd)

    return df_blk[["Horse","avg3_prize","avg5_prize","today_class_value","cd","ccs_5"]], today_class_value


# -------------------------------------------------
# UI — Two tabs: (1) Single-Paste + Process, (2) Manual 3-box (debug)
# -------------------------------------------------
TAB1, TAB2 = st.tabs(["Single Paste (auto-split)", "Manual 3-box (debug)"])

with TAB1:
    st.subheader("Single Paste")
    st.caption("Paste the whole text including the three sections in order. The app will split on the headings.")
    big = st.text_area("Paste the entire block here", height=420)

    col = st.columns(3)
    with col[0]: front_thr = st.number_input("Front <", value=1.6, step=0.1)
    with col[1]: prom_thr  = st.number_input("Prominent <", value=2.4, step=0.1)
    with col[2]: mid_thr   = st.number_input("Mid <", value=3.0, step=0.1)

    if st.button("🚀 Process All"):
        try:
            rs_text, or_text, rc_text = split_three_sections(big)

            rs_df = parse_run_style(rs_text, front_thr, prom_thr, mid_thr)
            # Ensure thresholds applied (already done in parse_run_style)
            or_df = parse_official_ratings(or_text)
            rc_df, today_cls_val = parse_race_class(rc_text)

            st.success("Split & parsed ✔")
            s1, s2, s3 = st.tabs(["Run Style ✓", "Official Ratings ✓", "Race Class ✓"])
            with s1: st.dataframe(rs_df, use_container_width=True)
            with s2: st.dataframe(or_df, use_container_width=True)
            with s3:
                st.write(f"Today Class £: **{today_cls_val if today_cls_val else 'n/a'}**")
                st.dataframe(rc_df, use_container_width=True)

            # Combine (outer joins on Horse)
            for df in (rs_df, or_df, rc_df):
                df["Horse"] = df["Horse"].astype(str).str.strip()
            merged = rs_df.merge(or_df, on="Horse", how="outer").merge(rc_df, on="Horse", how="outer")

            st.markdown("### 🧩 Combined Output")
            st.dataframe(merged, use_container_width=True)
            st.download_button(
                "💾 Download Combined CSV",
                merged.to_csv(index=False),
                "cleanpace_threeinone_combined.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed: {e}")

with TAB2:
    st.subheader("Manual 3-box (debug)")
    c1, c2 = st.columns(2)
    with c1:
        rs_raw = st.text_area("Run Style Figure (only)", height=220)
        if st.button("Parse RS", key="prs"):
            try:
                rs_df2 = parse_run_style(rs_raw, front_thr, prom_thr, mid_thr)
                st.dataframe(rs_df2, use_container_width=True)
            except Exception as e:
                st.error(e)
    with c2:
        or_raw = st.text_area("Official Ratings (only)", height=220)
        if st.button("Parse OR", key="por"):
            try:
                or_df2 = parse_official_ratings(or_raw)
                st.dataframe(or_df2, use_container_width=True)
            except Exception as e:
                st.error(e)
    rc_raw = st.text_area("Race Class - Today: ... (only)", height=180)
    if st.button("Parse Race Class", key="prc"):
        try:
            rc_df2, val = parse_race_class(rc_raw)
            st.write(f"Today Class £: {val}")
            st.dataframe(rc_df2, use_container_width=True)
        except Exception as e:
            st.error(e)

st.markdown("---")
st.caption(
    "Notes: RS_Avg uses Lto1–Lto5 (ignoring 0s), RS_Cat derives from RS_Avg with thresholds set above. "
    "Official Ratings adds unexposed handling and a highest-winning OR by code (if supplied via 'Highest'). "
    "Race Class extracts Today Class £ from the header and computes cd & ccs_5."
)

