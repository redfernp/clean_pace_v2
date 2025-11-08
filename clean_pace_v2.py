# clean_pace_v2_fivebox_full.py
# Streamlit cleaner with FIVE input boxes and a combined export for Suitability pipeline.
# Box 1: Run Style Figure (RS_Avg5/RS_Avg10, RS_Cat)
# Box 2: Official Ratings (today_or, or_3back, max_or_10, re_lb, d_or_trend, ts_5)
# Box 3: Race Class / Prize (avg3_prize, avg5_prize, today_class_value, cd, ccs_5)
# Box 4: Speed Figures -> key_speed_avg (AdjSpeed anchor) + last/high/avg3/avgall
# Box 5: Race Conditions (from same block format as Box 4) -> CRI features (dist/class/field/trackstyle etc.)

from __future__ import annotations

import re
from io import StringIO
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------
# App Meta
# --------------------------------------
st.set_page_config(page_title="CleanPace v2 – Five-Box Cleaner", page_icon="🏇", layout="wide")
st.title("🏇 CleanPace v2 — Five-Box Data Cleaner & Combiner")

st.markdown(
    """
Paste each source block into the correct box. Process each, then click **Combine & Export** to produce a single CSV
ready for your Suitability model.

**Key rules**
- `RS_Avg` uses **last 5** (Lto1–Lto5, ignoring 0s) and drives `RS_Cat`.
- `RS_Avg10` is shown for reference.
- Box 4 (Speed) provides **key_speed_avg** (= mean of Last, Highest, Avg3).
- Box 5 parses W/P/T strings into place% and 1–5 fitness scores; yields **CRI** features.
    """
)

# --------------------------------------
# Helpers
# --------------------------------------

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
    if avg < f:
        return "Front"
    if avg < p:
        return "Prominent"
    if avg < m:
        return "Mid"
    return "Hold-up"


def _to_float_money(val: str) -> Optional[float]:
    """Parse '£3.72K' -> 3720.0 (pounds)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # remove commas/spaces
    s = s.replace(",", "").replace(" ", "")
    m = re.search(r"£\s*([0-9]*\.?[0-9]+)\s*([KkMm])?", s)
    if not m:
        return None
    num = float(m.group(1))
    suf = m.group(2).upper() if m.group(2) else ""
    if suf == "K":
        num *= 1000
    elif suf == "M":
        num *= 1_000_000
    return float(num)


def _safe_int(x) -> Optional[int]:
    try:
        return int(str(x).replace("+", "").replace("-", "-").strip())
    except Exception:
        return None


# --------------------------------------
# Box 1 — Run Style Figure
# --------------------------------------

def parse_box1_run_style(raw: str) -> pd.DataFrame:
    """Parse Run Style Figure table and compute RS_Avg(5), RS_Avg10, RS_Cat."""
    lines = raw.strip().splitlines()
    if lines and lines[0].strip().lower().startswith("run style figure"):
        raw = "\n".join(lines[1:])
    df = _read_table_guess(raw)
    if "Horse" not in df.columns:
        raise ValueError("Box 1 requires a 'Horse' column.")
    df["Horse"] = df["Horse"].astype(str).str.strip()

    lto5 = [c for c in [f"Lto{i}" for i in range(1, 6)] if c in df.columns]
    lto10 = [c for c in [f"Lto{i}" for i in range(1, 11)] if c in df.columns]
    for c in lto10:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["RS_Avg"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto5]), axis=1)
    df["RS_Avg10"] = df.apply(lambda r: _mean_ignore_zero([r[c] for c in lto10]), axis=1)
    df["RS_Cat"] = df["RS_Avg"].apply(_rs_category)

    if "Dr.%" in df.columns:
        df["Dr.%"] = (
            df["Dr.%"].astype(str).str.strip().str.replace("%", "", regex=False).replace({"": np.nan}).astype(float)
        )

    keep = (
        ["Horse"]
        + lto10
        + ["RS_Avg", "RS_Avg10", "RS_Cat"]
        + [c for c in ["Mode 5", "Mode 10", "Total", "Mode All", "Draw", "Dr.%"] if c in df.columns]
    )
    return df[keep]


# --------------------------------------
# Box 2 — Official Ratings (block format with trailing 10 lines per horse)
# --------------------------------------

def _parse_or_rating_line(s: str) -> Optional[int]:
    """Extract rating from strings like '62/6', '58/U', '0/10'. Returns int (can be 0) or None."""
    s = str(s).strip()
    if not s:
        return None
    # Get part before '/'
    left = s.split("/")[0].strip()
    try:
        return int(left)
    except Exception:
        # if left not int, try to find first int in string
        m = re.search(r"-?\d+", left)
        return int(m.group(0)) if m else None


def parse_box2_official_ratings(raw: str) -> pd.DataFrame:
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    if lines and lines[0].lower().startswith("official ratings"):
        lines = lines[1:]

    # We'll treat a 'horse header line' as one with at least 5 tab-separated fields
    rows = []
    i = 0
    while i < len(lines):
        parts = lines[i].split("\t")
        # If header row, skip
        if i == 0 and parts and parts[0].lower().startswith("horse"):
            i += 1
            continue
        if len(parts) >= 5:
            # horse header
            horse = parts[0].strip()
            to_last = parts[1].strip()
            today = parts[2].strip()
            last = parts[3].strip()
            highest = parts[4].strip()
            i += 1
            # collect up to next 10 lines of Lto ratings (until next header-like line)
            ratings: List[int] = []
            for j in range(i, min(i + 20, len(lines))):
                if len(lines[j].split("\t")) >= 5:  # next horse header
                    break
                # parse rating/finish cell
                r = _parse_or_rating_line(lines[j])
                if r is not None:
                    ratings.append(r)
            # advance i to j
            i = j if 'j' in locals() else i + 1

            today_or = _safe_int(today)
            # Build outputs
            max10 = max(ratings) if ratings else None
            or_3 = ratings[2] if len(ratings) >= 3 else None
            re_lb = (max10 - today_or) if (max10 is not None and today_or is not None) else None
            dtrend = (today_or - or_3) if (today_or is not None and or_3 is not None) else None

            # TS scoring
            ts = None
            if re_lb is not None:
                if re_lb >= 8:
                    ts = 5
                elif re_lb >= 4:
                    ts = 4
                elif re_lb >= 0:
                    ts = 3
                elif re_lb >= -4:
                    ts = 2
                else:
                    ts = 1
                # trend nudge
                if dtrend is not None:
                    if dtrend >= 2:
                        ts = min(5, ts + 0.5)
                    elif dtrend <= -2:
                        ts = max(1, ts - 0.5)

            rows.append(
                dict(
                    Horse=horse,
                    today_or=today_or,
                    or_3back=or_3,
                    max_or_10=max10,
                    re_lb=re_lb,
                    d_or_trend=dtrend,
                    ts_5=ts,
                )
            )
        else:
            i += 1

    return pd.DataFrame(rows)


# --------------------------------------
# Box 3 — Race Class / Prize Money
# --------------------------------------

def parse_box3_prize(raw: str) -> Tuple[pd.DataFrame, Optional[float]]:
    lines = raw.strip().splitlines()
    today_class_value = None
    if lines and lines[0].lower().startswith("race class - today"):
        # Try to read '£xK' from header
        m = re.search(r"£\s*([0-9]*\.?[0-9]+)\s*([KkMm])?", lines[0])
        if m:
            today_class_value = _to_float_money(m.group(0))
        # drop header
        raw = "\n".join(lines[1:])

    df = _read_table_guess(raw)
    if "Horse" not in df.columns:
        raise ValueError("Box 3 requires a 'Horse' column.")
    df["Horse"] = df["Horse"].astype(str).str.strip()

    # Parse Avg 3 / Avg 5 monetary values
    def parse_money_series(x):
        return _to_float_money(x)

    if "Avg 3" in df.columns:
        df["avg3_prize"] = df["Avg 3"].apply(parse_money_series)
    else:
        df["avg3_prize"] = np.nan

    if "Avg 5" in df.columns:
        df["avg5_prize"] = df["Avg 5"].apply(parse_money_series)
    else:
        df["avg5_prize"] = np.nan

    # today_class_value per-row
    df["today_class_value"] = today_class_value

    # Class Differential and CCS
    def ccs_from_cd(cd: Optional[float]) -> Optional[float]:
        if cd is None or np.isnan(cd):
            return None
        if cd >= 0.4:
            return 5
        if cd >= 0.15:
            return 4
        if cd >= -0.15:
            return 3
        if cd >= -0.4:
            return 2
        return 1

    df["cd"] = (df["avg3_prize"] / df["today_class_value"]) - 1 if today_class_value else np.nan
    df["ccs_5"] = df["cd"].apply(ccs_from_cd)

    keep = ["Horse", "avg3_prize", "avg5_prize", "today_class_value", "cd", "ccs_5"]
    return df[keep], today_class_value


# --------------------------------------
# Box 4 — Speed Figures (11-line block per horse)
# --------------------------------------

def _extract_speed_series(line: str) -> List[int]:
    # Take numbers before the parenthesis average, if present
    pre = line.split("(")[0]
    nums = re.findall(r"\d+", pre)
    return [int(n) for n in nums]


def parse_box4_speed_blocks(raw: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines and lines[0].lower().startswith("horse"):
        lines = lines[1:]

    rows = []
    for i in range(0, len(lines), 11):
        block = lines[i : i + 11]
        if not block:
            continue
        horse = block[0].strip()
        if not horse:
            continue
        # speed line is 3rd index (0-based 3): "Speed Figures (Avg)"
        if len(block) < 4:
            rows.append(dict(Horse=horse))
            continue
        series = _extract_speed_series(block[3])
        if not series:
            rows.append(dict(Horse=horse))
            continue
        last = series[-1]
        high = max(series)
        avg3 = round(sum(series[-3:]) / min(3, len(series)), 1)
        avgall = round(sum(series) / len(series), 1)
        key = round((last + high + avg3) / 3, 1)
        rows.append(
            dict(
                Horse=horse,
                last_sf=last,
                high_sf=high,
                avg3_sf=avg3,
                avgall_sf=avgall,
                key_speed_avg=key,
            )
        )

    return pd.DataFrame(rows)


# --------------------------------------
# Box 5 — Race Conditions (from same block format as Box 4)
# --------------------------------------
CONDITION_ORDER = [
    "Crs %",
    "Dist %",
    "LHRH %",
    "Going %",
    "Class %",
    "Run. +/-1 %",
    "TrackStyle %",
]


def _parse_wpt(cell: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse strings like '7 (1/1/13)' -> (1,1,13). If missing, return (None,None,None)."""
    s = str(cell).strip()
    m = re.search(r"\((\d+)\/(\d+)\/(\d+)\)", s)
    if not m:
        return None, None, None
    w = int(m.group(1)); p = int(m.group(2)); t = int(m.group(3))
    return w, p, t


def _place_pct(w: Optional[int], p: Optional[int], t: Optional[int]) -> Optional[float]:
    if None in (w, p, t) or t == 0:
        return None
    return round(100.0 * (w + p) / t, 1)


def _fit_from_place_pct(x: Optional[float]) -> Optional[int]:
    if x is None:
        return None
    if x >= 50:
        return 5
    if x >= 40:
        return 4
    if x >= 30:
        return 3
    if x >= 20:
        return 2
    return 1


def parse_box5_conditions_from_blocks(raw: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines and lines[0].lower().startswith("horse"):
        lines = lines[1:]

    rows = []
    for i in range(0, len(lines), 11):
        block = lines[i : i + 11]
        if not block:
            continue
        horse = block[0].strip()
        if not horse:
            continue
        # After the speed line (index 3), the next 7 lines align with CONDITION_ORDER
        conds = block[4:11]
        data = {"Horse": horse}
        for label, cell in zip(CONDITION_ORDER, conds):
            w, p, t = _parse_wpt(cell)
            place = _place_pct(w, p, t)
            fit = _fit_from_place_pct(place)
            slug = label.lower().replace(" ", "_").replace("+/-", "pm1").replace("%", "pct")
            data[f"{slug}_runs"] = t
            data[f"{slug}_place_pct"] = place
            data[f"{slug}_fit"] = fit
        rows.append(data)

    return pd.DataFrame(rows)


# --------------------------------------
# UI — Five Boxes
# --------------------------------------
box1, box2 = st.columns(2)
with box1:
    st.subheader("Box 1 — Pace Run Style")
    b1_text = st.text_area("Paste Box 1 (Run Style Figure) block", height=240)
    if st.button("Process Box 1", type="primary"):
        try:
            df1 = parse_box1_run_style(b1_text)
            st.success("Box 1 parsed ✔")
            st.dataframe(df1, use_container_width=True)
            st.session_state.box1 = df1
            st.download_button("💾 Download Box 1 CSV", df1.to_csv(index=False), "box1_run_style.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Box 1 parsing failed: {e}")

with box2:
    st.subheader("Box 2 — Official Ratings")
    b2_text = st.text_area("Paste Box 2 (Official Ratings) block", height=240)
    if st.button("Process Box 2"):
        try:
            df2 = parse_box2_official_ratings(b2_text)
            st.success("Box 2 parsed ✔")
            st.dataframe(df2, use_container_width=True)
            st.session_state.box2 = df2
            st.download_button("💾 Download Box 2 CSV", df2.to_csv(index=False), "box2_official_ratings.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Box 2 parsing failed: {e}")

box3, box4 = st.columns(2)
with box3:
    st.subheader("Box 3 — Race Class / Prize Money")
    b3_text = st.text_area("Paste Box 3 (Race Class) block", height=260)
    if st.button("Process Box 3"):
        try:
            df3, today_cls_val = parse_box3_prize(b3_text)
            st.success(f"Box 3 parsed ✔  (Today Class £: {today_cls_val if today_cls_val else 'n/a'})")
            st.dataframe(df3, use_container_width=True)
            st.session_state.box3 = df3
            st.session_state.today_class_value = today_cls_val
            st.download_button("💾 Download Box 3 CSV", df3.to_csv(index=False), "box3_prize.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Box 3 parsing failed: {e}")

with box4:
    st.subheader("Box 4 — Adjusted Speed (anchors)")
    b4_text = st.text_area("Paste Box 4 (Speed Blocks) block", height=260)
    if st.button("Process Box 4"):
        try:
            df4 = parse_box4_speed_blocks(b4_text)
            st.success("Box 4 parsed ✔")
            st.dataframe(df4, use_container_width=True)
            st.session_state.box4 = df4
            st.download_button("💾 Download Box 4 CSV", df4.to_csv(index=False), "box4_speed.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Box 4 parsing failed: {e}")

st.subheader("Box 5 — Race Conditions (from same block as Box 4; speed ignored)")
b5_text = st.text_area("Paste Box 5 (Conditions) block", height=220)
if st.button("Process Box 5"):
    try:
        df5 = parse_box5_conditions_from_blocks(b5_text)
        st.success("Box 5 parsed ✔")
        st.dataframe(df5, use_container_width=True)
        st.session_state.box5 = df5
        st.download_button("💾 Download Box 5 CSV", df5.to_csv(index=False), "box5_conditions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Box 5 parsing failed: {e}")

st.markdown("---")

# --------------------------------------
# Combine & Export
# --------------------------------------

st.subheader("Combine & Export")
colz = st.columns(4)
with colz[0]:
    today_field_size = st.number_input("Today Field Size", value=11, min_value=2, max_value=24)
with colz[1]:
    class_par = st.number_input("Class Par", value=81.0, step=0.5)
with colz[2]:
    front_thr = st.number_input("Front <", value=1.6, step=0.1)
with colz[3]:
    prom_thr = st.number_input("Prominent <", value=2.4, step=0.1)
mid_thr = st.number_input("Mid <", value=3.0, step=0.1)

if st.button("🧩 Combine All Boxes"):
    try:
        if not all(k in st.session_state for k in ("box1", "box2", "box3", "box4", "box5")):
            st.warning("Please process all five boxes first.")
            st.stop()
        b1 = st.session_state.box1.copy()
        b2 = st.session_state.box2.copy()
        b3 = st.session_state.box3.copy()
        b4 = st.session_state.box4.copy()
        b5 = st.session_state.box5.copy()

        # Normalise keys
        for df in (b1, b2, b3, b4, b5):
            df["Horse"] = df["Horse"].astype(str).str.strip()

        merged = b1
        for d in (b2, b3, b4, b5):
            merged = pd.merge(merged, d, on="Horse", how="outer")

        # Ensure RS_Cat uses current thresholds on RS_Avg
        merged["RS_Cat"] = merged["RS_Avg"].apply(lambda x: _rs_category(x, front_thr, prom_thr, mid_thr))

        # Context columns
        merged["today_field_size"] = today_field_size
        merged["class_par"] = class_par

        # ΔvsPar (for model)
        if "key_speed_avg" in merged.columns:
            merged["d_vs_par"] = merged["key_speed_avg"] - merged["class_par"]

        # Minimal model-ready column order (keep others too)
        preferred = [
            "Horse",
            # Pace
            "RS_Avg", "RS_Avg10", "RS_Cat", "Lto1", "Lto2", "Lto3", "Lto4", "Lto5",
            # Speed
            "key_speed_avg", "last_sf", "high_sf", "avg3_sf",
            # Class/Prize
            "avg3_prize", "avg5_prize", "today_class_value", "cd", "ccs_5",
            # Official Ratings
            "today_or", "or_3back", "max_or_10", "re_lb", "d_or_trend", "ts_5",
            # Conditions (key CRI facets)
            "run._pm1_pct_runs", "run._pm1_pct_place_pct", "run._pm1_pct_fit",  # may not exist depending on slug
            "dist_%_runs", "dist_%_place_pct", "dist_%_fit",
            "class_%_runs", "class_%_place_pct", "class_%_fit",
            "trackstyle_%_runs", "trackstyle_%_place_pct", "trackstyle_%_fit",
            # Context
            "today_field_size", "class_par", "d_vs_par",
        ]

        # Show and export
        st.success("Combined ✔  (scroll right for all derived columns)")
        st.dataframe(merged, use_container_width=True)
        st.download_button("💾 Download Combined CSV", merged.to_csv(index=False), "cleanpace_v2_combined.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Combine failed: {e}")

st.caption("RS_Avg = last 5 (ignoring zeros) and drives RS_Cat; RS_Avg10 is for diagnostics. Box 5 builds CRI-style fits from W/P/T.")
