# cleanpace_both_v2.1.py
import streamlit as st
import pandas as pd
import re, math
from io import StringIO
from typing import Optional, Tuple, List

st.set_page_config(page_title="CleanPace — Both Inputs (v2.1)", page_icon="🐎", layout="wide")
st.title("🐎 CleanPace v2.1 — BOTH Inputs (Pace/OR/Prize + Speed + Conditions W/P/T)")

# ---------------- Sidebar defaults ----------------
with st.sidebar:
    st.header("⚙️ Defaults (used if not found in text)")
    default_class_value = st.number_input("Fallback: Today's Class Value (£)", value=6000.0, step=250.0)
    default_field_size  = st.number_input("Today's Field Size", value=14, step=1)
    default_class_par   = st.number_input("Class Par (ref only)", value=76.0, step=1.0)
    st.caption("Field size is also used to weight FieldFit inside CRI.")

# ---------------- Helpers ----------------
def _read_table_guess(raw: str) -> pd.DataFrame:
    try:    return pd.read_csv(StringIO(raw), sep="\t")
    except: return pd.read_csv(StringIO(raw))

def _num(x):
    if x is None: return None
    try:
        return float(str(x).replace("+","").replace(",","").strip())
    except Exception:
        return None

def parse_money_k(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    m = re.search(r'£?\s*([\d.]+)\s*K', s, flags=re.I)
    if m: return float(m.group(1))*1000.0
    m2 = re.search(r'£?\s*([\d.,]+)', s)
    if m2: return float(m2.group(1).replace(",",""))
    return None

def rs_category(avg: Optional[float]) -> Optional[str]:
    if avg is None or (isinstance(avg,float) and math.isnan(avg)): return None
    if avg < 1.6: return "Front"
    if avg < 2.4: return "Prominent"
    if avg < 3.0: return "Mid"
    return "Hold-up"

def ts_from_re(re_lb: float) -> float:
    if re_lb >= 8: return 5.0
    if re_lb >= 4: return 4.0
    if re_lb >= -3: return 3.0
    if re_lb >= -7: return 2.0
    return 1.0

def ccs_from_cd(cd: float) -> int:
    if cd >= 1.5: return 5
    if cd >= 0.5: return 4
    if cd >= -0.4: return 3
    if cd >= -0.8: return 2
    return 1

# ---- Box B: speed blocks ----
def extract_from_speed_block(block_lines: List[str]):
    name = block_lines[0].strip()
    if len(block_lines) < 4:
        return name, None, None, None, None, None
    speed_line = block_lines[3].strip()
    speed_str = speed_line.split('(')[0].strip()
    try:
        figs = list(map(int, re.findall(r'\d+', speed_str)))
        if not figs: return name, None, None, None, None, None
        last = figs[-1]
        highest = max(figs)
        avg3 = round(sum(figs[-3:]) / min(3, len(figs)), 1)
        avga = round(sum(figs) / len(figs), 1)
        key = round((last + highest + avg3) / 3, 1)  # your Adj Speed input
        return name, last, highest, avg3, avga, key
    except Exception:
        return name, None, None, None, None, None

def parse_speed_blocks(raw: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if lines and "Horse" in lines[0]:
        lines = lines[1:]
    rows = []
    for i in range(0, len(lines), 11):
        block = lines[i:i+11]
        if not block: continue
        rows.append(extract_from_speed_block(block))
    return pd.DataFrame(rows, columns=[
        "Horse","Last Race Speed Figure","Highest Speed Figure","Avg of Last 3","Avg of All","Key Speed Factors Average"
    ])

# ---- Box A split into 3 sections ----
def split_sections(raw: str):
    text = raw.strip().replace("\r\n","\n").replace("\r","\n")
    rs_idx = text.find("Run Style Figure")
    or_idx = text.find("Official Ratings")
    rc_idx = text.find("Race Class - Today")
    if rs_idx == -1 or or_idx == -1 or rc_idx == -1:
        return None, None, None, None
    rs_text = text[rs_idx:or_idx].strip()
    or_text = text[or_idx:rc_idx].strip()
    prize_text = text[rc_idx:].strip()
    m = re.search(r'Race Class\s*-\s*Today:\s*Class\s*\d+\s*£\s*([\d.]+)\s*K', prize_text, flags=re.I)
    today_class_val = float(m.group(1))*1000.0 if m else None
    return rs_text, or_text, prize_text, today_class_val

def parse_run_style_table_from_section(rs_text: str) -> pd.DataFrame:
    lines = rs_text.splitlines()
    table = "\n".join(lines[1:]) if lines and lines[0].strip().lower().startswith("run style figure") else rs_text
    df = _read_table_guess(table)
    # numeric Lto columns
    lto_cols = [c for c in df.columns if re.fullmatch(r"Lto\d+", str(c))]
    for c in lto_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    def rs_avg_row(r):
        vals = [r[c] for c in lto_cols if pd.notna(r[c]) and float(r[c])>0]
        return round(sum(vals)/len(vals), 2) if vals else None
    df["RS_Avg"] = df.apply(rs_avg_row, axis=1)
    df["RS_Cat"] = df["RS_Avg"].apply(rs_category)
    keep = ["Horse"] + lto_cols + ["RS_Avg","RS_Cat","Mode 5","Mode 10","Total","Mode All","Draw","Dr.%"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]

def parse_official_ratings(or_text: str) -> pd.DataFrame:
    lines = [ln for ln in or_text.splitlines() if ln.strip()]
    if lines and lines[0].strip().lower().startswith("official ratings"):
        lines = lines[1:]
    data = []
    i = 0; header_seen = False
    while i < len(lines):
        ln = lines[i].strip()
        if not header_seen and ln.lower().startswith("horse"):
            header_seen = True; i += 1; continue
        if not re.match(r'^\d+\s*/\s*\d+$', ln):
            parts = re.split(r'\t+', ln)
            if len(parts) == 1:
                parts = re.split(r'\s{2,}', ln)
            horse   = parts[0].strip()
            tolast  = parts[1].strip() if len(parts)>1 else ""
            today   = parts[2].strip() if len(parts)>2 else ""
            last    = parts[3].strip() if len(parts)>3 else ""
            highest = parts[4].strip() if len(parts)>4 else ""
            j = i+1; lto_list=[]
            while j < len(lines) and re.match(r'^\d+\s*/\s*\d+$', lines[j].strip()):
                lto_list.append(lines[j].strip()); j += 1
            data.append((horse, tolast, today, last, highest, lto_list))
            i = j
        else:
            i += 1
    rows=[]
    for horse, tolast, today, last, highest, ltos in data:
        today_or = _num(today)
        highest_or = _num(highest)
        or_nums=[]
        for cell in ltos:
            m = re.match(r'^\s*(\d+)\s*/', cell)
            if m: or_nums.append(int(m.group(1)))
        max_or10 = max(or_nums) if or_nums else (highest_or if highest_or is not None else None)
        or_3back = or_nums[2] if len(or_nums)>=3 else (or_nums[-1] if or_nums else None)
        re_lb = (max_or10 - today_or) if (max_or10 is not None and today_or is not None) else None
        dtrend = (today_or - or_3back) if (today_or is not None and or_3back is not None) else None
        ts = ts_from_re(re_lb) if re_lb is not None else None
        if dtrend is not None:
            if dtrend <= -4: ts = min(5.0, ts + 0.5)
            elif dtrend >= +4: ts = max(1.0, ts - 0.5)
        rows.append({"Horse":horse,"Today OR":today_or,"OR 3back":or_3back,"Max OR 10":max_or10,
                     "RE (lb)":re_lb,"ΔOR trend":dtrend,"TS (1-5)":ts})
    return pd.DataFrame(rows)

def parse_prize_block(prize_text: str, fallback_class_value: float):
    m = re.search(r'Race Class\s*-\s*Today:\s*Class\s*\d+\s*£\s*([\d.]+)\s*K', prize_text, flags=re.I)
    today_class_value = float(m.group(1))*1000.0 if m else fallback_class_value
    lines = prize_text.splitlines()
    start_idx=None
    for idx, ln in enumerate(lines):
        if ln.strip()=="Horse": start_idx=idx; break
    if start_idx is None:
        for idx, ln in enumerate(lines):
            if ln.strip().startswith("Horse\t"): start_idx=idx; break
    if start_idx is None:
        return pd.DataFrame(columns=["Horse","Avg 3 Prize","Avg 5 Prize","Today Class (£)"]), today_class_value
    sub=[ln for ln in lines[start_idx+1:] if ln.strip()]

    def is_horse_header(ln:str)->bool:
        s=ln.strip()
        if not s: return False
        if "£" in s or "Cl" in s or "/" in s: return False
        return True

    rows=[]; i=0
    while i < len(sub):
        if is_horse_header(sub[i]):
            horse=sub[i].strip()
            j=i+1; monies=[]
            while j < len(sub) and not is_horse_header(sub[j]):
                for mline in re.findall(r'£\s*[\d.]+\s*K', sub[j], flags=re.I):
                    monies.append(mline)
                j+=1
            avg3 = parse_money_k(monies[-2]) if len(monies) >= 2 else None
            avg5 = parse_money_k(monies[-1]) if len(monies) >= 1 else None
            rows.append({"Horse":horse,"Avg 3 Prize":avg3,"Avg 5 Prize":avg5})
            i=j
        else:
            i+=1
    df=pd.DataFrame(rows)
    if not df.empty:
        df["CD"] = (df["Avg 3 Prize"]/today_class_value) - 1.0
        df["CCS (1-5)"] = df["CD"].apply(lambda x: ccs_from_cd(x) if pd.notna(x) else None)
        df["Today Class (£)"] = today_class_value
    return df, today_class_value

# ---- Conditions W/P/T (Box B extra) ----
WPT_RE = re.compile(r'^\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$')
def parse_wpt(cell: Optional[str])->Tuple[int,int,int]:
    if cell is None or (isinstance(cell,float) and math.isnan(cell)): return 0,0,0
    s=str(cell).strip()
    m=WPT_RE.match(s)
    if not m:
        trip = re.findall(r'(\d+/\d+/\d+)', s)
        if trip: m=WPT_RE.match(trip[-1])
        if not m: return 0,0,0
    w,p,t = map(int, m.groups()); return w,p,t

def pct(n,d): return (100.0*n/d) if d else 0.0

def place_score(place_pct: float)->int:
    if place_pct >= 80: return 5
    if place_pct >= 60: return 4
    if place_pct >= 40: return 3
    if place_pct >= 20: return 2
    return 1

def field_weight(fs:int)->float:
    if fs <= 8: return 0.2
    if fs <=12: return 0.3
    return 0.4

def parse_conditions_wpt_table(raw: str, today_field_size:int) -> pd.DataFrame:
    """
    Expect a table with columns:
      Horse, Win %, Form Figures (Avg), Speed Figures (Avg),
      Crs %, Dist %, LHRH %, Going %, Class %, Run. +/-1 %, TrackStyle %
    We parse only the W/P/T columns -> place%, fit (1–5) and build CRI (field emphasis).
    """
    df = _read_table_guess(raw)
    # rename fuzzy
    ren={}
    for c in df.columns:
        cl=c.lower().strip()
        if cl in ["horse","name","horse name"]: ren[c]="Horse"
        elif cl.startswith("crs"): ren[c]="Crs %"
        elif cl.startswith("dist"): ren[c]="Dist %"
        elif cl.startswith("lhrh"): ren[c]="LHRH %"
        elif cl.startswith("class"): ren[c]="Class %"
        elif cl.startswith("run") and "+/-1" in cl: ren[c]="Run. +/-1 %"
        elif cl.startswith("trackstyle"): ren[c]="TrackStyle %"
        elif cl.startswith("going"): ren[c]="Going %"
        elif cl.startswith("win %"): ren[c]="Win %"
        elif "form figures" in cl: ren[c]="Form Figures (Avg)"
        elif "speed figures" in cl: ren[c]="Speed Figures (Avg)"
    df=df.rename(columns=ren)

    out=df.copy()
    # Ensure expected cols exist
    for col in ["Crs %","Dist %","LHRH %","Class %","Run. +/-1 %","TrackStyle %"]:
        if col not in out.columns: out[col]=None

    def add_wpt(prefix, series):
        w_col=f"{prefix}_wins"; p_col=f"{prefix}_places"; t_col=f"{prefix}_runs"
        wpct=f"{prefix}_win_pct"; ppct=f"{prefix}_place_pct"; fit=f"{prefix}_fit"
        wpt = series.apply(parse_wpt)
        out[[w_col,p_col,t_col]] = pd.DataFrame(wpt.tolist(), index=out.index)
        out[wpct] = out.apply(lambda r: pct(r[w_col], r[t_col]), axis=1)
        out[ppct]= out.apply(lambda r: pct(r[w_col]+r[p_col], r[t_col]), axis=1)
        out[fit] = out[ppct].apply(place_score)

    add_wpt("course", out["Crs %"])
    add_wpt("dist", out["Dist %"])
    add_wpt("lhrh", out["LHRH %"])
    add_wpt("class", out["Class %"])
    add_wpt("field", out["Run. +/-1 %"])
    add_wpt("trackstyle", out["TrackStyle %"])

    # CRI with field-size emphasis
    fw = field_weight(int(today_field_size))
    remain = 1.0 - fw
    # you can include 'course_fit' and 'lhrh_fit' if you like; here we keep signal tight
    components = ["dist_fit","class_fit","trackstyle_fit"]
    ow = remain/len(components) if components else 0.0

    def cri_row(r):
        parts=[]
        if pd.notna(r.get("field_fit")): parts.append(("field", float(r["field_fit"])))
        for c in components:
            v=r.get(c)
            if pd.notna(v): parts.append(("other", float(v)))
        if not parts: return None
        num=0.0; den=0.0
        for name,val in parts:
            w = fw if name=="field" else ow
            num += w*val; den += w
        return num/den if den>0 else None

    out["CRI (1-5)"] = out.apply(cri_row, axis=1)
    # Keep the engineered columns plus Horse
    keep = ["Horse",
            "dist_runs","dist_place_pct","dist_fit",
            "class_runs","class_place_pct","class_fit",
            "field_runs","field_place_pct","field_fit",
            "trackstyle_runs","trackstyle_place_pct","trackstyle_fit",
            "course_runs","course_place_pct","course_fit",
            "lhrh_runs","lhrh_place_pct","lhrh_fit",
            "CRI (1-5)"]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

# ---------------- UI ----------------
left, right = st.columns(2)

with left:
    st.subheader("📄 Box A — PAGE 1 paste")
    st.caption("Paste one block containing, in order:\n1) Run Style Figure table\n2) Official Ratings block\n3) Race Class / Prize block")
    box_a = st.text_area("Paste Page 1 block:", height=460, key="box_a")

with right:
    st.subheader("⚡ Box B — PAGE 2 data")
    st.caption("Paste your **Speed Blocks** as usual, and optionally paste/upload the **Conditions (W/P/T) table**.")
    speed_blocks = st.text_area("Speed Blocks (11 lines per horse):", height=220, key="box_b_speed")
    cond_source = st.radio("Conditions input:", ["Paste table", "Upload CSV/TSV", "Skip"], horizontal=True)
    cond_text = ""
    cond_file = None
    if cond_source == "Paste table":
        cond_text = st.text_area("Conditions W/P/T table (same format as example):", height=200, key="box_b_cond_text")
    elif cond_source == "Upload CSV/TSV":
        cond_file = st.file_uploader("Upload Conditions CSV/TSV", type=["csv","tsv"], key="box_b_cond_file")

# ---------------- Action ----------------
if st.button("🚀 Process Both"):
    if not box_a.strip():
        st.warning("Please paste the Page 1 block in Box A.")
    elif not speed_blocks.strip():
        st.warning("Please paste the Speed Blocks in Box B.")
    else:
        try:
            # ---- Parse Box A (3 sections)
            rs_text, or_text, prize_text, cls_from_text = split_sections(box_a)
            if not (rs_text and or_text and prize_text):
                st.error("Could not find all three sections in Box A. Ensure it includes 'Run Style Figure', 'Official Ratings', and 'Race Class - Today:' blocks.")
            else:
                # 1) Run Style
                rs_df = parse_run_style_table_from_section(rs_text)
                st.success("Run Style parsed")
                st.dataframe(rs_df, use_container_width=True)

                # 2) Official Ratings
                or_df = parse_official_ratings(or_text)
                st.success("Official Ratings parsed")
                st.dataframe(or_df, use_container_width=True)

                # 3) Prize/Class
                prize_df, cls_val = parse_prize_block(prize_text, fallback_class_value=default_class_value)
                today_class_val = cls_val or default_class_value
                st.success(f"Prize/Class parsed (Today Class ≈ £{int(today_class_val):,})")
                st.dataframe(prize_df, use_container_width=True)

                # Merge A side
                for d in (rs_df, or_df, prize_df):
                    d["Horse"] = d["Horse"].astype(str).str.strip()
                merged_a = rs_df.merge(or_df, on="Horse", how="outer").merge(prize_df, on="Horse", how="outer")

                # ---- Parse Box B (speed + conditions)
                sp_df = parse_speed_blocks(speed_blocks)
                sp_df["Horse"] = sp_df["Horse"].astype(str).str.strip()
                st.success("Speed Blocks parsed")
                st.dataframe(sp_df, use_container_width=True)

                cond_df = None
                if cond_source == "Paste table" and cond_text.strip():
                    cond_df = parse_conditions_wpt_table(cond_text, today_field_size=default_field_size)
                elif cond_source == "Upload CSV/TSV" and cond_file is not None:
                    raw = cond_file.read().decode("utf-8", errors="ignore")
                    try: df_cond = pd.read_csv(StringIO(raw), sep="\t")
                    except: df_cond = pd.read_csv(StringIO(raw))
                    cond_df = parse_conditions_wpt_table(df_cond.to_csv(index=False), today_field_size=default_field_size) \
                              if not isinstance(df_cond, pd.DataFrame) else parse_conditions_wpt_table(raw, today_field_size=default_field_size)

                if cond_df is not None:
                    cond_df["Horse"] = cond_df["Horse"].astype(str).str.strip()
                    st.success("Conditions W/P/T parsed → CRI computed")
                    st.dataframe(cond_df, use_container_width=True)

                # ---- Combine everything
                combined = sp_df.merge(merged_a, on="Horse", how="outer")
                if cond_df is not None:
                    combined = combined.merge(cond_df, on="Horse", how="outer")

                combined["Today Class (£)"] = combined.get("Today Class (£)", today_class_val)
                combined["Today Field Size"] = default_field_size
                combined["Class Par"] = default_class_par

                # Backfill CD/CCS if missing
                if "CD" not in combined.columns and "Avg 3 Prize" in combined.columns:
                    combined["CD"] = (combined["Avg 3 Prize"] / combined["Today Class (£)"]) - 1.0
                if "CCS (1-5)" not in combined.columns and "CD" in combined.columns:
                    combined["CCS (1-5)"] = combined["CD"].apply(lambda x: ccs_from_cd(x) if pd.notna(x) else None)

                # Order columns for readability
                preferred = [
                    "Horse",
                    # Speed
                    "Key Speed Factors Average","Last Race Speed Figure","Highest Speed Figure","Avg of Last 3","Avg of All",
                    # Run Style
                    "RS_Avg","RS_Cat","Mode 5","Mode 10","Total","Mode All","Draw","Dr.%"
                ] + [c for c in combined.columns if re.fullmatch(r"Lto\d+", str(c))] + [
                    # OR
                    "Today OR","OR 3back","Max OR 10","RE (lb)","ΔOR trend","TS (1-5)",
                    # Prize
                    "Avg 3 Prize","Avg 5 Prize","CD","CCS (1-5)","Today Class (£)",
                    # Conditions Fit
                    "dist_runs","dist_place_pct","dist_fit",
                    "class_runs","class_place_pct","class_fit",
                    "field_runs","field_place_pct","field_fit",
                    "trackstyle_runs","trackstyle_place_pct","trackstyle_fit",
                    "course_runs","course_place_pct","course_fit",
                    "lhrh_runs","lhrh_place_pct","lhrh_fit",
                    "CRI (1-5)",
                    # Context
                    "Today Field Size","Class Par"
                ]
                cols = [c for c in preferred if c in combined.columns] + [c for c in combined.columns if c not in preferred]
                combined = combined[cols]

                st.markdown("### ✅ Combined Output")
                st.dataframe(combined, use_container_width=True)
                st.download_button("💾 Download Combined CSV",
                                   combined.to_csv(index=False),
                                   file_name="cleanpace_both_v2_1_combined.csv",
                                   mime="text/csv",
                                   use_container_width=True)

        except Exception as e:
            st.error(f"Processing error: {e}")

st.markdown("---")
st.caption("CleanPace — Both Inputs v2.1 • Box A: Run Style + OR + Prize • Box B: Speed Blocks + Conditions W/P/T • Outputs one modeling-ready CSV")
