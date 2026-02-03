#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, time, json
from glob import glob
from typing import List, Dict

import numpy as np
import pandas as pd

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_list(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]

def as_bool(s) -> bool:
    return str(s).strip().lower() in ("yes","y","true","1","on")

# ---------- 标签解析：直接沿用你原来的 flexible 逻辑 ----------
def _first_existing(df: pd.DataFrame, names: List[str]):
    for n in names:
        if n in df.columns:
            return n
    return None

def read_labels_flexible(label_csv: str, domain_col: str|None, label_col: str|None) -> pd.DataFrame:
    df = pd.read_csv(label_csv)
    df.columns = [c.strip() for c in df.columns]

    # Domain
    if domain_col:
        if domain_col not in df.columns:
            raise ValueError(f"--domain-col={domain_col} not in columns: {df.columns.tolist()}")
        df["Domain"] = df[domain_col].astype(str)
    else:
        dn = _first_existing(df, ["Domain","PDBID","pdbid","domain"])
        dn = dn if dn is not None else df.columns[0]
        df["Domain"] = df[dn].astype(str)

    # L1/L2/L3
    C_col = _first_existing(df, ["C","L1","Class","class"])
    A_col = _first_existing(df, ["A","L2","Architecture","architecture"])
    T_col = _first_existing(df, ["T","L3","Topology","topology"])

    if C_col is not None:
        C = df[C_col].astype(str)
        A = df[A_col].astype(str) if A_col is not None else None
        T = df[T_col].astype(str) if T_col is not None else None
        df["L1"] = C
        df["L2"] = (C + "_" + A).astype(str) if A is not None else C
        if A is not None and T is not None:
            df["L3"] = (C + "_" + A + "_" + T).astype(str)
            df["Label"] = df["L3"]
        elif A is not None:
            df["L3"] = (C + "_" + A).astype(str)
            df["Label"] = df["L3"]
        else:
            df["L3"] = C
            df["Label"] = C
    else:
        if label_col and label_col != "auto":
            if label_col not in df.columns:
                raise ValueError(f"--label-col={label_col} not in columns: {df.columns.tolist()}")
            lab = df[label_col].astype(str)
        else:
            lab = df["Label"].astype(str) if "Label" in df.columns else df.iloc[:,1].astype(str)

        parts = lab.str.split("_", n=2, expand=True)
        while parts.shape[1] < 3:
            parts[parts.shape[1]] = ""
        C = parts[0].fillna("").astype(str)
        A = parts[1].fillna("").astype(str)
        T = parts[2].fillna("").astype(str)
        df["L1"] = C
        df["L2"] = (C + "_" + A).str.rstrip("_")
        df["L3"] = (C + "_" + A + "_" + T).str.rstrip("_")
        df["Label"] = df["L3"]

    out = df[["Domain","Label","L1","L2","L3"]].dropna().drop_duplicates().reset_index(drop=True)
    return out

# ---------- robust 统计 ----------
def robust_stats_1d(x: np.ndarray) -> Dict[str,float]:
    if x.size == 0 or not np.isfinite(x).any():
        return dict(mean=np.nan, std=np.nan, median=np.nan, mad=np.nan,
                    p10=np.nan, p25=np.nan, p75=np.nan, p90=np.nan,
                    min=np.nan, max=np.nan, length=0, energy=np.nan,
                    iqr=np.nan, tail_ratio=np.nan, z_median=np.nan, range_norm=np.nan)
    x = x[np.isfinite(x)].astype(np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    p10 = np.quantile(x, 0.10)
    p25 = np.quantile(x, 0.25)
    p75 = np.quantile(x, 0.75)
    p90 = np.quantile(x, 0.90)
    iqr = p75 - p25 + 1e-9
    tail_ratio = np.abs(p90) / (np.abs(p10) + 1e-9)
    z_med = (np.mean(x) - med) / mad
    range_norm = (np.max(x) - np.min(x)) / iqr
    return dict(mean=float(np.mean(x)),
                std=float(np.std(x, ddof=1)) if x.size>1 else 0.0,
                median=float(med), mad=float(mad-1e-9),
                p10=float(p10), p25=float(p25), p75=float(p75), p90=float(p90),
                min=float(np.min(x)), max=float(np.max(x)),
                length=int(x.size), energy=float(np.sum(x**2)),
                iqr=float(iqr-1e-9), tail_ratio=float(tail_ratio),
                z_median=float(z_med), range_norm=float(range_norm))

# ---------- base-name 展开 ----------
def expand_base_cols(df: pd.DataFrame, base_names: List[str]) -> List[str]:
    if not base_names:
        out = []
        for c in df.columns:
            if c.lower() == "residueindex":
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    out.append(c)
            except Exception:
                pass
        return out

    import re
    regexes = []
    for base in base_names:
        key = base.split("_k")[0]
        pat = rf"^{re.escape(key)}(?:_k\d+)?$"
        regexes.append((base, re.compile(pat)))

    selected = []
    seen = set()
    base_set = set(base_names)

    # 精确名优先
    for c in df.columns:
        if c in base_set and c not in seen:
            selected.append(c); seen.add(c)

    # *_k\d+ 扩展
    for c in df.columns:
        for base, rgx in regexes:
            if rgx.fullmatch(c):
                if c not in seen:
                    selected.append(c); seen.add(c)
                break
    return selected

def collect_union_columns(paths: List[str]) -> List[str]:
    seen=set(); union_cols=[]
    for p in paths:
        try:
            dfh = pd.read_csv(p, nrows=2)
        except Exception:
            continue
        for c in dfh.columns:
            if c not in seen:
                union_cols.append(c); seen.add(c)
    return union_cols

# ---------- 读 CY-like 残基层面 CSV ----------
def load_cylike_file(path: str, use_global_residue: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ResidueIndex" in df.columns and not use_global_residue:
        rid = pd.to_numeric(df["ResidueIndex"], errors="coerce")
        df = df[~(rid == -1)].copy()
    if "ResidueIndex" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing ResidueIndex")
    return df

def build_cylike_features_for_domain(csv_path: str,
                                     cols_numeric: List[str],
                                     use_global_residue: bool) -> Dict[str,float]:
    df = load_cylike_file(csv_path, use_global_residue=use_global_residue)

    # 只对 ResidueIndex >= 0 的残基层面做统计
    loc = df[df["ResidueIndex"]>=0].copy()
    if loc.empty:
        loc = df.copy()

    if cols_numeric and len(cols_numeric)>0:
        local_cols = expand_base_cols(loc, cols_numeric)
    else:
        local_cols = [c for c in loc.columns
                      if c.lower()!="residueindex"
                      and pd.api.types.is_numeric_dtype(loc[c])]

    feats={}
    for c in local_cols:
        x = pd.to_numeric(loc[c], errors="coerce").to_numpy()
        st = robust_stats_1d(x)
        for k,v in st.items():
            feats[f"all_{c}_{k}"] = v
    return feats

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate multi-scale CY-like residue features per domain.")
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--cylike_dir", required=True,
                    help="Dir with <Domain>.csv (CY-like residue features).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--domain-col", default="PDBID")
    ap.add_argument("--label-col",  default="auto")
    ap.add_argument("--cols_numeric",
                    default="integrability_err,kahler_err,omega_norm,logdetG,ricci_proxy,B_avg",
                    help="comma list of base names; expand to *_k<digits> 等。")
    ap.add_argument("--useGlobalResidue", default="yes",
                    help="yes/no: 是否保留 ResidueIndex=-1 的行参与统计（默认 no）。")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    use_global_flag = as_bool(args.useGlobalResidue)

    # 读标签
    labels = read_labels_flexible(args.label_csv, args.domain_col, args.label_col)
    labs = set(labels["Domain"])

    # 匹配 CY-like CSV：文件名去掉 .csv 后就是 Domain
    files_all = glob(os.path.join(args.cylike_dir, "*.csv"))
    dom2file={}
    for p in files_all:
        fn = os.path.basename(p)
        if fn.endswith(".csv"):
            dom = fn[:-4]
        else:
            continue
        if dom in labs:
            dom2file[dom]=p

    log(f"[CYLIKE] domains: labels={len(labels)} files={len(files_all)} intersection={len(dom2file)}")

    # 列名总览
    base_names = parse_list(args.cols_numeric)
    matched_paths=list(dom2file.values())
    union_cols = collect_union_columns(matched_paths)
    union_df = pd.DataFrame(columns=union_cols)
    selected_cols_global = expand_base_cols(union_df, base_names)

    try:
        with open(os.path.join(args.outdir,"original_columns_union.json"),"w",encoding="utf-8") as f:
            json.dump(union_cols,f,ensure_ascii=False,indent=2)
        with open(os.path.join(args.outdir,"selected_columns_expanded.json"),"w",encoding="utf-8") as f:
            json.dump(selected_cols_global,f,ensure_ascii=False,indent=2)
    except Exception:
        pass

    # 聚合
    rows=[]
    for i,(dom,lab,l1,l2,l3) in enumerate(labels[["Domain","Label","L1","L2","L3"]].itertuples(index=False),1):
        p = dom2file.get(dom)
        if not p:
            continue
        try:
            feats = build_cylike_features_for_domain(
                p, cols_numeric=base_names, use_global_residue=use_global_flag
            )
            if feats:
                feats["Domain"]=dom; feats["Label"]=lab; feats["L1"]=l1; feats["L2"]=l2; feats["L3"]=l3
                rows.append(feats)
        except Exception as e:
            log(f"[WARN] {dom}: {e}")
        if i % 4000 == 0:
            log(f"[CYLIKE] processed {i}/{len(labels)}")

    df = pd.DataFrame(rows)
    if df.empty:
        log("[ERR] No domains aggregated.")
        return

    # 去掉全空列
    df = df.dropna(axis=1, how="all")

    # === 关键：把 Domain,Label,L1,L2,L3 放在最前面 ===
    front = ["Domain","Label","L1","L2","L3"]
    others = [c for c in df.columns if c not in front]
    df = df[front + others]

    # 中位数填补特征列
    feat_cols = [c for c in df.columns if c not in front]
    df[feat_cols] = df[feat_cols].apply(lambda s: s.fillna(s.median()), axis=0)

    out_csv = os.path.join(args.outdir,"aggregated_cylike_features.csv")
    df.to_csv(out_csv,index=False)
    log(f"[OK] Aggregated CY-like features saved: {out_csv} (n={len(df)}, d={len(feat_cols)})")

if __name__ == "__main__":
    main()

