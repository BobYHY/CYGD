#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CATH CY-like feature aggregation only:
- 读取标签并规范 L1/L2/L3；
- 从 <Domain>_CY*.csv 中提取富 CY 统计特征；
- 不做任何 XGB/聚类，只输出 aggregated_cylike_features.csv；
- 在输出文件中增加一列 `name`（=Domain，方便下游统一用 name 标识样本）。

变更要点（保留自 plus_addCY 富集版）：
1) 标签解析与层级规范：
   - 统一支持 Domain + (C/A/T | Class/Architecture/Topology | L1/L2/L3) 或两列 (Domain, Label)；
   - 规范层级：L1=C；L2=C_A；L3=C_A_T；若只有整体 Label 则自动拆分。
2) CY 特征“富化”：
   - 对数值列进行稳健统计（mean/std/median/MAD/p10/p25/p75/p90/min/max/len/energy/IQR/tail_ratio/z_median/range_norm）；
   - 可选 FFT/ACF/直方图；
   - 多分组：全体(all)、k_local 分组(k_used==k)、二级结构分组(SS=H/E/C)、B 因子分箱（阈值自定）。
3) 新增：
   - --cols_numeric 作为“基础名（base names）”，自动扩展为所有匹配列：
     base -> base 或 base_k<digits>（不限 k，正则 ^<base>(?:_k\\d+)?$）。
   - 在 main() 中对已对齐的 CY 文件做“原始列并集 vs 实际使用列（按基础名展开）”的打印与落盘。
   - 输出文件命名为 aggregated_cylike_features.csv，并添加 name 列。
"""

import os, argparse, time, warnings, json, re
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Dict

warnings.filterwarnings("ignore")


# ----------------- utils -----------------
def log(msg): 
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_list(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]

def as_bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


# ----------------- Label parsing (rich & robust) -----------------
def _first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def read_labels_flexible(label_csv: str, domain_col: str | None, label_col: str | None) -> pd.DataFrame:
    """
    兼容：
      1) 无表头两列: Domain, Label
      2) 有表头: PDBID + (L1/L2/L3) 或 (C/A/T) 或 (Class/Architecture/Topology)
      3) 仅整体 Label（如 "1_10_8"）

    规范化：
      L1 = C
      L2 = C_A
      L3 = C_A_T
    """
    df = pd.read_csv(label_csv)
    df.columns = [c.strip() for c in df.columns]

    # Domain
    if domain_col:
        if domain_col not in df.columns:
            raise ValueError(f"--domain-col={domain_col} not in columns: {df.columns.tolist()}")
        df["Domain"] = df[domain_col].astype(str)
    else:
        dn = _first_existing(df, ["Domain", "PDBID", "pdbid", "domain"])
        dn = dn if dn is not None else df.columns[0]
        df["Domain"] = df[dn].astype(str)

    # 层级来源：优先显式三列
    C_col = _first_existing(df, ["C", "L1", "Class", "class"])
    A_col = _first_existing(df, ["A", "L2", "Architecture", "architecture"])
    T_col = _first_existing(df, ["T", "L3", "Topology", "topology"])

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
        # 用整体 Label（显式列或第二列）
        if label_col and label_col != "auto":
            if label_col not in df.columns:
                raise ValueError(f"--label-col={label_col} not in columns: {df.columns.tolist()}")
            lab = df[label_col].astype(str)
        else:
            lab = df["Label"].astype(str) if "Label" in df.columns else df.iloc[:, 1].astype(str)

        parts = lab.str.split("_", n=2, expand=True)
        while parts.shape[1] < 3:
            parts[parts.shape[1]] = ""
        C = parts[:, 0].fillna("").astype(str)
        A = parts[:, 1].fillna("").astype(str)
        T = parts[:, 2].fillna("").astype(str)
        df["L1"] = C
        df["L2"] = (C + "_" + A).str.rstrip("_")
        df["L3"] = (C + "_" + A + "_" + T).str.rstrip("_")
        df["Label"] = df["L3"]

    out = df[["Domain", "Label", "L1", "L2", "L3"]].dropna().drop_duplicates().reset_index(drop=True)
    n1, n2, n3 = out["L1"].nunique(), out["L2"].nunique(), out["L3"].nunique()
    log(f"[Labels] Rows={len(out)} | |L1|={n1} |L2|={n2} | |L3|={n3}")
    if n1 == n2 and (out["L1"].astype(str).values == out["L2"].astype(str).values).all():
        log("[WARN] L1 and L2 are identical after parsing — 检查 A/Architecture 是否缺失或命名不一致。")
    return out


# ----------------- CY rich features -----------------
def robust_stats_1d(x: np.ndarray) -> Dict[str, float]:
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

    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        median=float(med),
        mad=float(mad - 1e-9),
        p10=float(p10),
        p25=float(p25),
        p75=float(p75),
        p90=float(p90),
        min=float(np.min(x)),
        max=float(np.max(x)),
        length=int(x.size),
        energy=float(np.sum(x ** 2)),
        iqr=float(iqr - 1e-9),
        tail_ratio=float(tail_ratio),
        z_median=float(z_med),
        range_norm=float(range_norm),
    )

def fft_feats(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or x.size == 0:
        return np.zeros(k, dtype=np.float32)
    x = x.astype(np.float64)
    xf = np.fft.rfft(x - np.mean(x))
    mags = np.abs(xf)
    out = np.zeros(k, dtype=np.float32)
    m = min(k, mags.shape[0])
    out[:m] = mags[:m]
    return out

def acf_feats(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or x.size == 0:
        return np.zeros(k, dtype=np.float32)
    x = x.astype(np.float64)
    n = len(x)
    fft = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf = acf / acf[0] if acf[0] != 0 else acf
    out = np.zeros(k, dtype=np.float32)
    m = min(k, n)
    out[:m] = acf[:m]
    return out

def hist_feats(x: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0 or x.size == 0:
        return np.zeros(bins, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(bins, dtype=np.float32)
    lo, hi = np.quantile(x, 0.02), np.quantile(x, 0.98)
    if hi <= lo:
        lo, hi = x.min(), x.max() + 1e-6
    h, _ = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi), density=False)
    h = h.astype(np.float32)
    h /= (h.sum() + 1e-9)
    return h

def load_cy_file(path: str, use_global_residue: bool) -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- drop ResidueIndex == -1 when requested ---
    if not use_global_residue:
        if "ResidueIndex" in df.columns:
            rid = pd.to_numeric(df["ResidueIndex"], errors="coerce")
            before = len(df)
            df = df[~(rid == -1)].copy()
            after = len(df)
            if before != after:
                print(f"[useGlobalResidue=no] Dropped {before - after} global row(s) in {os.path.basename(path)}")
        else:
            print(f"[useGlobalResidue=no] WARN: no 'ResidueIndex' column in {os.path.basename(path)}")

    if "ResidueIndex" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing ResidueIndex")
    return df


def build_block(df: pd.DataFrame, cols: List[str], prefix: str,
                fft_k: int, acf_k: int, hist_bins: int) -> Dict[str, float]:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy()
        st = robust_stats_1d(x)
        for k, v in st.items():
            out[f"{prefix}_{c}_{k}"] = v
        if fft_k > 0:
            fv = fft_feats(x, fft_k)
            for i, val in enumerate(fv):
                out[f"{prefix}_{c}_fft{i}"] = float(val)
        if acf_k > 0:
            av = acf_feats(x, acf_k)
            for i, val in enumerate(av):
                out[f"{prefix}_{c}_acf{i}"] = float(val)
        if hist_bins > 0:
            hv = hist_feats(x, hist_bins)
            for i, val in enumerate(hv):
                out[f"{prefix}_{c}_hist{i}"] = float(val)
    return out


# ---------- NEW: base-name expansion helpers ----------
def expand_base_cols(df: pd.DataFrame, base_names: List[str]) -> List[str]:
    """
    将基础名扩展为 df 中实际存在的列：
      - 精确匹配基础名（如 'lag_mean'）
      - 以及所有 '基础名_k<数字>'（不限 k 值，正则匹配 ^<base>(?:_k\\d+)?$）

    若 base_names 为空：默认选择所有数值列（除 ResidueIndex）。
    返回保持 df.columns 原始顺序的去重列表。
    """
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

    # 编译：^<base>(?:_k\d+)?$
    regexes = []
    for base in base_names:
        key = base.split("_k")[0]  # 用户若误传 *_k12，也退回基础键
        pat = rf"^{re.escape(key)}(?:_k\d+)?$"
        regexes.append((base, re.compile(pat)))

    selected: List[str] = []
    seen = set()

    # 1) 精确基础名优先（也覆盖用户直接给全名如 *_k12 的情况）
    base_set = set(base_names)
    for c in df.columns:
        if c in base_set and c not in seen:
            selected.append(c)
            seen.add(c)

    # 2) 正则扩展
    for c in df.columns:
        for base, rgx in regexes:
            if rgx.fullmatch(c):
                if c not in seen:
                    selected.append(c)
                    seen.add(c)
                break

    return selected

def collect_union_columns(paths: List[str]) -> List[str]:
    """对齐后的 CY 文件列表，收集列名并集（保持首次出现顺序）。"""
    seen = set()
    union_cols = []
    for p in paths:
        try:
            dfh = pd.read_csv(p, nrows=2)
        except Exception:
            continue
        for c in dfh.columns:
            if c not in seen:
                union_cols.append(c)
                seen.add(c)
    return union_cols
# ---------- END NEW ----------


def build_cy_features_for_domain(csv_path: str,
                                 cols_numeric: List[str],
                                 fft_cols: List[str],
                                 k_local_list: List[int],
                                 ss_enable: bool,
                                 bfactor_enable: bool,
                                 b_cut: List[float],
                                 fft_k: int, acf_k: int, hist_bins: int,
                                 use_global_residue: bool) -> Dict[str, float]:
    df = load_cy_file(csv_path, use_global_residue=use_global_residue)
    feats = {}

    # 将基础名扩展为实际列（每个 df 单独展开）
    if cols_numeric and len(cols_numeric) > 0:
        local_cols = expand_base_cols(df, cols_numeric)
    else:
        local_cols = [
            c for c in df.columns
            if c.lower() != "residueindex"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    # 全局行（ResidueIndex == -1）
    if (df["ResidueIndex"] == -1).any():
        g = df[df["ResidueIndex"] == -1]
        for c in local_cols:
            if c in g.columns:
                v = pd.to_numeric(g[c], errors="coerce").to_numpy()
                if v.size > 0 and np.isfinite(v).any():
                    feats[f"global_{c}"] = float(np.nanmean(v))

    # 非全局序列
    loc = df[df["ResidueIndex"] >= 0].copy()
    if loc.empty:
        loc = df.copy()

    # 全体
    feats.update(build_block(
        loc, local_cols, prefix="all",
        fft_k=fft_k if not fft_cols else 0,
        acf_k=acf_k if not fft_cols else 0,
        hist_bins=hist_bins
    ))

    # 指定列再做 FFT/ACF（避免重复）
    for c in fft_cols:
        if c in loc.columns:
            x = pd.to_numeric(loc[c], errors="coerce").to_numpy()
            if fft_k > 0:
                fv = fft_feats(x, fft_k)
                for i, val in enumerate(fv):
                    feats[f"all_{c}_fft{i}"] = float(val)
            if acf_k > 0:
                av = acf_feats(x, acf_k)
                for i, val in enumerate(av):
                    feats[f"all_{c}_acf{i}"] = float(val)

    # k_local 分组（需有 k_used 列）
    if k_local_list and "k_used" in loc.columns:
        for kval in k_local_list:
            sub = loc[loc["k_used"] == kval]
            if sub.empty:
                continue
            feats.update(build_block(
                sub, local_cols, prefix=f"k{kval}",
                fft_k=0 if fft_cols else fft_k,
                acf_k=0 if fft_cols else acf_k,
                hist_bins=hist_bins
            ))
            for c in fft_cols:
                if c in sub.columns:
                    x = pd.to_numeric(sub[c], errors="coerce").to_numpy()
                    if fft_k > 0:
                        fv = fft_feats(x, fft_k)
                        for i, val in enumerate(fv):
                            feats[f"k{kval}_{c}_fft{i}"] = float(val)
                    if acf_k > 0:
                        av = acf_feats(x, acf_k)
                        for i, val in enumerate(av):
                            feats[f"k{kval}_{c}_acf{i}"] = float(val)

    # 二级结构分组
    if ss_enable and "SS" in loc.columns:
        ssu = loc["SS"].astype(str).str.upper()
        for tag, mask in [("H", ssu.str.startswith("H")),
                          ("E", ssu.str.startswith("E"))]:
            sub = loc[mask]
            if not sub.empty:
                feats.update(build_block(
                    sub, local_cols, prefix=f"ss{tag}",
                    fft_k=0 if fft_cols else fft_k,
                    acf_k=0 if fft_cols else acf_k,
                    hist_bins=hist_bins
                ))
        sub = loc[~(ssu.str.startswith("H") | ssu.str.startswith("E"))]
        if not sub.empty:
            feats.update(build_block(
                sub, local_cols, prefix="ssC",
                fft_k=0 if fft_cols else fft_k,
                acf_k=0 if fft_cols else acf_k,
                hist_bins=hist_bins
            ))

    # B 因子分箱
    if bfactor_enable and "B" in loc.columns:
        b = pd.to_numeric(loc["B"], errors="coerce")
        thr = sorted(b_cut)
        edges = [-np.inf] + thr + [np.inf]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sub = loc[(b > lo) & (b <= hi)]
            if sub.empty:
                continue
            feats.update(build_block(
                sub, local_cols, prefix=f"Bbin{i}",
                fft_k=0 if fft_cols else fft_k,
                acf_k=0 if fft_cols else acf_k,
                hist_bins=hist_bins
            ))
    return feats


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="CATH CY-like feature aggregation only")
    # I/O
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--cy_dir",    required=True, help="Dir with <Domain>_CY.csv")
    ap.add_argument("--outdir",    required=True)
    ap.add_argument("--domain-col", default="PDBID")
    ap.add_argument("--label-col",  default="auto")
    # feats
    ap.add_argument("--cols_numeric", default="",
                    help="comma list; interpreted as BASE names (auto-expand to *_k<digits>). "
                         "Leave empty to use all numeric except ResidueIndex.")
    ap.add_argument("--cy_fft_cols",  default="",
                    help="extra FFT/ACF only on these columns (avoid duplicate)")
    ap.add_argument("--fft_k", type=int, default=16)
    ap.add_argument("--acf_k", type=int, default=16)
    ap.add_argument("--hist_bins", type=int, default=12)
    ap.add_argument("--k_local_list", default="",
                    help="used if 'k_used' exists in CY")
    ap.add_argument("--group_ss", default="yes")
    ap.add_argument("--group_b",  default="yes")
    ap.add_argument("--b_cuts",   default="20,40")
    # 控制 ResidueIndex==-1
    ap.add_argument("--useGlobalResidue", default="yes",
                    help="yes/no: if 'no', drop rows with ResidueIndex == -1 from CY files before any aggregation")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    use_global_flag = as_bool(args.useGlobalResidue)

    # 读取标签
    labels = read_labels_flexible(args.label_csv, args.domain_col, args.label_col)
    labs = set(labels["Domain"])

    # 对齐 CY 文件（兼容 *_CY.csv / *_CY_multiK.csv / *_CYcore.csv）
    files_all = glob(os.path.join(args.cy_dir, "*_CY*.csv"))
    dom2file = {}
    for p in files_all:
        filename = os.path.basename(p)
        if filename.endswith("_CY.csv"):
            dom = filename.replace("_CY.csv", "")
        elif filename.endswith("_CY_multiK.csv"):
            dom = filename.replace("_CY_multiK.csv", "")
        elif filename.endswith("_CYcore.csv"):
            dom = filename.replace("_CYcore.csv", "")
        else:
            continue
        if dom in labs:
            dom2file[dom] = p

    log(f"[CY] domains: labels={len(labels)} files={len(files_all)} intersection={len(dom2file)}")

    # 原始列并集 vs 选用列（按基础名展开）
    base_names = parse_list(args.cols_numeric)
    matched_paths = list(dom2file.values())
    union_cols = collect_union_columns(matched_paths)
    union_df_for_expand = pd.DataFrame(columns=union_cols)
    selected_cols_global = expand_base_cols(union_df_for_expand, base_names)

    log("\n===== ORIGINAL (UNION) COLUMNS =====")
    log(f"count={len(union_cols)}")
    log(str(union_cols))

    log("\n===== SELECTED (EXPANDED) COLUMNS =====")
    log(f"count={len(selected_cols_global)}")
    log(str(selected_cols_global))

    try:
        with open(os.path.join(args.outdir, "original_columns_union.json"), "w", encoding="utf-8") as f:
            json.dump(union_cols, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.outdir, "selected_columns_expanded.json"), "w", encoding="utf-8") as f:
            json.dump(selected_cols_global, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"[WARN] failed to write columns summary: {e}")

    # 解析其他参数
    cols_numeric = base_names  # 注意：这里是“基础名列表”，真正展开在函数里按各 df 进行
    fft_cols     = parse_list(args.cy_fft_cols)
    k_local_list = [int(x) for x in parse_list(args.k_local_list)] if args.k_local_list else []
    b_cuts       = [float(x) for x in parse_list(args.b_cuts)] if args.b_cuts else [20.0, 40.0]

    # 构造特征
    rows = []
    for i, (dom, lab, l1, l2, l3) in enumerate(labels[["Domain", "Label", "L1", "L2", "L3"]].itertuples(index=False), 1):
        p = dom2file.get(dom)
        if not p:
            continue
        try:
            feats = build_cy_features_for_domain(
                p, cols_numeric=cols_numeric, fft_cols=fft_cols,
                k_local_list=k_local_list,
                ss_enable=as_bool(args.group_ss),
                bfactor_enable=as_bool(args.group_b),
                b_cut=b_cuts,
                fft_k=args.fft_k, acf_k=args.acf_k, hist_bins=args.hist_bins,
                use_global_residue=use_global_flag
            )
            if feats:
                feats["Domain"] = dom
                feats["Label"] = lab
                feats["L1"] = l1
                feats["L2"] = l2
                feats["L3"] = l3
                rows.append(feats)
        except Exception as e:
            log(f"[WARN] {dom}: {e}")
        if i % 4000 == 0:
            log(f"[CY] processed {i}/{len(labels)}")

    df = pd.DataFrame(rows)
    if df.empty:
        log("[ERR] No domains aggregated. 请检查 cy_dir 与 label 的域名是否一致。")
        return

    # 去空列
    df = df.dropna(axis=1, how="all")

    # 增加 name 列（=Domain）
    df.insert(df.columns.get_loc("Domain") + 1, "name", df["Domain"].astype(str))

    # === 关键：把 Domain,name,Label,L1,L2,L3 放到最前面 ===
    front = ["Domain", "name", "Label", "L1", "L2", "L3"]
    others = [c for c in df.columns if c not in front]
    df = df[front + others]

    # 中位数填补特征列
    feat_cols = [c for c in df.columns if c not in front]
    df[feat_cols] = df[feat_cols].apply(lambda s: s.fillna(s.median()), axis=0)

    out_csv = os.path.join(args.outdir, "aggregated_cylike_features.csv")
    df.to_csv(out_csv, index=False)
    log(f"[OK] Aggregated CY-like features saved: {out_csv}  (n={len(df)}, d={len(feat_cols)})")

if __name__ == "__main__":
    main()

