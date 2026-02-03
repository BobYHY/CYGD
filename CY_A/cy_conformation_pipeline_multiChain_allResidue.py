#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cy_conformation_pipeline_multiChain_allResidue.py

给定一个 PDB 和链选择 (A/H/L/ALL)，计算：

1. 全局 CY-style 量（由每残基局部量聚合）
2. 每个残基的局部 CY-style 量（以该残基为中心的 k 近邻）

输出：一个 CSV，其中
  - ResidueIndex = -1 : 全局统计行（global row）
  - ResidueIndex >= 0 : 每残基局部行（local rows）

所有行使用同一套特征列名：
  ResidueIndex, ChainID, ResSeq, InsCode, Resname3, Resname1,
  phi, psi, chi1,
  lag_local_mean, lag_local_p90, tangent_dim,
  sl_im, sl_abs, sl_im_calib, CYloc,
  ContactCount, ContactOrder, Bfactor_norm

其中：
  - sl_im        : |Im(Ω_i)|
  - sl_im_calib  : |Im(e^{-i θ*} Ω_i)|，全局相位校准后的 Im
  - CYloc        : 每残基的 CY-like 程度（色度场 / Chromatic Field），
                   在 [0,1]，1 = “最像 CY”（Im 最小）

注意：
- special-Lagrangian 相位 θ* 只在内存中用于几何和打分，
  不写入 CSV，以保证列语义在全局/局部一致。

如果加上 --plot yes，会同时画出：
  - CY-style 网格曲面 (mesh)
  - 诊断面板 (panels)

色度场（Chromatic Field）：
  - 由 --color 选出的列（默认优先 CYloc），用来着色 & 控制局部扭曲。
打分源（score source）：
  - 由 --score-source 选出的列，用于构造 CY_Score（全局分数）。
  - 对 CYloc 会内部转成 1-CYloc 当作“坏度”，值越小越好；
    对其他列直接当坏度（如 sl_im_calib, sl_im 等小越好）。

用法示例：

  python cy_conformation_pipeline_multiChain_allResidue.py \
      --pdb 7r58/7R58.pdb --chain ALL \
      --angles phi,psi,chi1 --k 30 --pca-var 0.90 \
      --contacts-cutoff 8.0 \
      --out-csv CY_ALL/7R58_ALL_CY.csv \
      --plot yes \
      --color CYloc \
      --score-source CYloc

"""

import os
import sys
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, is_aa, calc_dihedral

# --- 绘图相关 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# ================== 基础表 &工具 ==================

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

CHI1_GAMMA = {
    "ARG":"CG","LYS":"CG","GLN":"CG","GLU":"CG","MET":"CG",
    "LEU":"CG","ILE":"CG1","VAL":"CG1","THR":"OG1","SER":"OG",
    "CYS":"SG","ASN":"CG","ASP":"CG","HIS":"CG","PHE":"CG",
    "TYR":"CG","TRP":"CG","PRO":"CG"
}


def circular_mean(angles):
    """Circular mean of angles (radians), ignoring NaN."""
    arr = np.asarray(angles, float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return math.atan2(np.mean(np.sin(arr)), np.mean(np.cos(arr)))


def backbone_dihedrals(res_prev, res, res_next):
    """Return (phi, psi) for central residue; NaN if missing atoms."""
    phi = psi = float("nan")
    # phi: C(i-1) - N(i) - CA(i) - C(i)
    if res_prev is not None:
        C_prev = res_prev["C"] if "C" in res_prev else None
        N = res["N"] if "N" in res else None
        CA = res["CA"] if "CA" in res else None
        C = res["C"] if "C" in res else None
        if C_prev is not None and N is not None and CA is not None and C is not None:
            phi = calc_dihedral(C_prev.get_vector(), N.get_vector(),
                                CA.get_vector(), C.get_vector())
    # psi: N(i) - CA(i) - C(i) - N(i+1)
    if res_next is not None:
        N = res["N"] if "N" in res else None
        CA = res["CA"] if "CA" in res else None
        C = res["C"] if "C" in res else None
        N_next = res_next["N"] if "N" in res_next else None
        if N is not None and CA is not None and C is not None and N_next is not None:
            psi = calc_dihedral(N.get_vector(), CA.get_vector(),
                                C.get_vector(), N_next.get_vector())
    return phi, psi


def chi1(res):
    """Return chi1 dihedral in radians, or NaN if not defined/atoms missing."""
    name = res.get_resname()
    N  = res["N"]  if "N" in res  else None
    CA = res["CA"] if "CA" in res else None
    CB = res["CB"] if "CB" in res else None
    CGname = CHI1_GAMMA.get(name)
    CG = res[CGname] if (CGname and CGname in res) else None
    if N is None or CA is None or CB is None or CG is None:
        return float("nan")
    return calc_dihedral(N.get_vector(), CA.get_vector(),
                         CB.get_vector(), CG.get_vector())


def extract_residues(pdb_path, chain, model_index=0):
    """
    Extract standard-AA residues from a given chain or ALL chains.
    Returns a list of (chain_id, res_3letter, residue_obj).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    models = list(structure.get_models())
    if not models:
        raise ValueError("No models in PDB")
    if model_index >= len(models):
        raise ValueError(f"model_index={model_index} out of range")
    model = models[model_index]

    residues = []
    if chain is None or str(chain).upper() == "ALL":
        for ch in model:
            for res in ch:
                if is_aa(res, standard=True):
                    residues.append((ch.id, res.get_resname(), res))
    else:
        ch_sel = None
        for ch in model:
            if ch.id == chain:
                ch_sel = ch
                break
        if ch_sel is None:
            raise ValueError(f"Chain {chain} not found")
        for res in ch_sel:
            if is_aa(res, standard=True):
                residues.append((ch_sel.id, res.get_resname(), res))

    if not residues:
        raise ValueError("No standard amino-acid residues found")

    return structure, residues


def compute_angles_for_residues(residues, angles=("phi","psi","chi1")):
    """
    Given a list of (chain_id, resname3, res_obj), compute requested angles.
    Returns dict with keys:
      - phi, psi, chi1: lists
      - chain_ids, resname3, resname1, ca_xyz, bfactor_main
    """
    n = len(residues)
    out = {
        "phi": [], "psi": [], "chi1": [],
        "chain_ids": [], "resname3": [], "resname1": [],
        "resseq": [], "inscode": [],
        "ca_xyz": [], "bfactor_main": []
    }
    use_phi = "phi" in angles
    use_psi = "psi" in angles
    use_chi = "chi1" in angles  # 避免与函数 chi1 同名

    for i, (cid, name3, res) in enumerate(residues):
        # neighbors only within same chain (don't cross chains)
        res_prev = None
        res_next = None
        if i > 0 and residues[i-1][0] == cid:
            res_prev = residues[i-1][2]
        if i+1 < n and residues[i+1][0] == cid:
            res_next = residues[i+1][2]

        phi_val = psi_val = chi1_val = float("nan")
        if use_phi or use_psi:
            phi_val, psi_val = backbone_dihedrals(res_prev, res, res_next)
        if use_chi:
            chi1_val = chi1(res)

        out["phi"].append(phi_val if use_phi else float("nan"))
        out["psi"].append(psi_val if use_psi else float("nan"))
        out["chi1"].append(chi1_val if use_chi else float("nan"))

        out["chain_ids"].append(cid)
        out["resname3"].append(name3)
        out["resname1"].append(AA3_TO_1.get(name3, "X"))

        # PDB residue identifier: (hetflag, resseq, icode)
        _rid = res.get_id()
        out["resseq"].append(int(_rid[1]) if _rid and _rid[1] is not None else np.nan)
        _ic = _rid[2] if _rid and len(_rid) >= 3 else " "
        out["inscode"].append("" if _ic == " " else str(_ic).strip())

        CA = res["CA"] if "CA" in res else None
        if CA is not None:
            out["ca_xyz"].append(CA.get_coord().astype(float))
            b = CA.get_bfactor()
        else:
            # average backbone B-factor
            bb = []
            for nm in ("N","CA","C","O"):
                if nm in res:
                    bb.append(res[nm].get_bfactor())
            b = float(np.mean(bb)) if bb else float("nan")
            out["ca_xyz"].append(None)
        out["bfactor_main"].append(b)

    return out


def torus_embed(angle_mat):
    """Embed angles (N,m) -> (N,2m) as cos/sin."""
    A = np.asarray(angle_mat, float)
    cos = np.cos(A)
    sin = np.sin(A)
    return np.concatenate([cos, sin], axis=1)


def standard_J(dim_even):
    """Build standard symplectic matrix J (2m x 2m)."""
    assert dim_even % 2 == 0
    J = np.zeros((dim_even, dim_even), float)
    half = dim_even // 2
    for i in range(half):
        J[i, i+half] = -1.0
        J[i+half, i] = 1.0
    return J


def local_tangent_pca(Y, k, pca_var=0.9):
    """
    For each point Y[i], find k-NN, run PCA, and pick minimal
    dimension d s.t. cumulative var >= pca_var.

    Returns:
      - bases_list: list of (D x d_i) orthonormal bases
      - dims: np.array shape (N,) with each d_i
    """
    Y = np.asarray(Y, float)
    N, D = Y.shape
    k_eff = min(max(2, k), N)
    bases_list = []
    dims = np.zeros((N,), int)

    # precompute distances
    dmat = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)

    for i in range(N):
        idx = np.argsort(dmat[i])[:k_eff]  # includes self
        neigh = Y[idx]
        mu = np.mean(neigh, axis=0)
        Xc = neigh - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        if np.allclose(S, 0.0):
            # degenerate; choose 1D arbitrary
            d = 1
            B = np.eye(D, 1)
        else:
            var = (S**2) / np.sum(S**2)
            cumsum = np.cumsum(var)
            d = int(np.searchsorted(cumsum, pca_var) + 1)
            d = max(1, min(d, D))
            B = Vt[:d].T  # D x d
        bases_list.append(B)
        dims[i] = d

    return bases_list, dims


def lagrangian_scores(Y, bases_list, J):
    """
    For each local basis B (D x d), compute |ω| statistics.
    ω(u,v) = <J u, v>

    Returns arrays (N,) for lag_local_mean, lag_local_p90.
    """
    N, D = Y.shape
    lag_local_mean = np.zeros((N,), float)
    lag_local_p90  = np.zeros((N,), float)

    for i in range(N):
        B = bases_list[i]
        d = B.shape[1]
        if d < 2:
            lag_local_mean[i] = 0.0
            lag_local_p90[i] = 0.0
            continue
        ws = []
        for a in range(d):
            for b in range(a+1, d):
                u = B[:, a]
                v = B[:, b]
                w = float(np.dot(J @ u, v))
                ws.append(abs(w))
        if not ws:
            lag_local_mean[i] = 0.0
            lag_local_p90[i] = 0.0
            continue
        ws = np.asarray(ws, float)
        lag_local_mean[i] = float(ws.mean())
        lag_local_p90[i]  = float(np.quantile(ws, 0.90))
    return lag_local_mean, lag_local_p90


def special_lagrangian_m3(Y, bases_list):
    """
    For m=3 (angles*window=3), embed Y: (N,6) = (Re z1, Re z2, Re z3, Im z1, Im z2, Im z3).
    For each basis B, take first 3 columns and build complex 3x3 W, Ω = det(W).

    Returns:
      - omegas: complex array (N,) with Ω_i (may contain NaN+1jNaN)
      - stats: dict with global sl_abs_mean, sl_im_mean, sl_im_p90, sl_phase_var
    """
    Y = np.asarray(Y, float)
    N, D = Y.shape
    assert D == 6, "Special-Lagrangian m=3 requires D=6 (cos/sin of 3 angles)."

    omegas = np.full((N,), np.nan + 1j*np.nan, dtype=np.complex128)

    for i in range(N):
        B = bases_list[i]
        d = B.shape[1]
        if d < 3:
            continue
        B3 = B[:, :3]        # D x 3
        Re = B3[:3, :]       # 3 x 3
        Im = B3[3:, :]       # 3 x 3
        W = Re + 1j * Im
        omegas[i] = np.linalg.det(W)

    good = ~np.isnan(omegas)
    if not np.any(good):
        return omegas, {
            "sl_abs_mean": float("nan"),
            "sl_im_mean": float("nan"),
            "sl_im_p90": float("nan"),
            "sl_phase_var": float("nan"),
            "used_points": 0
        }

    v = omegas[good]
    abs_v = np.abs(v)
    im_abs = np.abs(v.imag)
    phases = np.angle(v)
    stats = {
        "sl_abs_mean": float(abs_v.mean()),
        "sl_im_mean": float(im_abs.mean()),
        "sl_im_p90": float(np.quantile(im_abs, 0.90)),
        "sl_phase_var": float(np.var(phases)),
        "used_points": int(good.sum())
    }
    return omegas, stats


def calibrate_phase_l1(omegas, grid_deg=1.0):
    """
    Find θ* in [0,2π) minimizing mean |Im(e^{-iθ} Ω_i)| using coarse grid.
    Returns (theta_star, rotated_omegas).
    """
    omegas = np.asarray(omegas, np.complex128)
    good = ~np.isnan(omegas)
    if not np.any(good):
        return 0.0, omegas

    v = omegas[good]
    n_steps = max(4, int(round(360.0 / grid_deg)))
    thetas = np.linspace(0.0, 2*np.pi, n_steps, endpoint=False)

    def obj(theta):
        r = v * np.exp(-1j * theta)
        return float(np.mean(np.abs(r.imag)))

    vals = [obj(t) for t in thetas]
    idx = int(np.argmin(vals))
    theta_star = float(thetas[idx])
    rotated = omegas * np.exp(-1j * theta_star)
    return theta_star, rotated


def compute_contacts_and_bfactor(angle_info, cutoff=8.0):
    """
    Compute per-residue contact count & mean sequence separation and normalized B-factor.

    angle_info: dict from compute_angles_for_residues.
    Returns:
      - contact_count: (N,) float
      - contact_order: (N,) float
      - bfactor_norm:  (N,) float in ~[0.5,1.5]
    """
    N = len(angle_info["chain_ids"])
    xyz = angle_info["ca_xyz"]
    pts = []
    idx_map = []
    for i, p in enumerate(xyz):
        if p is not None:
            pts.append(p)
            idx_map.append(i)
    if not pts:
        contact_count = np.zeros((N,), float)
        contact_order = np.zeros((N,), float)
    else:
        P = np.vstack(pts)
        contact_count = np.zeros((N,), float)
        contact_order = np.zeros((N,), float)
        for a in range(P.shape[0]):
            i = idx_map[a]
            pa = P[a]
            local_orders = []
            c = 0
            for b in range(P.shape[0]):
                if a == b:
                    continue
                j = idx_map[b]
                # exclude sequential neighbors
                if abs(i - j) <= 1:
                    continue
                if np.linalg.norm(pa - P[b]) <= cutoff:
                    c += 1
                    local_orders.append(abs(i - j))
            contact_count[i] = c
            contact_order[i] = np.mean(local_orders) if local_orders else 0.0

    # normalize B-factor
    b = np.asarray(angle_info["bfactor_main"], float)
    if np.all(np.isnan(b)):
        bfactor_norm = np.ones((N,), float)
    else:
        m = np.nanmean(b)
        s = np.nanstd(b)
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0
        z = (b - m) / s
        bfactor_norm = 1.0 + 0.5 * np.tanh(z)  # ~[0.5,1.5]
    return contact_count, contact_order, bfactor_norm


# ================== 绘图辅助（本脚本内部用，不再从 CSV 重读） ==================

def local_phase_energy(x, smooth_win=11):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([])
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        E = np.zeros_like(x)
    else:
        xn = (x - xmin) / (xmax - xmin)
        phase = np.unwrap(2*np.pi * xn)
        g = np.gradient(phase)
        E = g**2
        if smooth_win and smooth_win > 1:
            k = np.ones(smooth_win, float); k /= k.sum()
            E = np.convolve(E, k, mode="same")
        Emin, Emax = E.min(), E.max()
        if Emax - Emin < 1e-12:
            E = np.zeros_like(E)
        else:
            E = (E - Emin) / (Emax - Emin)
    return E


def choose_color_series(df, prefer="auto"):
    """
    从 df 中选一条用于「色度场 / Chromatic Field」的序列（仅用 ResidueIndex>=0 的行）。
    默认优先顺序：CYloc → sl_im_calib → sl_im → lag_local_mean。
    """
    mask = (df["ResidueIndex"] >= 0)
    if prefer != "auto" and prefer in df.columns:
        s = pd.to_numeric(df.loc[mask, prefer], errors="coerce").to_numpy(float)
        s = s[np.isfinite(s)]
        if s.size > 0:
            return s, prefer

    for cand in ["CYloc", "sl_im_calib", "sl_im", "lag_local_mean"]:
        if cand in df.columns:
            s = pd.to_numeric(df.loc[mask, cand], errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if s.size > 0:
                return s, cand
    return np.array([]), None


def choose_score_series(df, score_source="auto", default_col=None):
    """
    选择一条用于「打分源 (score source)」的序列。
    - score_source != auto 时：优先用该列；
    - 否则：默认用 default_col（通常是色度场列）；
    - 再不行则退回 choose_color_series(auto)。
    返回：(series, colname)。
    """
    mask = (df["ResidueIndex"] >= 0)

    # 1) 显式指定
    if score_source and score_source != "auto":
        if score_source in df.columns:
            s = pd.to_numeric(df.loc[mask, score_source], errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if s.size > 0:
                return s, score_source

    # 2) 默认用着色列
    if default_col and default_col in df.columns:
        s = pd.to_numeric(df.loc[mask, default_col], errors="coerce").to_numpy(float)
        s = s[np.isfinite(s)]
        if s.size > 0:
            return s, default_col

    # 3) 最后 fallback：重新走一遍 auto
    return choose_color_series(df, prefer="auto")

def build_mesh_stats(df, score_series, theta_star):
    """
    根据用于打分的序列（score_series）和 global row 构建 mesh 所需的全局统计。

    - lag_mean, lag_p90, slvar 来自 score_series（ResidueIndex>=0 的那一条曲线）
    - sl_im_mean_calib / sl_im_mean / sl_abs_mean / tangent_dim_mean
      来自 CSV 里的全局行 (ResidueIndex = -1)，语义是「全局平均」
    - theta_star 只从主程序内存传入，用来控制 mesh 的整体扭转相位
    """
    stats = {}

    # 1) 由打分序列本身统计 "平不平"
    xs = np.asarray(score_series, float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        stats["lag_mean"] = np.nan
        stats["lag_p90"]  = np.nan
        stats["slvar"]    = np.nan
    else:
        stats["lag_mean"] = float(np.nanmean(xs))
        stats["lag_p90"]  = float(np.quantile(xs, 0.90))
        stats["slvar"]    = float(np.nanvar(xs))

    # 2) 从全局行 (ResidueIndex = -1) 里取出若干全局量
    g = df.loc[df["ResidueIndex"] == -1]
    if len(g) > 0:
        g0 = g.iloc[0]

        def gval(col):
            if col not in g0:
                return np.nan
            v = pd.to_numeric(g0[col], errors="coerce")
            return float(v) if np.isfinite(v) else np.nan

        # 全局 special-Lagrangian 相关统计（都是用统一列名）
        stats["sl_im_mean_calib"] = gval("sl_im_calib")
        stats["sl_im_mean"]       = gval("sl_im")
        stats["sl_abs_mean"]      = gval("sl_abs")
        stats["tangent_dim_mean"] = gval("tangent_dim")
    else:
        stats["sl_im_mean_calib"] = np.nan
        stats["sl_im_mean"]       = np.nan
        stats["sl_abs_mean"]      = np.nan
        stats["tangent_dim_mean"] = np.nan

    # 3) theta_star 直接从主程序传进来
    stats["theta"] = float(theta_star) if np.isfinite(theta_star) else 0.0
    return stats

def cy_score(stats, mode="pos01"):
    """
    根据 lag_mean 和 slvar 合成一个 [0,1] 的 CY_Score，仅用于 mesh 标题展示。
    此处的 lag_mean/slvar 已经是“坏度”（badness）统计：小=好，因此 map01_small_is_better。
    """
    lm = stats.get("lag_mean", np.nan)
    sv = stats.get("slvar", np.nan)
    if not np.isfinite(lm) or not np.isfinite(sv):
        return np.nan

    def map01_small_is_better(x):
        if not np.isfinite(x): return np.nan
        if x <= 0: return 1.0
        if x >= 1.0: return 0.0
        return float(1.0 - x)

    a = map01_small_is_better(lm)
    b = map01_small_is_better(sv)
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if mode == "neg":
        return -10.0 * ((1-a) + (1-b))
    else:
        s = (a + b) / 2.0
        return float(np.clip(s, 0.0, 1.0))


def render_mesh(dom, stats, cseq, out_png, lw=0.7, cmap_name="viridis",
                tau_alpha=0.3, radius_alpha=0.08, pos_score=None,
                colorbar_label="Chromatic Field"):
    """
    根据 stats 和 cseq 画 CY-style 网格。
    - lm/slvar (来自打分源) 控制整体半径 R 和扭曲参数
    - sl_im_mean_calib 控制管子粗细 r
    - cseq（色度场序列）控制局部扭曲纹路 & 颜色
    """
    lm = stats.get("lag_mean", np.nan)
    sv  = stats.get("slvar", np.nan)
    th  = stats.get("theta", 0.0)
    imc = stats.get("sl_im_mean_calib", np.nan)

    # radius vs lag_mean (smaller badness → larger radius)
    R = 1.6 + 0.6 * np.tanh(np.nan_to_num(lm, nan=0.2))
    # tube thickness vs global sl_im_mean_calib
    r = 0.55 + 0.35 * np.exp(-np.nan_to_num(imc, nan=0.1)*3)

    svn = np.nan_to_num(sv, nan=0.2)
    tau_base = 1.0 + 3.0 * (svn/(svn+1.0)) + 0.5*np.cos(np.nan_to_num(th, nan=0.0))

    U, V = 220, 90
    u = np.linspace(0, 2*np.pi, U, endpoint=False)
    v = np.linspace(0, 2*np.pi, V, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    # 几何扭曲：仍然用「色度场」序列的局部相位能量
    if cseq is not None and len(cseq)>0:
        E = local_phase_energy(cseq)
    else:
        E = np.array([])

    if E.size>0:
        eU = np.interp(np.linspace(0, len(E)-1, U), np.arange(len(E)), E)
        eU = eU - np.mean(eU)
        tau_u = tau_base + tau_alpha*eU
        r_u   = r*(1.0 + radius_alpha*eU)
    else:
        tau_u = np.full(U, tau_base); r_u = np.full(U, r)

    w = 0.0
    if cseq is not None and len(cseq)>0:
        cs = np.asarray(cseq, float); cs = cs[np.isfinite(cs)]
        if cs.size:
            csn = (cs - cs.min())/(cs.max()-cs.min()+1e-9)
            w = 0.15 * (np.interp(np.linspace(0, len(csn)-1, U), np.arange(len(csn)), csn) - 0.5)
        else:
            w = np.zeros(U)
    else:
        w = np.zeros(U)

    Rterm = r_u[:,None]*(1+w[:,None])*np.cos(vv + tau_u[:,None]*uu)
    X = (R + Rterm) * np.cos(uu)
    Y = (R + Rterm) * np.sin(uu)
    Z =  r_u[:,None]*(1+w[:,None])*np.sin(vv + 0.65*np.sin(tau_u[:,None]*uu))

    # 色度场着色
    if cseq is not None and len(cseq)>0:
        cs = np.asarray(cseq, float); cs = cs[np.isfinite(cs)]
        if cs.size:
            cvals = (cs - cs.min())/(cs.max()-cs.min()+1e-9)
            cstrip = np.interp(np.linspace(0, len(cvals)-1, U), np.arange(len(cvals)), cvals)
        else:
            cstrip = np.linspace(0,1,U)
    else:
        cstrip = np.linspace(0,1,U)

    cmap = get_cmap(cmap_name)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")

    for i in range(U):
        col = cmap(cstrip[i])
        ax.plot(X[i,:], Y[i,:], Z[i,:], lw=lw, color=col, alpha=0.95)
    for j in range(0, V, 2):
        col = cmap(cstrip[j % U])
        ax.plot(X[:,j], Y[:,j], Z[:,j], lw=lw, color=col, alpha=0.6)

    ax.set_axis_off()
    ax.view_init(elev=24, azim=30)

    title = f"{dom}: CY-style mesh"
    if pos_score is not None and np.isfinite(pos_score):
        title += f"  |  CY_Score={pos_score:.3f}"
    plt.title(title, color="white")

    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_clim(0.0, 1.0)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(colorbar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

    plt.tight_layout()
    fig.savefig(out_png, facecolor="black", dpi=180)
    plt.close(fig)


def plot_panels(dom, df, out_png):
    """
    诊断面板：局部 CY 序列、局部“相位能量”、直方图和角度曲线。
    使用列：sl_im/sl_im_calib 或 lag_local_mean，以及 phi/psi/chi1。
    """
    cols = df.columns
    angles = [c for c in ["phi","psi","chi1"] if c in cols]
    mask = (df["ResidueIndex"]>=0)

    # Try to visualize sl_im vs sl_im_calib (or lag_local_mean) if present
    raw_col  = None
    cali_col = None
    if "sl_im" in cols and "sl_im_calib" in cols:
        raw_col, cali_col = "sl_im", "sl_im_calib"
    elif "lag_local_mean" in cols:
        raw_col, cali_col = "lag_local_mean", None

    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(2,2,1)
    if raw_col:
        s1 = pd.to_numeric(df.loc[mask, raw_col], errors="coerce").to_numpy(float)
        s1 = s1[np.isfinite(s1)]
        if cali_col:
            s2 = pd.to_numeric(df.loc[mask, cali_col], errors="coerce").to_numpy(float)
            s2 = s2[np.isfinite(s2)]
            if s1.size and s2.size:
                ax1.plot(s1, label=f"{raw_col}")
                ax1.plot(s2, label=f"{cali_col}")
                ax1.legend()
        elif s1.size:
            ax1.plot(s1, label=f"{raw_col}")
            ax1.legend()
        ax1.set_title(f"{dom}: local CY metrics")
        ax1.set_xlabel("Residue index")
        ax1.set_ylabel("value")
    else:
        ax1.text(0.5,0.5,"no raw/calib columns", ha="center", va="center")

    ax2 = fig.add_subplot(2,2,2)
    # If CYloc present, show phase energy of CYloc; else fallback to lag_local_mean
    series_for_E = None
    if "CYloc" in cols:
        series_for_E = pd.to_numeric(df.loc[mask, "CYloc"], errors="coerce").to_numpy(float)
    elif "lag_local_mean" in cols:
        series_for_E = pd.to_numeric(df.loc[mask, "lag_local_mean"], errors="coerce").to_numpy(float)
    if series_for_E is not None:
        series_for_E = series_for_E[np.isfinite(series_for_E)]
        E = local_phase_energy(series_for_E, 7)
        if E.size: ax2.plot(E)
    ax2.set_title("Local phase energy proxy"); ax2.set_xlabel("Residue index"); ax2.set_ylabel("E")

    ax3 = fig.add_subplot(2,2,3)
    if raw_col:
        s = pd.to_numeric(df.loc[mask, raw_col], errors="coerce").to_numpy(float)
        s = s[np.isfinite(s)]
        if s.size: ax3.hist(s, bins=40)
    ax3.set_title("Histogram of local metric"); ax3.set_xlabel("value"); ax3.set_ylabel("count")

    ax4 = fig.add_subplot(2,2,4)
    if angles:
        for c in angles:
            s = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if s.size: ax4.plot(s, label=c)
        ax4.legend(); ax4.set_title("Angles (phi/psi/chi1)")
    else:
        ax4.text(0.5,0.5,"No angle columns", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ================== main ==================

def main():
    ap = argparse.ArgumentParser(
        description="Per-residue CY-style Lagrangian/Special-Lagrangian metrics for PDB (with optional plotting)."
    )
    ap.add_argument("--pdb", required=True, help="Input PDB file")
    ap.add_argument("--chain", default="ALL", help="Chain ID (e.g. A) or ALL for all chains")
    ap.add_argument("--angles", default="phi,psi,chi1",
                    help="Comma-separated list from {phi,psi,chi1}")
    ap.add_argument("--k", type=int, default=30, help="k-NN size for local tangent PCA")
    ap.add_argument("--pca-var", type=float, default=0.90,
                    help="PCA cumulative variance threshold in [0,1]")
    ap.add_argument("--aa-allow", default=None,
                    help="Optional 1-letter AA whitelist, e.g. L,I,V,M,F")
    ap.add_argument("--aa-min-frac", type=float, default=None,
                    help="If set, require fraction of allowed AAs >= this threshold (0-1)")
    ap.add_argument("--contacts-cutoff", type=float, default=8.0,
                    help="CA-CA distance cutoff for contacts (Å)")
    ap.add_argument("--grid-deg", type=float, default=1.0,
                    help="Grid step (degrees) for theta_star search")
    ap.add_argument("--out-csv", required=True, help="Output CSV path (e.g. *_CY.csv)")

    # 绘图控制
    ap.add_argument("--plot", default="no", choices=["yes","no"],
                    help="Whether to render CY mesh and panels (default: no)")
    ap.add_argument("--mesh-png", default=None,
                    help="Optional output path for mesh PNG; "
                         "if not set and --plot yes, derive from out-csv")
    ap.add_argument("--panels-png", default=None,
                    help="Optional output path for panels PNG; "
                         "if not set and --plot yes, derive from out-csv")
    ap.add_argument("--color", default="auto",
                    help="Chromatic Field column for coloring (default=auto: CYloc/sl_im_calib/...)")
    # ★ 新增：打分所用的序列（色度场可以不同）
    ap.add_argument("--score-source", default="CYloc",
                    help="Column used to compute CY_Score; "
                         "one of CYloc, sl_im_calib, sl_im, lag_local_mean (default: CYloc)")
    ap.add_argument("--score-mode", default="pos01", choices=["neg","pos01"],
                    help="CY_Score mode; pos01 gives 0..1 higher=better")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--lw", type=float, default=0.7)
    ap.add_argument("--tau-alpha", type=float, default=0.3)
    ap.add_argument("--radius-alpha", type=float, default=0.08)

    args = ap.parse_args()

    angles_list = [s.strip() for s in args.angles.split(",") if s.strip()]
    # Ensure valid names
    valid_names = {"phi","psi","chi1"}
    for a in angles_list:
        if a not in valid_names:
            raise ValueError(f"Unsupported angle name: {a}")

    # 1) Extract residues & angle series
    structure, residues = extract_residues(args.pdb, args.chain, model_index=0)
    angle_info = compute_angles_for_residues(residues, angles=tuple(angles_list))
    N_total = len(residues)

    # 2) Build basic angle matrix (per residue, window=1)
    angle_mat = []
    aa_allow_set = None
    if args.aa_allow:
        aa_allow_set = set(a.strip() for a in args.aa_allow.split(",") if a.strip())

    keep_idx = []
    for i in range(N_total):
        aa1 = angle_info["resname1"][i]
        if aa_allow_set is not None and args.aa_min_frac is not None:
            # For window=1, aa_min_frac is effectively all-or-nothing
            if aa1 not in aa_allow_set:
                continue
        row = []
        for a in angles_list:
            row.append(angle_info[a][i])
        angle_mat.append(row)
        keep_idx.append(i)

    if not keep_idx:
        raise ValueError("No residues left after AA filtering.")

    angle_mat = np.asarray(angle_mat, float)
    M = angle_mat.shape[0]
    m = angle_mat.shape[1]

    # 2.5) 处理缺失角度：对每一列用 circular mean 填补 NaN
    for j in range(m):
        col = angle_mat[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            raise ValueError(f"All values are NaN for angle {angles_list[j]}")
        mean_j = circular_mean(col[finite])
        col[~finite] = mean_j
        angle_mat[:, j] = col

    # 3) Torus embedding + local tangent PCA
    Y = torus_embed(angle_mat)          # (M, 2m)
    D = Y.shape[1]
    J = standard_J(D)
    bases_list, dims = local_tangent_pca(Y, k=args.k, pca_var=args.pca_var)

    lag_local_mean, lag_local_p90 = lagrangian_scores(Y, bases_list, J)

    # 4) Special-Lagrangian if m=3
    omegas = None
    sl_stats = None
    theta_star = None
    omegas_rot = None
    sl_im = sl_abs = sl_im_calib = None
    CYloc = None

    if m == 3:
        omegas, sl_stats = special_lagrangian_m3(Y, bases_list)
        theta_star, omegas_rot = calibrate_phase_l1(omegas, grid_deg=args.grid_deg)
        # per-residue metrics
        sl_im = np.full((M,), np.nan, float)
        sl_abs = np.full((M,), np.nan, float)
        sl_im_calib = np.full((M,), np.nan, float)
        good = ~np.isnan(omegas)
        if np.any(good):
            v = omegas[good]
            v_rot = omegas_rot[good]
            sl_im[good] = np.abs(v.imag)
            sl_abs[good] = np.abs(v)
            sl_im_calib[good] = np.abs(v_rot.imag)

        # CY-likeness in [0,1] from sl_im_calib
        CYloc = np.full((M,), np.nan, float)
        good2 = np.isfinite(sl_im_calib)
        if np.any(good2):
            s = sl_im_calib[good2]
            smin, smax = float(s.min()), float(s.max())
            if smax - smin < 1e-12:
                CYloc[good2] = 1.0
            else:
                norm = (s - smin) / (smax - smin)
                CYloc[good2] = 1.0 - norm
        # fill sl_stats missing fields if necessary
        if sl_stats is None:
            sl_stats = {}
        sl_stats.setdefault("sl_abs_mean", float("nan"))
        sl_stats.setdefault("sl_im_mean", float("nan"))
        sl_stats.setdefault("sl_im_p90", float("nan"))
        sl_stats.setdefault("sl_phase_var", float("nan"))
        sl_stats.setdefault("used_points", int(np.isfinite(omegas).sum()))
    else:
        # No special-Lagrangian; all related fields NaN
        sl_stats = {
            "sl_abs_mean": float("nan"),
            "sl_im_mean": float("nan"),
            "sl_im_p90": float("nan"),
            "sl_phase_var": float("nan"),
            "used_points": 0
        }
        theta_star = 0.0

    # 5) Contacts + normalized B-factor (computed on full residue list, then subset)
    contact_count_full, contact_order_full, bfactor_norm_full = compute_contacts_and_bfactor(
        angle_info, cutoff=args.contacts_cutoff
    )
    contact_count = np.asarray(contact_count_full)[keep_idx]
    contact_order = np.asarray(contact_order_full)[keep_idx]
    bfactor_norm = np.asarray(bfactor_norm_full)[keep_idx]

    # 6) Global stats（通过局部量聚合）
    lag_mean_global = float(np.mean(lag_local_mean))
    lag_p90_global  = float(np.quantile(lag_local_mean, 0.90))
    tangent_dim_mean = float(np.mean(dims)) if dims.size > 0 else float("nan")

    sl_abs_mean  = float(sl_stats.get("sl_abs_mean", float("nan")))
    sl_im_mean   = float(sl_stats.get("sl_im_mean",  float("nan")))
    sl_im_p90    = float(sl_stats.get("sl_im_p90",   float("nan")))
    sl_phase_var = float(sl_stats.get("sl_phase_var", float("nan")))

    if sl_im_calib is not None and np.any(np.isfinite(sl_im_calib)):
        s = sl_im_calib[np.isfinite(sl_im_calib)]
        sl_im_mean_calib = float(np.mean(s))
        sl_im_p90_calib  = float(np.quantile(s, 0.90))
    else:
        sl_im_mean_calib = float("nan")
        sl_im_p90_calib  = float("nan")

    # CYloc 全局：取 CYloc 的平均即可
    if CYloc is not None and np.any(np.isfinite(CYloc)):
        CYloc_global = float(np.nanmean(CYloc))
    else:
        CYloc_global = float("nan")

    # Contact / B-factor 全局平均
    ContactCount_global  = float(np.nanmean(contact_count)) if contact_count.size else float("nan")
    ContactOrder_global  = float(np.nanmean(contact_order)) if contact_order.size else float("nan")
    Bfactor_norm_global  = float(np.nanmean(bfactor_norm)) if bfactor_norm.size else float("nan")

    # 7) Build DataFrame
    rows = []

    # 7.1 Global row: ResidueIndex = -1，列名与局部一样，语义为全局统计
    global_row = {
        "ResidueIndex": -1,
        "ChainID": "GLOBAL",
        "ResSeq": np.nan,
        "InsCode": "",
        "Resname3": "",
        "Resname1": "",
        "phi": np.nan,
        "psi": np.nan,
        "chi1": np.nan,
        "lag_local_mean": lag_mean_global,
        "lag_local_p90":  lag_p90_global,
        "tangent_dim":    tangent_dim_mean,
        "sl_im":          sl_im_mean,
        "sl_abs":         sl_abs_mean,
        "sl_im_calib":    sl_im_mean_calib,
        "CYloc":          CYloc_global,
        "ContactCount":   ContactCount_global,
        "ContactOrder":   ContactOrder_global,
        "Bfactor_norm":   Bfactor_norm_global,
    }
    rows.append(global_row)

    # 7.2 Per-residue rows（真正的局部特征）
    for j, idx in enumerate(keep_idx):
        row = {
            "ResidueIndex": j,  # local index among kept residues
            "ChainID":  angle_info["chain_ids"][idx],
            "ResSeq":   int(angle_info["resseq"][idx]) if np.isfinite(angle_info["resseq"][idx]) else np.nan,
            "InsCode":  str(angle_info["inscode"][idx]) if angle_info["inscode"][idx] is not None else "",
            "Resname3": angle_info["resname3"][idx],
            "Resname1": angle_info["resname1"][idx],
            "phi": angle_info["phi"][idx] if "phi" in angles_list else np.nan,
            "psi": angle_info["psi"][idx] if "psi" in angles_list else np.nan,
            "chi1": angle_info["chi1"][idx] if "chi1" in angles_list else np.nan,
            "lag_local_mean": float(lag_local_mean[j]),
            "lag_local_p90":  float(lag_local_p90[j]),
            "tangent_dim":    int(dims[j]),
            "sl_im":          float(sl_im[j]) if sl_im is not None else np.nan,
            "sl_abs":         float(sl_abs[j]) if sl_abs is not None else np.nan,
            "sl_im_calib":    float(sl_im_calib[j]) if sl_im_calib is not None else np.nan,
            "CYloc":          float(CYloc[j]) if CYloc is not None else np.nan,
            "ContactCount":   float(contact_count[j]),
            "ContactOrder":   float(contact_order[j]),
            "Bfactor_norm":   float(bfactor_norm[j]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved per-residue CY metrics to {args.out_csv}")
    print(f"[INFO] Residues kept: {M} / {N_total}; m={m}, k={args.k}, pca_var={args.pca_var}")

    # 8) 可选：绘图
    if args.plot.lower() == "yes":
        # 1) 色度场：用于上色
        cseq, cname = choose_color_series(df, prefer=args.color)
        if cseq.size == 0:
            print("[WARN] No usable series for coloring; skip plotting.")
            return

        # 2) 打分曲线：用于 CY_Score 以及 mesh 的整体几何参数
        score_seq, sname = choose_color_series(df, prefer=args.score_source)
        if score_seq.size == 0:
            # 找不到就退回到色度场同一条
            score_seq, sname = cseq, cname
            print(f"[WARN] score-source '{args.score_source}' not found; "
                  f"fallback to color series '{cname}'.")

        # 3) 用打分序列构建 stats（lag_mean/var 等）
        stats = build_mesh_stats(df, score_seq, theta_star)
        score = cy_score(stats, mode=args.score_mode)

        # 4) 构造标题中的名字：7R58_<链>_<score-source>
        pdb_base = os.path.splitext(os.path.basename(args.pdb))[0]
        chain_label = str(args.chain).upper()
        dom_title = f"{pdb_base}_{chain_label}_{sname}"

        # 5) 默认文件名：7R58_<链>_<score-source>_<color>_CYmesh/_panels
        out_dir = os.path.dirname(os.path.abspath(args.out_csv))
        color_label = cname if cname is not None else args.color
        file_base = f"{pdb_base}_{chain_label}_{sname}_{color_label}"

        mesh_png = args.mesh_png or os.path.join(out_dir, f"{file_base}_CYmesh.png")
        panels_png = args.panels_png or os.path.join(out_dir, f"{file_base}_panels.png")

        # 6) 色条标签：Chromatic Field (xxx)
        if cname is None:
            colorbar_label = "Chromatic Field"
        else:
            colorbar_label = f"Chromatic Field ({cname})"

        render_mesh(dom_title, stats, cseq, mesh_png,
                    lw=args.lw, cmap_name=args.cmap,
                    tau_alpha=args.tau_alpha,
                    radius_alpha=args.radius_alpha,
                    pos_score=score,
                    colorbar_label=colorbar_label)

        plot_panels(dom_title, df, panels_png)
        print(f"[OK] Mesh plot saved to {mesh_png}")
        print(f"[OK] Panels plot saved to {panels_png}")

if __name__ == "__main__":
    main()


