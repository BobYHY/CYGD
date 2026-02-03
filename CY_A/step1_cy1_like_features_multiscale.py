#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CY-core multi-k extractor (Scheme A, enhanced, v2)
--------------------------------------------------

改动要点：
- 仍然只取最长的标准氨基酸链。
- 角度：phi, psi, omega, chi1，按原脚本方式计算。
- CY 背景：用 (phi, psi, chi1) 嵌入 (S^1)^3 → R^6。
- 每个残基、每个 k：在 R^6 中取最近 k 个邻居做 PCA，
  得到局部切空间本征维数 d（用于 dim_stats），
  实际用于 Special-Lagrangian 计算的主方向数强制 ≥3（若样本数足够）。
- 计算：
    * Lagrangian 偏差: lag_mean_kX
    * Special Lagrangian 相关: sl_abs_mean_kX, sl_im_mean_calib_kX, theta_star_kX
    * CYloc_kX：基于 sl_im_mean_calib_kX 在同一条链上的归一化 CY 色度场，
      值越接近 1 越“CY-like”，越接近 0 偏离 CY。
- 仍支持可选 enhanced 特征（lag_p90 / sl_im 等），
  但不再把 n_points / k_used 当作默认“核心特征”；
  默认 enhanced 只包含 B_factor，n_points / k_used 仍可显式要求。
- 新增与 k 无关的接触特征：
    * ContactCount：每个残基 CA 在 cutoff 内（去掉 |i-j|<=1 的邻近）的接触个数。
    * ContactOrder：这些接触的平均 |i-j|（序列距离）。
- 自动填补 NaN/inf → 0.0。
- 每个 PDB 输出: <stem>_CYcore.csv（每行一个 ResidueIndex）。
- 额外输出: dim_stats.csv (每个 k 的 d 分布统计；d 为 PCA 的本征维数)。

注意：
- ContactCount / ContactOrder 与 k 无关，放在基础列中，每残基一组。
- CYloc_kX 作为“与 k 相关的核心特征”加入输出。
"""

import os
import sys
import math
import argparse
import csv
import glob
import traceback
import warnings
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from io import StringIO

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# 限制 BLAS 线程数
for kenv in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(kenv, "1")


# =========================
# 1. 角度 / B-factor / 接触 提取
# =========================

CHI1_GAMMA = {
    "ARG": "CG", "LYS": "CG", "GLN": "CG", "GLU": "CG", "MET": "CG",
    "LEU": "CG", "ILE": "CG1", "VAL": "CG1", "THR": "OG1", "SER": "OG",
    "CYS": "SG", "ASN": "CG", "ASP": "CG", "HIS": "CG", "PHE": "CG",
    "TYR": "CG", "TRP": "CG", "PRO": "CG"
}


def backbone_angles(res_prev, res, res_next):
    """
    计算 backbone dihedral：
    - phi(i)   = C(i-1)-N(i)-CA(i)-C(i)
    - psi(i)   = N(i)-CA(i)-C(i)-N(i+1)
    - omega(i) 这里沿用旧脚本做法，也用 C(i-1)-N(i)-CA(i)-C(i)
    """
    phi = psi = omega = None
    if res_prev is not None:
        C_prev = _get_atom(res_prev, "C")
        N = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        C = _get_atom(res, "C")
        if (C_prev is not None) and (N is not None) and (CA is not None) and (C is not None):
            phi = calc_dihedral(C_prev.get_vector(), N.get_vector(),
                                CA.get_vector(), C.get_vector())
            omega = calc_dihedral(C_prev.get_vector(), N.get_vector(),
                                  CA.get_vector(), C.get_vector())
    if res_next is not None:
        N = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        C = _get_atom(res, "C")
        Nn = _get_atom(res_next, "N")
        if (N is not None) and (CA is not None) and (C is not None) and (Nn is not None):
            psi = calc_dihedral(N.get_vector(), CA.get_vector(),
                                C.get_vector(), Nn.get_vector())
    return phi, psi, omega


def chi1(res):
    N = _get_atom(res, "N")
    CA = _get_atom(res, "CA")
    CB = _get_atom(res, "CB")
    CGname = CHI1_GAMMA.get(res.get_resname(), None)
    CG = _get_atom(res, CGname) if CGname else None
    if (N is not None) and (CA is not None) and (CB is not None) and (CG is not None):
        return calc_dihedral(N.get_vector(), CA.get_vector(),
                             CB.get_vector(), CG.get_vector())
    return None

def _get_atom(res, name):
    try:
        return res[name]
    except KeyError:
        return None

def _residue_bfactor(res):
    """
    与 CY2 的 B_avg 对齐的定义：
    - 优先使用 backbone 四个原子 N, CA, C, O 的 B-factor 平均；
    - 如果这四个里只存在一部分，就对存在的那几项平均；
    - 如果一个都没有，则退回到 CA；
    - 再不行，才用所有原子 B-factor 的平均。
    """
    backbone_names = ("N", "CA", "C", "O")
    vals = []
    for name in backbone_names:
        atom = _get_atom(res, name)
        if atom is not None:
            try:
                vals.append(float(atom.get_bfactor()))
            except Exception:
                pass

    # 1) 若有至少一个 backbone 原子，按这些做平均
    if vals:
        return float(np.mean(vals))

    # 2) 若没有 backbone，则尝试单独 CA
    ca = _get_atom(res, "CA")
    if ca is not None:
        try:
            return float(ca.get_bfactor())
        except Exception:
            pass

    # 3) 仍不行则所有原子平均兜底
    vals = []
    for atom in res.get_unpacked_list():
        try:
            vals.append(float(atom.get_bfactor()))
        except Exception:
            pass
    if vals:
        return float(np.mean(vals))

    return np.nan

def circular_mean(rad_list):
    xs = ys = 0.0
    cnt = 0
    for v in rad_list:
        if v is None:
            continue
        xs += math.cos(v)
        ys += math.sin(v)
        cnt += 1
    if cnt == 0:
        return None
    return math.atan2(ys, xs)


def circular_mean_window(vals, i, w):
    if vals is None or len(vals) == 0:
        return None
    if w <= 1:
        return vals[i]
    r = w // 2
    lo = max(0, i - r)
    hi = min(len(vals), i + r + 1)
    return circular_mean(vals[lo:hi])

def _is_supported_residue(res):
    """
    判断是否参与 CY 计算的残基：
    - 不再只限于 standard=True；
    - 只要是 amino acid（包括 MSE 等非标准），或者至少有 CA 原子，就认为是“可用残基”。
    """
    # 1) Biopython 认为的氨基酸（包括非标准，例如 MSE）
    if is_aa(res, standard=False):
        return True

    # 2) UNK/其它自定义名字，至少要有 CA 原子
    if _get_atom(res, "CA") is not None:
        return True

    return False

def parse_pdb_safely(pdb_path):
    """
    安全解析 PDB：
    1）先尝试直接解析整文件；
    2）若失败或没有任何残基，再尝试去掉 ANISOU 解析；
    3）若仍然没有残基，则从第一条 ATOM/HETATM 开始截断（去掉前面的 MODEL 等）再解析。
    目标：返回一个包含至少 1 个残基的 structure。
    """
    # 先读入所有行，后面多种策略复用
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    def _try_parse(lines_variant, tag):
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        try:
            handle = StringIO("".join(lines_variant))
            structure = parser.get_structure("x", handle)
        except Exception:
            return None
        # 看看有没有残基
        try:
            n_res = sum(1 for _ in structure.get_residues())
        except Exception:
            n_res = 0
        if n_res > 0:
            # print(f"[DEBUG] parse_pdb_safely: {tag} -> {n_res} residues")
            return structure
        return None

    # 1) 原始全文本尝试（可能在 set_anisou 这里爆掉）
    structure = _try_parse(lines, "full")
    if structure is not None:
        return structure

    # 2) 去掉 ANISOU 再试
    clean_lines = [ln for ln in lines if not ln.startswith("ANISOU")]
    structure = _try_parse(clean_lines, "noANISOU_full")
    if structure is not None:
        return structure

    # 3) 去掉 ANISOU 后，从第一个 ATOM/HETATM 行开始截断
    atom_start = None
    for i, ln in enumerate(clean_lines):
        if ln.startswith("ATOM") or ln.startswith("HETATM"):
            atom_start = i
            break
    if atom_start is not None:
        structure = _try_parse(clean_lines[atom_start:], "noANISOU_trunc")
        if structure is not None:
            return structure

    # 4) 实在不行：用原始 lines，从第一个 ATOM/HETATM 开始截断再试一次
    atom_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("ATOM") or ln.startswith("HETATM"):
            atom_start = i
            break
    if atom_start is not None:
        structure = _try_parse(lines[atom_start:], "full_trunc")
        if structure is not None:
            return structure

    # 5) 所有策略都失败，给出明确信息
    raise ValueError(f"parse_pdb_safely: could not parse any residues from {pdb_path}")

def extract_chain_and_residues(pdb_path):
    # 使用安全解析，自动处理坏 ANISOU
    structure = parse_pdb_safely(pdb_path)
    model = next(structure.get_models())
    best_chain, best_len = None, -1
    for ch in model:
        residues = [r for r in ch if _is_supported_residue(r)]
        if len(residues) > best_len:
            best_chain, best_len = ch, len(residues)
    residues = [r for r in best_chain if _is_supported_residue(r)] if best_chain else []
    return structure, model, best_chain, residues

def extract_angles_longest_chain(pdb_path):
    structure, model, chain, residues = extract_chain_and_residues(pdb_path)
    L = len(residues)
    phi_list, psi_list, omg_list, chi1_list, b_list = [], [], [], [], []
    for i, res in enumerate(residues):
        prev = residues[i - 1] if i > 0 else None
        nxt = residues[i + 1] if i + 1 < L else None
        phi, psi, omg = backbone_angles(prev, res, nxt)
        phi_list.append(phi)
        psi_list.append(psi)
        omg_list.append(omg)
        chi1_list.append(chi1(res))
        b_list.append(_residue_bfactor(res))
    return structure, chain, residues, [phi_list, psi_list, omg_list, chi1_list], b_list

def extract_angles_multi_chain(pdb_path, chain_mode="ALL"):
    """
    提取多条链的 backbone 角度 & chi1 & B-factor：
    - chain_mode = "ALL": 使用所有标准氨基酸链；
    - chain_mode = 具体链名（如 "A"）: 只用该链。
    """
    structure = parse_pdb_safely(pdb_path)
    model = next(structure.get_models())

    chain_mode_up = str(chain_mode).upper()
    use_all = (chain_mode_up == "ALL")

    residues_all = []
    phi_list, psi_list, omg_list, chi1_list, b_list = [], [], [], [], []

    for ch in model:
        if (not use_all) and (ch.id != chain_mode):
            continue
        residues = [r for r in ch if _is_supported_residue(r)]
        if not residues:
            continue

        L = len(residues)
        for i, res in enumerate(residues):
            prev = residues[i - 1] if i > 0 else None
            nxt  = residues[i + 1] if i + 1 < L else None
            phi, psi, omg = backbone_angles(prev, res, nxt)
            phi_list.append(phi)
            psi_list.append(psi)
            omg_list.append(omg)
            chi1_list.append(chi1(res))
            b_list.append(_residue_bfactor(res))
            residues_all.append(res)

    return structure, model, residues_all, [phi_list, psi_list, omg_list, chi1_list], b_list

def build_angles_matrix(angles_lists, window):
    phi, psi, omg, chi = angles_lists
    L = len(phi)
    if L == 0:
        return np.empty((0, 3), float), [], [], [], []
    if window < 1 or (window % 2 == 0):
        window = 1

    phi_s = [circular_mean_window(phi, i, window) for i in range(L)]
    psi_s = [circular_mean_window(psi, i, window) for i in range(L)]
    omg_s = [circular_mean_window(omg, i, window) for i in range(L)]
    chi_s = [circular_mean_window(chi, i, window) for i in range(L)]

    mu_phi = circular_mean(phi_s)
    mu_psi = circular_mean(psi_s)
    mu_omg = circular_mean(omg_s)
    mu_chi = circular_mean(chi_s)

    rows = []
    for a, b, c, d in zip(phi_s, psi_s, omg_s, chi_s):
        aa = mu_phi if a is None else a
        bb = mu_psi if b is None else b
        cc = mu_omg if c is None else c
        dd = mu_chi if d is None else d
        # 嵌入用的角度：phi, psi, chi1
        rows.append([aa, bb, dd])
    X = np.array(rows, dtype=float)
    return X, phi_s, psi_s, omg_s, chi_s

def compute_contacts(residues, cutoff=8.0):
    """
    基于 CA-CA 距离的接触定义（支持多链）：
    - ContactCount(i): 满足 dist(CA_i, CA_j) <= cutoff 且
        * 同链时要求 |ResSeq_i - ResSeq_j| > 1
        * 跨链时只要求 j != i
    - ContactOrder(i):
        * 同链接触：用 |ResSeq_i - ResSeq_j| 做“序列距离”
        * 跨链接触：退化为全局索引距离 |i - j|
    """
    n = len(residues)
    if n == 0:
        return np.zeros(0, float), np.zeros(0, float)

    coords   = np.full((n, 3), np.nan, float)
    chains   = []
    resseqs  = []

    for i, res in enumerate(residues):
        ca = _get_atom(res, "CA")
        if ca is not None:
            coords[i] = ca.get_coord().astype(float)
        # 链 ID & ResSeq
        parent = res.get_parent()
        chain_id = getattr(parent, "id", "?")
        chains.append(chain_id)
        # res.id 是类似 (' ', resseq, icode) 的 tuple
        try:
            resseqs.append(int(res.get_id()[1]))
        except Exception:
            resseqs.append(i)  # 兜底：用全局索引代替

    contact_count = np.zeros((n,), float)
    contact_order = np.zeros((n,), float)

    for i in range(n):
        if not np.all(np.isfinite(coords[i])):
            continue
        di = np.linalg.norm(coords - coords[i], axis=1)

        partners = []
        orders   = []
        for j in range(n):
            if j == i:
                continue
            if not np.all(np.isfinite(coords[j])):
                continue

            same_chain = (chains[i] == chains[j])
            if same_chain and abs(resseqs[i] - resseqs[j]) <= 1:
                # 同链且相邻 / 自身 → 当作主链邻居，不计入接触
                continue

            if di[j] <= cutoff:
                partners.append(j)
                if same_chain:
                    seqsep = abs(resseqs[i] - resseqs[j])
                else:
                    # 跨链 → 用索引差当作粗略序列距离
                    seqsep = abs(i - j)
                orders.append(seqsep)

        contact_count[i] = float(len(partners))
        contact_order[i] = float(np.mean(orders)) if orders else 0.0

    return contact_count, contact_order

# =========================
# 2. CY 嵌入 + 局部 PCA
# =========================
def torus_embed(X):
    # 先把 NaN / inf 填成 0.0，避免 NearestNeighbors 报错
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.cos(X)
    S = np.sin(X)
    return np.concatenate([C, S], axis=1)  # (N,6)


def standard_J(dim_even):
    J = np.zeros((dim_even, dim_even))
    for k in range(dim_even // 2):
        a, b = 2 * k, 2 * k + 1
        J[a, b] = -1.0
        J[b, a] = 1.0
    return J


def precompute_neighbors(Y, max_k):
    n = len(Y)
    k_eff = min(max_k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(Y)
    dists, inds = nbrs.kneighbors(Y)
    return dists, inds  # inds[i][0] ~ self


def local_PCA(Y, idxs, pca_var=0.90, enforce_min_dim=3):
    """
    在邻域 idxs 上做 PCA。
    - 返回 U: (d_use, D)（注意 sklearn 的 components_ 是 (n_components, D)）
    - d_eig: 本征维数（按 pca_var 截断），用于 dim_stats 统计。
    - 实际用于 SL 计算的主方向数 d_use 至少为 enforce_min_dim（若样本数允许）。
    """
    if len(idxs) < 2:
        return None, 0
    X = Y[idxs] - Y[idxs[0]]
    if not np.any(np.var(X, axis=0) > 1e-12):
        return None, 0

    pca = PCA(n_components=min(3, X.shape[1])).fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    d_thr = int(np.searchsorted(cum, pca_var) + 1)
    d_thr = max(1, min(d_thr, pca.n_components_))

    d_eig = d_thr  # 本征维数

    # 用于 SL 的维数（尽量 >= enforce_min_dim）
    if enforce_min_dim is not None and pca.n_components_ >= enforce_min_dim:
        d_use = max(d_eig, enforce_min_dim)
        d_use = min(d_use, pca.n_components_)
    else:
        d_use = min(d_eig, pca.n_components_)

    U = pca.components_[:d_use]  # (d_use, D)
    return U, d_eig


def holo_volume(u1, u2, u3):
    def toC(u):
        return np.array([u[0] + 1j * u[1],
                         u[2] + 1j * u[3],
                         u[4] + 1j * u[5]], dtype=np.complex128)
    W = np.vstack([toC(u1), toC(u2), toC(u3)])
    return np.linalg.det(W)


def theta_best(vals, grid_deg=1.0):
    thetas = np.arange(0.0, 360.0, grid_deg) * math.pi / 180.0
    best = None
    best_theta = 0.0
    for th in thetas:
        im = np.abs(np.imag(vals * np.exp(-1j * th)))
        score = float(np.mean(im))
        if best is None or score < best:
            best = score
            best_theta = th
    return best_theta

# ====== 1A. CY 单链版：omega / local_tangent / global scores (从 cy_conformation_pipeline 适配) ======

def omega(u, v, J):
    """标准辛形式 ω(u,v) = <Ju, v>."""
    return float((J @ u).dot(v))


def local_tangent(Y, idx, k=20, pca_var=0.90, max_tangent_dim=None, nbrs=None):
    """
    与 cy_conformation_pipeline.py 中一致：
    - 邻域里去掉自己
    - 邻居不足或零方差 → 返回空切空间
    """
    n = len(Y)
    if n <= 1:
        return np.zeros((0, Y.shape[1])), np.array([idx]), None
    k_eff = min(k + 1, n)  # 包含自己，稍后去掉
    if nbrs is None:
        nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(Y)
    dists, inds = nbrs.kneighbors(Y[idx:idx+1])
    inds = inds[0]
    inds_wo_self = [j for j in inds if j != idx]
    if len(inds_wo_self) < 2:
        return np.zeros((0, Y.shape[1])), np.array(inds), nbrs
    Xn = Y[np.array(inds_wo_self)] - Y[idx]
    if not np.any(np.var(Xn, axis=0) > 1e-12):
        return np.zeros((0, Y.shape[1])), np.array(inds), nbrs
    pca = PCA().fit(Xn)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = int(np.searchsorted(cum, pca_var) + 1)
    if max_tangent_dim is None:
        max_tangent_dim = Y.shape[1] // 2
    d = max(1, min(d, max_tangent_dim))
    U = pca.components_[:d]
    return U, np.array(inds), nbrs


def lagrangian_scores(Y, J, k=20, pca_var=0.90):
    """
    最早单链版的 global Lagrangian 统计：
    - 遍历每个样本 i 的局部切空间 U_i
    - 把所有 |ω(u_a, u_b)| 丢到一个大列表里，再整体算 mean / p90
    """
    n = len(Y)
    m = Y.shape[1] // 2
    vals, used = [], 0
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm="auto").fit(Y)
    for i in range(n):
        U, _, _ = local_tangent(Y, i, k=k, pca_var=pca_var, max_tangent_dim=m, nbrs=nbrs)
        d = U.shape[0]
        if d < 2:
            continue
        used += 1
        for a in range(d):
            for b in range(a+1, d):
                vals.append(abs(omega(U[a], U[b], J)))
    if not vals:
        return dict(lag_mean=np.nan, lag_p90=np.nan, used_points=0)
    arr = np.array(vals, float)
    return dict(
        lag_mean=float(arr.mean()),
        lag_p90=float(np.percentile(arr, 90)),
        used_points=int(used),
    )


def _sl_stats_from_vals(vals):
    """
    从一组 Ω（复数）统计 |ImΩ|、相位方差、|Ω|。
    与 cy_conformation_pipeline 保持一致。
    """
    if vals is None:
        return dict(sl_im_mean=None, sl_im_p90=None, sl_phase_var=None, sl_abs_mean=None)
    vals = np.asarray(vals, np.complex128)
    if vals.size == 0:
        return dict(sl_im_mean=None, sl_im_p90=None, sl_phase_var=None, sl_abs_mean=None)

    imvals = np.abs(vals.imag)
    mags   = np.abs(vals)
    phases = np.unwrap(np.angle(vals))
    return dict(
        sl_im_mean=float(np.mean(imvals)),
        sl_im_p90=float(np.percentile(imvals, 90)),
        sl_phase_var=float(np.var(phases)),
        sl_abs_mean=float(np.mean(mags)),
    )


def holomorphic_volume_m3_exact(u1, u2, u3):
    """
    直接复用本文件的 holo_volume，实现与单链版同样的 Ω 计算。
    """
    return holo_volume(u1, u2, u3)


def compute_omega_vals_m3(Y, k=30, pca_var=0.95):
    """
    与单链版一样：遍历每个样本 i 的 3 维切空间，算 Ω_i。
    """
    n = len(Y)
    m = Y.shape[1] // 2
    assert m == 3
    vals = []
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm="auto").fit(Y)
    for i in range(n):
        U, _, _ = local_tangent(Y, i, k=k, pca_var=pca_var, max_tangent_dim=m, nbrs=nbrs)
        if U.shape[0] != 3:
            continue
        U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        vals.append(holomorphic_volume_m3_exact(U_norm[0], U_norm[1], U_norm[2]))
    return vals


def special_lag_scores_m3(
    Y, J, k=30, pca_var=0.95,
    calibrate=True, calibrate_mode='l1', grid_deg=1.0,
    theta_scan_png=None, theta_scan_steps=360,
):
    """
    完整拷贝单链版的 global SL 逻辑（去掉绘图依赖）：
      - sl_im_mean / sl_im_p90 / sl_phase_var / sl_abs_mean
      - 校准后的 sl_im_mean_calib / sl_im_p90_calib / sl_phase_var_calib / calib_theta
    """
    vals = compute_omega_vals_m3(Y, k=max(k, 10), pca_var=max(pca_var, 0.95))
    used = len(vals)
    stats = _sl_stats_from_vals(vals)
    stats["used_points"] = int(used)
    stats.update(dict(
        sl_im_mean_calib=None,
        sl_im_p90_calib=None,
        sl_phase_var_calib=None,
        calib_theta=None,
    ))

    if used == 0 or not calibrate:
        return stats

    vals = np.asarray(vals, np.complex128)
    if calibrate_mode == 'l2':
        theta = float(np.angle(np.sum(vals)))
    elif calibrate_mode in ('l1', 'grid'):
        # grid 和 l1 在单链版中都用格点搜索 mean |Im(e^{-iθ}Ω)|
        if calibrate_mode == 'grid':
            thetas = np.linspace(0.0, 2*np.pi, max(3, theta_scan_steps), endpoint=False)
        else:
            step = math.radians(grid_deg)
            nstep = int(np.ceil(2*np.pi/step))
            thetas = np.linspace(0.0, 2*np.pi, max(3, nstep), endpoint=False)

        def mean_abs_im(theta):
            v = vals * np.exp(-1j*theta)
            return float(np.mean(np.abs(v.imag)))

        ys = [mean_abs_im(t) for t in thetas]
        theta = float(thetas[int(np.argmin(ys))])
    else:
        # 兜底：用 circular_mean
        theta = float(circular_mean(np.angle(vals)) or 0.0)

    vcal = vals * np.exp(-1j * theta)
    cstats = _sl_stats_from_vals(vcal)
    stats.update(dict(
        sl_im_mean_calib=cstats["sl_im_mean"],
        sl_im_p90_calib=cstats["sl_im_p90"],
        sl_phase_var_calib=cstats["sl_phase_var"],
        calib_theta=float(theta),
    ))
    return stats


def cy_global_stats_single_chain(Y, ks, pca_var, grid_deg):
    """
    给定一条链的 torus 嵌入 Y，按「单链版」逻辑对每个 k 计算一组 global 统计。
    返回: {k: {feature_name: value}}
    """
    if Y.size == 0:
        return {k: {} for k in ks}

    J = standard_J(Y.shape[1])  # dim_even = 6
    m = Y.shape[1] // 2
    has_m3 = (m == 3)

    out = {}
    for k in ks:
        g = {}
        lag = lagrangian_scores(Y, J, k=k, pca_var=pca_var)
        g["lag_mean"] = lag.get("lag_mean", np.nan)
        g["lag_p90"]  = lag.get("lag_p90",  np.nan)

        if has_m3:
            sl = special_lag_scores_m3(
                Y, J, k=max(k, 20), pca_var=max(pca_var, 0.95),
                calibrate=True, calibrate_mode='l1',
                grid_deg=grid_deg, theta_scan_png=None, theta_scan_steps=360,
            )
            g["sl_im_mean"]          = sl.get("sl_im_mean",          np.nan)
            g["sl_im_p90"]           = sl.get("sl_im_p90",           np.nan)
            g["sl_phase_var"]        = sl.get("sl_phase_var",        np.nan)
            g["sl_abs_mean"]         = sl.get("sl_abs_mean",         np.nan)
            g["sl_im_mean_calib"]    = sl.get("sl_im_mean_calib",    np.nan)
            g["sl_im_p90_calib"]     = sl.get("sl_im_p90_calib",     np.nan)
            g["sl_phase_var_calib"]  = sl.get("sl_phase_var_calib",  np.nan)
            g["theta_star"]          = sl.get("calib_theta",         np.nan)
        out[k] = g
    return out



def fill_nans_inplace(d):
    """把 float 中的 NaN/inf 统一填成 0.0"""
    for k, v in list(d.items()):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                d[k] = 0.0

def _angle_global_mean(angle_list):
    """
    对一条链上的角度列表做 circular mean。
    angle_list 里元素可以是弧度或 None；如果全是 None 返回 np.nan。
    """
    v = circular_mean(angle_list)
    if v is None:
        return np.nan
    return float(v)

def cy_features_for_k(Y, inds_full, center, k, pca_var,
                      sl_auto_k=False, max_k_auto=None, grid_deg=1.0):
    """
    对单个残基、单个 k 计算 CY 几何特征。
    sl_auto_k=True 时，若 d_eig<3，将自动增大邻域 k→k+2→... 直到 d_eig>=3 或达到 max_k_auto。
    返回: (features dict, d_eig)，其中 d_eig 为 PCA 本征维数（用于 dim_stats）。
    实际用于 SL 计算的主方向数 ≥3（若样本数足够）。
    """
    n_total = len(Y)
    if max_k_auto is None:
        max_k_auto = n_total

    cur_k = max(k, 3)
    U = None
    d_eig = 0
    idxs = None

    while True:
        cur_k = min(cur_k, n_total)
        idxs = inds_full[center][:cur_k]
        U, d_eig = local_PCA(Y, idxs, pca_var=pca_var, enforce_min_dim=3)

        if not sl_auto_k:
            break

        if d_eig >= 3:
            break
        if cur_k >= max_k_auto or cur_k >= n_total:
            break

        cur_k += 2

    if U is None:
        feats = dict(
            lag_mean=np.nan,
            sl_im_mean_calib=np.nan,
            theta_star=np.nan,
            sl_abs_mean=np.nan,

            lag_p90=np.nan,
            sl_im_mean=np.nan,
            sl_im_p90=np.nan,
            sl_phase_var=np.nan,
            sl_im_p90_calib=np.nan,
            sl_phase_var_calib=np.nan,

            n_points=len(idxs) if idxs is not None else 0,
            k_used=cur_k,
        )
        fill_nans_inplace(feats)
        return feats, d_eig

    J = standard_J(Y.shape[1])
    lag_vals = []
    d_use = U.shape[0]

    if d_use >= 2:
        for i in range(d_use):
            for j in range(i + 1, d_use):
                lag_vals.append(abs((J @ U[i]).dot(U[j])))
        lag_vals = np.array(lag_vals, float)
        lag_mean = float(lag_vals.mean())
        lag_p90 = float(np.percentile(lag_vals, 90))
    else:
        lag_mean = lag_p90 = np.nan

    sl_vals = []
    if d_use >= 3:
        for i in range(d_use):
            for j in range(i + 1, d_use):
                for l in range(j + 1, d_use):
                    sl_vals.append(holo_volume(U[i], U[j], U[l]))
        sl_vals = np.array(sl_vals, dtype=np.complex128)

        abs_vals = np.abs(sl_vals)
        sl_abs_mean = float(abs_vals.mean()) if abs_vals.size else np.nan

        im_raw = np.abs(np.imag(sl_vals)) if sl_vals.size else np.array([], float)
        sl_im_mean = float(im_raw.mean()) if im_raw.size else np.nan
        sl_im_p90 = float(np.percentile(im_raw, 90)) if im_raw.size else np.nan
        sl_phase_var = float(np.var(np.angle(sl_vals))) if sl_vals.size else np.nan

        if sl_vals.size:
            th = theta_best(sl_vals, grid_deg=grid_deg)
            sl_calib = sl_vals * np.exp(-1j * th)
            im_c = np.abs(np.imag(sl_calib))
            sl_im_mean_calib = float(im_c.mean())
            sl_im_p90_calib = float(np.percentile(im_c, 90))
            sl_phase_var_calib = float(np.var(np.angle(sl_vals) - th))
            theta_star = float(th)
        else:
            sl_im_mean_calib = sl_im_p90_calib = sl_phase_var_calib = np.nan
            theta_star = np.nan
    else:
        sl_abs_mean = sl_im_mean = sl_im_p90 = sl_phase_var = np.nan
        sl_im_mean_calib = sl_im_p90_calib = sl_phase_var_calib = np.nan
        theta_star = np.nan

    feats = dict(
        lag_mean=lag_mean,
        sl_im_mean_calib=sl_im_mean_calib,
        theta_star=theta_star,
        sl_abs_mean=sl_abs_mean,

        lag_p90=lag_p90,
        sl_im_mean=sl_im_mean,
        sl_im_p90=sl_im_p90,
        sl_phase_var=sl_phase_var,
        sl_im_p90_calib=sl_im_p90_calib,
        sl_phase_var_calib=sl_phase_var_calib,

        n_points=len(idxs),
        k_used=cur_k,
    )

    fill_nans_inplace(feats)
    return feats, d_eig


# =========================
# 3. worker：单个 PDB → CSV
# =========================
def process_one(pdb_path, out_dir, ks, pca_var, window,
                enhanced_list, sl_auto_k, max_k_auto,
                contacts_cutoff, grid_deg, chain_mode):
    """
    返回: (pdb_path, ok, msg, dim_stats_local)
    其中 dim_stats_local 是 {k: {dim_eig: count}} 的字典，用于汇总 d 分布。
    """
    base = os.path.basename(pdb_path)
    stem, _ = os.path.splitext(base)
    out_csv = os.path.join(out_dir, f"{stem}_CYcore.csv")

    dim_stats_local = {}

    # 如果输出文件已经存在且非空，则跳过（认为已经处理过）
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return (pdb_path, True, "skip_existing", dim_stats_local)

    try:
        chain_mode_up = str(chain_mode).upper()

        if chain_mode_up == "LONGEST":
            # 兼容旧行为：只取最长标准氨基酸链
            structure, chain, residues, angles_lists, b_vals = extract_angles_longest_chain(pdb_path)
        else:
            # 多链模式：使用 ALL 或指定链 ID
            structure, model, residues, angles_lists, b_vals = extract_angles_multi_chain(
                pdb_path,
                chain_mode=chain_mode_up if chain_mode_up != "ALL" else "ALL",
            )

        X, phi_s, psi_s, omg_s, chi_s = build_angles_matrix(angles_lists, window)

        if X.shape[0] == 0:
            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                #writer.writerow(["ResidueIndex"])
                writer.writerow(["ResidueIndex", "Chain", "ResSeq", "ResName"])
            return (pdb_path, True, "empty", dim_stats_local)

        # 与 k 无关的接触特征
        contact_count, contact_order = compute_contacts(residues, cutoff=contacts_cutoff)

        Y = torus_embed(X)
        N = len(Y)
        ks = sorted(set(ks))
        maxK = max(ks)
        _, inds_full = precompute_neighbors(Y, maxK)

        base_cols = [
            #"ResidueIndex",
            "ResidueIndex", "Chain", "ResSeq", "ResName",
            "phi", "psi", "omega", "chi1",
            "ContactCount", "ContactOrder",
        ]
        extra_cols = []
        if "B_factor" in enhanced_list:
            extra_cols.append("B_factor")

        # 核心 CY 特征（不含 CYloc；CYloc 单独计算并输出）
        K_DEP_CORE = [
            "lag_mean",
            "sl_im_mean_calib",
            "theta_star",
            "sl_abs_mean",
        ]

        ENHANCED_ALL = [
            "lag_p90",
            "sl_im_mean",
            "sl_im_p90",
            "sl_phase_var",
            "sl_im_p90_calib",
            "sl_phase_var_calib",
            "n_points",
            "k_used",
        ]

        enhanced_k_feats = [c for c in enhanced_list if c in ENHANCED_ALL]
        K_DEP = K_DEP_CORE + enhanced_k_feats

        feats_by_k = {}
        dims_by_k = {}
        for k in ks:
            feats_by_k[k] = {name: np.zeros((N,), float) for name in K_DEP}
            dims_by_k[k] = np.zeros((N,), int)

        # 第一遍：CY 几何量
        for i in range(N):
            for k in ks:
                feats, d_eig = cy_features_for_k(
                    Y, inds_full, i, k, pca_var=pca_var,
                    sl_auto_k=sl_auto_k, max_k_auto=max_k_auto,
                    grid_deg=grid_deg
                )
                dim_stats_local.setdefault(k, {})
                dim_stats_local[k][d_eig] = dim_stats_local[k].get(d_eig, 0) + 1
                dims_by_k[k][i] = d_eig

                for name in K_DEP:
                    feats_by_k[k][name][i] = feats[name]

        # 第二遍：基于 sl_im_mean_calib_k? 计算 CYloc_k?
        cyloc_by_k = {}
        for k in ks:
            s = feats_by_k[k]["sl_im_mean_calib"].copy()
            if s.size == 0:
                cyloc_by_k[k] = np.zeros((N,), float)
                continue
            s_min = float(np.min(s))
            s_max = float(np.max(s))
            if (not np.isfinite(s_min)) or (not np.isfinite(s_max)) or (s_max - s_min) < 1e-12:
                cyloc_by_k[k] = np.ones((N,), float)
            else:
                norm = (s - s_min) / (s_max - s_min)
                cyloc_by_k[k] = 1.0 - norm

        # 写出 CSV
        header = base_cols + extra_cols
        for k in ks:
            header += [f"{c}_k{k}" for c in K_DEP]
            header.append(f"CYloc_k{k}")

        # 先按「单链版」逻辑为每个 k 计算 global 统计
        global_stats_by_k = cy_global_stats_single_chain(Y, ks, pca_var=pca_var, grid_deg=grid_deg)

        os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # ===== 2A. 写 global 行：ResidueIndex = -1 =====
            global_row = []

            # (a) base_cols 部分
            global_row.append(-1)   # ResidueIndex
            global_row.extend(["GLOBAL", -1, "GLOBAL"])  # Chain, ResSeq, ResName

            # phi / psi / omega / chi1：用整条链的 circular mean 作为 global
            phi_global  = _angle_global_mean(phi_s)
            psi_global  = _angle_global_mean(psi_s)
            omg_global  = _angle_global_mean(omg_s)
            chi_global  = _angle_global_mean(chi_s)
            global_row.extend([phi_global, psi_global, omg_global, chi_global])

            # ContactCount / ContactOrder 用 per-res 的平均（原逻辑保持不变）
            if contact_count.size:
                cc_mean = float(np.nanmean(contact_count))
            else:
                cc_mean = 0.0
            if contact_order.size:
                co_mean = float(np.nanmean(contact_order))
            else:
                co_mean = 0.0
            global_row.extend([cc_mean, co_mean])

            # extra_cols: 目前只可能有 B_factor
            if "B_factor" in enhanced_list:
                bf_mean = float(np.nanmean(np.asarray(b_vals, float))) if len(b_vals) else np.nan
                global_row.append(bf_mean)

            # (b) k 相关的列：优先用单链版 global，没有定义的再用 per-res 均值
            for k in ks:
                gstats = global_stats_by_k.get(k, {})  # 单链版得到的 global 量

                for name in K_DEP:
                    # 1) 先看单链版有没有给这个名字
                    if name in gstats:
                        val = gstats[name]
                    else:
                        # 2) 没有的话，用 per-res 的平均（你要求的 fallback）
                        arr = feats_by_k[k][name]
                        v = np.asarray(arr, float)
                        v = v[np.isfinite(v)]
                        val = float(v.mean()) if v.size else np.nan

                    # 3) 再兜底一下（避免 inf / nan），先尝试转成 float
                    try:
                        val_f = float(val)
                    except Exception:
                        val_f = np.nan

                    if not np.isfinite(val_f):
                        arr = feats_by_k[k][name]
                        v = np.asarray(arr, float)
                        v = v[np.isfinite(v)]
                        val_f = float(v.mean()) if v.size else 0.0

                    global_row.append(val_f)

                # CYloc_k? 单链版没有定义 → 直接用 per-res 的平均
                arr_cy = np.asarray(cyloc_by_k[k], float)
                vcy = arr_cy[np.isfinite(arr_cy)]
                cy_mean = float(vcy.mean()) if vcy.size else np.nan
                if not np.isfinite(cy_mean):
                    cy_mean = 0.0
                global_row.append(cy_mean)

            writer.writerow(global_row)

            # ===== 2B. 原来的 per-residue 行（完全不动） =====
            for i in range(N):
                resid_idx = i + 1      # 让残基从 1 开始编号（和 CY2 一致）
                phi_val = X[i, 0]
                psi_val = X[i, 1]
                chi_val = X[i, 2]
                omg_val = omg_s[i] if (i < len(omg_s) and omg_s[i] is not None) else np.nan

                # --- new: Chain / ResSeq / ResName ---
                res_i = residues[i] if i < len(residues) else None
                if res_i is not None:
                    try:
                        chain_i = getattr(res_i.get_parent(), "id", "?")
                    except Exception:
                        chain_i = "?"
                    try:
                        resseq_i = int(res_i.get_id()[1])
                    except Exception:
                        resseq_i = -1
                    try:
                        resname_i = str(res_i.get_resname())
                    except Exception:
                        resname_i = ""
                else:
                    chain_i, resseq_i, resname_i = "?", -1, ""

                row = [
                    resid_idx,
                    chain_i,
                    resseq_i,
                    resname_i,
                    phi_val,
                    psi_val,
                    omg_val,
                    chi_val,
                    float(contact_count[i]) if i < len(contact_count) else 0.0,
                    float(contact_order[i]) if i < len(contact_order) else 0.0,
                ]

                if "B_factor" in enhanced_list:
                    bf = b_vals[i] if i < len(b_vals) else np.nan
                    row.append(bf)

                for k in ks:
                    feats_k = feats_by_k[k]
                    row.extend([feats_k[c][i] for c in K_DEP])
                    row.append(cyloc_by_k[k][i])

                writer.writerow(row)

        return (pdb_path, True, "OK", dim_stats_local)

    except Exception as e:
        err = traceback.format_exc()
        return (pdb_path, False, str(e) + "\n" + err, dim_stats_local)


# =========================
# 4. main：并行调用 + 统计 d 分布
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ks", default="4,6,8,10,12")
    ap.add_argument("--pca-var", type=float, default=0.90)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument(
        "--enhanced",
        default="B_factor",
        help=("逗号分隔的增强特征名子集："
              "lag_p90,sl_im_mean,sl_im_p90,sl_phase_var,"
              "sl_im_p90_calib,sl_phase_var_calib,n_points,k_used,B_factor")
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--sl-auto-k", action="store_true",
                    help="启用自动扩 k，尽量保证 d_eig>=3 再算 SL 特征")
    ap.add_argument("--dim-stats-out", default="dim_stats.csv",
                    help="维度分布统计输出文件名（默认在 out-dir 下）")
    ap.add_argument("--contacts-cutoff", type=float, default=8.0,
                    help="CA-CA 接触距离 cutoff（Å），用于 ContactCount/ContactOrder")
    ap.add_argument("--grid-deg", type=float, default=1.0,
                    help="theta* 搜索时的角度步长（度）")
    ap.add_argument(
        "--chain",
        default="ALL",
        help="链选择: 'ALL' 使用所有标准 AA 链; "
             "'LONGEST' 使用最长标准 AA 链(旧行为); "
             "或指定某条链如 'A'。"
    )

    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    enhanced_list = [x.strip() for x in args.enhanced.split(",") if x.strip()]

    pdb_list = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_list:
        print(f"[WARN] no pdb found in: {args.pdb_dir}")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    max_k_auto = max(ks)

    worker = partial(
        process_one,
        out_dir=args.out_dir,
        ks=ks,
        pca_var=args.pca_var,
        window=args.window,
        enhanced_list=enhanced_list,
        sl_auto_k=args.sl_auto_k,
        max_k_auto=max_k_auto,
        contacts_cutoff=args.contacts_cutoff,
        grid_deg=args.grid_deg,
        chain_mode=args.chain,
    )

    global_dim_stats = {}
    errors = []

    # === 单进程模式：jobs <= 1 时，不用进程池，便于调试 ===
    if args.jobs <= 1:
        print(f"[INFO] Running in single-process mode (jobs={args.jobs})")
        for p in pdb_list:
            pdb_path, ok, msg, dim_stats_local = worker(p)
            base = os.path.basename(pdb_path)
            if ok:
                print(f"[OK]  {base}: {msg}")
            else:
                first_line = str(msg).splitlines()[0] if msg else ""
                print(f"[ERR] {base}: {first_line}", file=sys.stderr)
                errors.append((base, msg))

            for k, dmap in dim_stats_local.items():
                global_dim_stats.setdefault(k, {})
                for d, cnt in dmap.items():
                    global_dim_stats[k][d] = global_dim_stats[k].get(d, 0) + cnt

    # === 多进程模式：jobs > 1 时，用 ProcessPoolExecutor（默认 fork） ===
    else:
        print(f"[INFO] Running with ProcessPoolExecutor (jobs={args.jobs})")

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(worker, p) for p in pdb_list]

            for fu in as_completed(futs):
                try:
                    pdb_path, ok, msg, dim_stats_local = fu.result()
                except Exception as e:
                    # 这里连 pdb_path 都取不到，只能标成 Unknown
                    pdb_path = "UNKNOWN"
                    ok = False
                    msg = repr(e)
                    dim_stats_local = {}

                base = os.path.basename(pdb_path)
                if ok:
                    print(f"[OK]  {base}: {msg}")
                else:
                    first_line = str(msg).splitlines()[0] if msg else ""
                    print(f"[ERR] {base}: {first_line}", file=sys.stderr)
                    errors.append((base, msg))

                for k, dmap in dim_stats_local.items():
                    global_dim_stats.setdefault(k, {})
                    for d, cnt in dmap.items():
                        global_dim_stats[k][d] = global_dim_stats[k].get(d, 0) + cnt

    dim_stats_path = os.path.join(args.out_dir, args.dim_stats_out)
    with open(dim_stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "dim_eig", "count", "fraction"])
        for k in sorted(global_dim_stats.keys()):
            dmap = global_dim_stats[k]
            total = sum(dmap.values()) if dmap else 0
            for d in sorted(dmap.keys()):
                cnt = dmap[d]
                frac = cnt / total if total > 0 else 0.0
                writer.writerow([k, d, cnt, frac])

    print(f"[DONE] out_dir={args.out_dir}, files={len(pdb_list)}, errors={len(errors)}")
    print(f"[INFO] dim stats written to: {dim_stats_path}")

    # 如果有出错的 PDB，把详细错误写到单独的日志文件里
    if errors:
        err_log_path = os.path.join(args.out_dir, "cy1_errors.log")
        with open(err_log_path, "w", encoding="utf-8") as f:
            for base, msg in errors:
                f.write(f"=== {base} ===\n")
                f.write(str(msg).rstrip() + "\n")
                f.write("=" * 80 + "\n\n")
        print(f"[WARN] {len(errors)} PDBs failed; see {err_log_path} for full tracebacks")

if __name__ == "__main__":
    main()

