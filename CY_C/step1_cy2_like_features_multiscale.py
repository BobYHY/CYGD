#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从一个目录中的所有 PDB 文件（*.pdb）构造“多尺度 CY-like” 残基层面特征，
并在需要时对每个 PDB 画一张 CY-like 网格图。

数值部分（和旧版一致）：
- 对每个 PDB（可选指定 chain）：
  - 用 N, CA, C, O 坐标构造每个残基的 12 维嵌入向量；
  - 对给定的多个 k（--ks，例如 10,20,30）：
      - 用 k 近邻估计局部协方差度量 G_i 和切空间 T_i（tan-dim）；
      - 计算 integrability_err, kahler_err, omega_norm, logdetG, ricci_proxy。
  - 输出 basename.csv 到 --output 目录下：
      ResidueIndex, Chain, ResSeq, ResName, B_avg,
      integrability_err_k10, kahler_err_k10, ..., ricci_proxy_k10,
      integrability_err_k20, ..., ricci_proxy_k20, ...
  - 第一行为全局平均特征（ResidueIndex=-1, Chain='GLOBAL', ResSeq=-1, ResName='GLOBAL'）。

可视化部分（新加）：
- 通过 --plot yes 开启。
- 从残基层（ResidueIndex > 0）选一条列序列作为“色度场（Chromatic Field）”，用于着色；
- 选一条列序列作为 CY_Score 的打分源（score-source），可以与色度场相同或不同；
- 把打分源的均值/方差映射到 [0,1]，得到 CY_Score；
- 使用色度场的“局部相位能量”调制 CY-like 管状网格的扭曲与粗细，生成一张三维图。
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# ---------- 参数解析 ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute multi-scale CY-like features for all PDBs in a directory, with optional CY-like visualization."
    )
    p.add_argument("--input", required=True,
                   help="Input directory containing *.pdb files.")
    p.add_argument("--output", required=True,
                   help="Output directory to store CSVs.")
    p.add_argument("--chain", default="ALL",
               help="Chain ID to use (e.g. A) or ALL for all chains (default: ALL).")
    p.add_argument("--ks", default="10,20,30",
                   help="Comma-separated list of neighbor sizes, e.g. '10,20,30'.")
    p.add_argument("--tan-dim", type=int, default=4,
                   help="Local tangent dimension for PCA (single-scale, default: 4).")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Diagonal regularization for local metric (default: 1e-6).")
    # 新增：接触定义的 cutoff
    p.add_argument("--contacts-cutoff", type=float, default=8.0,
                   help="CA–CA distance cutoff in Å for defining contacts (default: 8.0).")
    # 绘图控制
    p.add_argument("--plot", default="no", choices=["yes", "no"],
                   help="Whether to render a CY-like mesh per PDB (default: no).")
    p.add_argument("--color-col", default="auto",
                   help="Column name used as Chromatic Field (per-residue); "
                        "default 'auto' will choose from integrability_err_k*, "
                        "kahler_err_k*, ricci_proxy_k*, omega_norm_k*, logdetG_k*.")
    p.add_argument("--score-source", default="auto",
                   help="Column used for CY_Score; default 'auto' → same as color-col.")
    p.add_argument("--score-mode", default="pos01", choices=["pos01", "neg"],
                   help="CY_Score mode: pos01→[0,1] higher better, neg→legacy negative score.")
    p.add_argument("--cmap", default="viridis",
                   help="Colormap for the mesh (default: viridis).")
    p.add_argument("--lw", type=float, default=0.7,
                   help="Line width for mesh tubes.")
    p.add_argument("--tau-alpha", type=float, default=0.3,
                   help="Strength of twisting modulation from phase energy.")
    p.add_argument("--radius-alpha", type=float, default=0.08,
                   help="Strength of radius modulation from phase energy.")
    return p.parse_args()

def parse_pdb_backbone_with_fallback(pdb_path, chain_id=None):
    """
    解析 PDB，并提供两种模式：
      - 'full':  有 N,CA,C,O 的主链残基；
      - 'ca':    如果没有任何 full-backbone 残基，则回退到 Cα-only 模式；
      - 'none':  既没有 full-backbone，也没有 CA（极端情况）。

    这里做了两个最小修改：
      1) altloc 接受 "", "A", "B"（9bts_a.pdb 里是 'B'）；
      2) 链名比较改成大小写不敏感（chain / chain_id 都转成大写）。
    """
    backbone_atoms = {"N", "CA", "C", "O"}

    res_map = {}
    res_order = []

    # 统一链名大小写
    chain_id_up = str(chain_id).upper() if chain_id is not None else None

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            altloc = line[16].strip()

            # ✅ 修改 1：接受 B 构象（9bts_a.pdb 只有 B）
            #if altloc not in ("", "A", "B"):
            if altloc not in ("", "A", "B", "C"):
                continue

            resname = line[17:20].strip()
            chain = line[21].strip() or "?"
            chain_up = chain.upper()

            resseq_str = line[22:26]
            icode = line[26].strip()

            # ✅ 修改 2：链名大小写不敏感比较
            if (chain_id_up is not None) and (chain_id_up != "ALL") and (chain_up != chain_id_up):
                continue

            try:
                resseq = int(resseq_str)
            except ValueError:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66])
            except ValueError:
                continue

            key = (chain, resseq, icode)
            if key not in res_map:
                res_map[key] = {
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "atoms": {}
                }
                res_order.append(key)

            res_entry = res_map[key]
            if atom_name not in res_entry["atoms"]:
                res_entry["atoms"][atom_name] = ((x, y, z), b)

    # full-backbone 模式
    residues_full = []
    for key in res_order:
        entry = res_map[key]
        atoms = entry["atoms"]
        if not all(a in atoms for a in backbone_atoms):
            continue

        coords = np.zeros((4, 3), dtype=float)
        b_factors = np.zeros(4, dtype=float)
        for idx, a in enumerate(["N", "CA", "C", "O"]):
            (x, y, z), b = atoms[a]
            coords[idx] = (x, y, z)
            b_factors[idx] = b

        residues_full.append({
            "chain": entry["chain"],
            "resseq": entry["resseq"],
            "icode": entry["icode"],
            "resname": entry["resname"],
            "coords": coords,
            "b_factors": b_factors
        })

    if len(residues_full) > 0:
        return residues_full, "full"

    # CA-only fallback（这里不需要额外改动，沿用原来逻辑即可）
    ca_entries = []
    for key in res_order:
        entry = res_map[key]
        atoms = entry["atoms"]
        if "CA" not in atoms:
            continue
        (x, y, z), b = atoms["CA"]
        ca_entries.append({
            "chain": entry["chain"],
            "resseq": entry["resseq"],
            "icode": entry["icode"],
            "resname": entry["resname"],
            "coord": np.array([x, y, z], dtype=float),
            "b": float(b),
        })

    if len(ca_entries) == 0:
        return [], "none"

    residues_ca = []
    m = len(ca_entries)
    for i, ent in enumerate(ca_entries):
        prev_coord = ca_entries[i - 1]["coord"] if i > 0 else ent["coord"]
        next_coord = ca_entries[i + 1]["coord"] if i < m - 1 else ent["coord"]
        this_coord = ent["coord"]

        coords = np.vstack([prev_coord, this_coord, next_coord, this_coord])
        b_val = ent["b"]
        b_factors = np.full(4, b_val, dtype=float)

        residues_ca.append({
            "chain": ent["chain"],
            "resseq": ent["resseq"],
            "icode": ent["icode"],
            "resname": ent["resname"],
            "coords": coords,
            "b_factors": b_factors
        })

    return residues_ca, "ca"
#N–CA / C–O 配对
# ---------- 几何构造部分 ----------
def build_embedding_from_residues(residues):
    """
    residues 列表 -> X (n, 12), B_avg (n,)

    约定：
      coords[i] 顺序为 N, CA, C, O，每个为 3 维 (x,y,z)，先做平移中心化。
      然后在 12 维实向量中按如下次序排列：

        [ N_x,  CA_x,
          N_y,  CA_y,
          N_z,  CA_z,
          C_x,  O_x,
          C_y,  O_y,
          C_z,  O_z ]

      在 build_standard_J(12) 生成的 J 下，
      6 对复坐标依次为：

        z1 = N_x  + i CA_x
        z2 = N_y  + i CA_y
        z3 = N_z  + i CA_z
        z4 = C_x  + i O_x
        z5 = C_y  + i O_y
        z6 = C_z  + i O_z

      即：x 对 x, y 对 y, z 对 z，并且 (N, CA) 与 (C, O) 成对。
    """
    n = len(residues)
    X = np.zeros((n, 12), dtype=float)
    B_avg = np.zeros(n, dtype=float)

    for i, res in enumerate(residues):
        coords = res["coords"].astype(float)        # (4, 3): N, CA, C, O
        b_factors = res["b_factors"].astype(float)  # (4,)

        # 居中，去掉刚体平移
        center = coords.mean(axis=0, keepdims=True)  # (1, 3)
        centered = coords - center                   # (4, 3)

        # 拆出 4 个原子
        N_xyz  = centered[0]  # (x_N,  y_N,  z_N)
        CA_xyz = centered[1]  # (x_CA, y_CA, z_CA)
        C_xyz  = centered[2]  # (x_C,  y_C,  z_C)
        O_xyz  = centered[3]  # (x_O,  y_O,  z_O)

        # 注意顺序：N–CA / C–O，且 x-x, y-y, z-z
        X[i, :] = np.array([
            N_xyz[0],  CA_xyz[0],   # N_x,  CA_x
            N_xyz[1],  CA_xyz[1],   # N_y,  CA_y
            N_xyz[2],  CA_xyz[2],   # N_z,  CA_z
            C_xyz[0],  O_xyz[0],    # C_x,  O_x
            C_xyz[1],  O_xyz[1],    # C_y,  O_y
            C_xyz[2],  O_xyz[2],    # C_z,  O_z
        ], dtype=float)

        B_avg[i] = b_factors.mean()

    return X, B_avg
'''
#CA–C / N–O
def build_embedding_from_residues(residues):
    """
    residues 列表 -> X (n, 12), B_avg (n,)

    约定：
      coords[i] 顺序为 N, CA, C, O，每个为 3 维 (x,y,z)，先做平移中心化。
      然后在 12 维实向量中按如下次序排列：

        [CA_x, C_x,
         CA_y, C_y,
         CA_z, C_z,
         N_x,  O_x,
         N_y,  O_y,
         N_z,  O_z]

      这样在 build_standard_J(12) 生成的 J 下，
      6 对复坐标依次为：

        z1 = CA_x + i C_x
        z2 = CA_y + i C_y
        z3 = CA_z + i C_z
        z4 = N_x  + i O_x
        z5 = N_y  + i O_y
        z6 = N_z  + i O_z

      即：x 对 x, y 对 y, z 对 z，并且 (CA, C) 与 (N, O) 成对。
    """
    n = len(residues)
    X = np.zeros((n, 12), dtype=float)
    B_avg = np.zeros(n, dtype=float)

    for i, res in enumerate(residues):
        coords = res["coords"].astype(float)        # (4, 3) → N, CA, C, O
        b_factors = res["b_factors"].astype(float)  # (4,)

        # 居中（去掉刚体平移）
        center = coords.mean(axis=0, keepdims=True)  # (1, 3)
        centered = coords - center                   # (4, 3)

        # 拆出 4 个原子
        N_xyz  = centered[0]  # (x_N,  y_N,  z_N)
        CA_xyz = centered[1]  # (x_CA, y_CA, z_CA)
        C_xyz  = centered[2]  # (x_C,  y_C,  z_C)
        O_xyz  = centered[3]  # (x_O,  y_O,  z_O)

        # 重新按“x-x, y-y, z-z + 原子成对”的顺序拼成 12 维向量
        X[i, :] = np.array([
            CA_xyz[0], C_xyz[0],   # CA_x, C_x
            CA_xyz[1], C_xyz[1],   # CA_y, C_y
            CA_xyz[2], C_xyz[2],   # CA_z, C_z
            N_xyz[0],  O_xyz[0],   # N_x,  O_x
            N_xyz[1],  O_xyz[1],   # N_y,  O_y
            N_xyz[2],  O_xyz[2],   # N_z,  O_z
        ], dtype=float)

        B_avg[i] = b_factors.mean()

    return X, B_avg
'''
'''
def build_embedding_from_residues(residues):
    """
    residues 列表 -> X (n, 12), B_avg (n,)
    """
    n = len(residues)
    X = np.zeros((n, 12), dtype=float)
    B_avg = np.zeros(n, dtype=float)

    for i, res in enumerate(residues):
        coords = res["coords"].astype(float)        # (4, 3)
        b_factors = res["b_factors"].astype(float)  # (4,)

        center = coords.mean(axis=0, keepdims=True)  # (1, 3)
        centered = coords - center                   # (4, 3)
        X[i, :] = centered.reshape(-1)               # (12,)

        B_avg[i] = b_factors.mean()

    return X, B_avg
'''
def compute_contacts_and_order(residues, cutoff=8.0):
    """
    基于每个残基的“中心 CA 点”计算：
      - ContactCount[i] : 在 cutoff 以内的非相邻残基个数
      - ContactOrder[i] : 这些接触残基的平均序列间隔（同链用 ResSeq 差，不同链退化为索引差）

    这里使用 residues[i]["coords"][1] 作为 CA 点：
      - full 模式下是 N,CA,C,O -> 下标 1 正好是 CA；
      - ca 模式下 coords = [CA_{i-1}, CA_i, CA_{i+1}, CA_i]，
        下标 1 仍然是当前残基的 CA 代理。
    """
    n = len(residues)
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    pts = np.zeros((n, 3), dtype=float)
    chains = []
    resseqs = []
    for i, res in enumerate(residues):
        coords = res["coords"].astype(float)
        ca = coords[1]                 # 中心 CA
        pts[i] = ca
        chains.append(res["chain"])
        resseqs.append(res["resseq"])

    contact_count = np.zeros(n, dtype=float)
    contact_order = np.zeros(n, dtype=float)

    for i in range(n):
        pi = pts[i]
        local_orders = []
        ci = 0
        for j in range(n):
            if i == j:
                continue
            # 排除同链上的序列邻居（i,i±1）
            if chains[i] == chains[j] and abs(resseqs[i] - resseqs[j]) <= 1:
                continue
            d = np.linalg.norm(pi - pts[j])
            if d <= cutoff:
                ci += 1
                if chains[i] == chains[j]:
                    seqsep = abs(resseqs[i] - resseqs[j])
                else:
                    # 跨链时用索引差作为一个粗略的“顺序距离”
                    seqsep = abs(i - j)
                local_orders.append(seqsep)

        contact_count[i] = ci
        contact_order[i] = np.mean(local_orders) if local_orders else 0.0

    return contact_count, contact_order

def build_standard_J(dim):
    """
    在 R^{dim} 上构造标准复结构矩阵 J（dim = 2n）。
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even to build a complex structure J.")
    n = dim // 2
    J = np.zeros((dim, dim), dtype=float)
    for k in range(n):
        i = 2 * k
        J[i, i + 1] = -1.0
        J[i + 1, i] = 1.0
    return J


def knn_indices(X, k):
    """
    朴素 O(n^2) kNN: 每个点的 k 近邻（不含自身）。
    返回 neighbors: (n, k) int 索引数组。
    """
    n, d = X.shape
    diff = X[:, None, :] - X[None, :, :]   # (n, n, d)
    D2 = np.sum(diff * diff, axis=-1)      # (n, n)

    neighbors = np.zeros((n, k), dtype=int)
    for i in range(n):
        idx_sorted = np.argsort(D2[i])
        neighbors[i] = idx_sorted[1:k + 1]  # 跳过自身
    return neighbors


def local_pca_tangent(Y, tan_dim):
    """
    对邻域点集 Y (m, d) 做 PCA，返回主方向 T (d, t)。
    """
    m, d = Y.shape
    if m == 0:
        return None

    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    rank = np.sum(S > 1e-12)
    t = min(tan_dim, rank, d)
    if t == 0:
        return None
    T = Vt[:t, :].T  # (d, t)
    return T


def compute_local_features_multiscale(X, ks, tan_dim=4, eps=1e-6):
    """
    对点云 X (n, d) 和多个 k 值 ks 计算多尺度 CY-like 特征。

    返回：
        features[k] = {
            'integrability_err': (n,),
            'kahler_err': (n,),
            'omega_norm': (n,),
            'logdetG': (n,),
            'ricci_proxy': (n,)
        }
    """
    n, d = X.shape
    if n <= 1:
        raise ValueError("Not enough residues (n <= 1) to build local geometry.")

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    if not ks:
        raise ValueError("ks must contain at least one positive integer.")

    Kmax = min(max(ks), n - 1)
    if Kmax < 1:
        raise ValueError("All ks are >= n, cannot build neighbors. Reduce ks or use longer chains.")

    neighbors_all = knn_indices(X, Kmax)
    J = build_standard_J(d)

    features = {}

    for k in ks:
        k_eff = min(k, n - 1)
        nb = neighbors_all[:, :k_eff]  # (n, k_eff)

        integrability_err = np.full(n, np.nan, dtype=float)
        kahler_err = np.full(n, np.nan, dtype=float)
        omega_norm = np.full(n, np.nan, dtype=float)
        logdetG = np.full(n, np.nan, dtype=float)

        for i in range(n):
            nb_idx = nb[i]
            Y = X[nb_idx]  # (k_eff, d)

            Yc = Y - Y.mean(axis=0, keepdims=True)
            m = Yc.shape[0]

            if m < 2:
                G_i = np.eye(d, dtype=float)
                integrability_err[i] = np.nan
                JGJ = J.T @ G_i @ J
                num_k = np.linalg.norm(JGJ - G_i, ord="fro")
                den_k = np.linalg.norm(G_i, ord="fro")
                kahler_err[i] = num_k / den_k if den_k > 0 else np.nan
                Omega = G_i @ J
                omega_norm[i] = np.linalg.norm(Omega, ord="fro")
                sign, logdet = np.linalg.slogdet(G_i + eps * np.eye(d))
                logdetG[i] = logdet
                continue

            G_i = (Yc.T @ Yc) / float(m - 1)
            G_i = G_i + eps * np.eye(d)

            T_i = local_pca_tangent(Y, tan_dim)
            if T_i is None:
                integrability_err[i] = np.nan
            else:
                JT = J @ T_i
                M = T_i.T @ T_i
                try:
                    M_inv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    M_inv = np.linalg.pinv(M)
                P = T_i @ M_inv @ T_i.T
                R = (np.eye(d) - P) @ JT
                num = np.linalg.norm(R, ord="fro")
                den = np.linalg.norm(JT, ord="fro")
                integrability_err[i] = num / den if den > 0 else np.nan

            JGJ = J.T @ G_i @ J
            num_k = np.linalg.norm(JGJ - G_i, ord="fro")
            den_k = np.linalg.norm(G_i, ord="fro")
            kahler_err[i] = num_k / den_k if den_k > 0 else np.nan

            Omega = G_i @ J
            omega_norm[i] = np.linalg.norm(Omega, ord="fro")

            sign, logdet = np.linalg.slogdet(G_i)
            if sign <= 0:
                G_reg = G_i + eps * np.eye(d)
                sign2, logdet2 = np.linalg.slogdet(G_reg)
                logdetG[i] = logdet2
            else:
                logdetG[i] = logdet

        ricci_proxy = np.full(n, np.nan, dtype=float)
        for i in range(n):
            s_i = logdetG[i]
            nb_idx = nb[i]
            s_nb = logdetG[nb_idx]
            s_nb = s_nb[~np.isnan(s_nb)]
            if np.isnan(s_i) or s_nb.size == 0:
                ricci_proxy[i] = np.nan
            else:
                ricci_proxy[i] = abs(s_i - s_nb.mean())

        features[k] = {
            "integrability_err": integrability_err,
            "kahler_err": kahler_err,
            "omega_norm": omega_norm,
            "logdetG": logdetG,
            "ricci_proxy": ricci_proxy,
        }

    return features


# ---------- 可视化相关辅助 ----------

def local_phase_energy(x, smooth_win=11):
    """
    给定一条实数序列 x，构造一个“相位能量” E，用于调制 CY 网格的扭曲/粗细。
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([])
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        E = np.zeros_like(x)
    else:
        xn = (x - xmin) / (xmax - xmin)
        phase = np.unwrap(2 * np.pi * xn)
        g = np.gradient(phase)
        E = g ** 2
        if smooth_win and smooth_win > 1:
            k = np.ones(smooth_win, float)
            k /= k.sum()
            E = np.convolve(E, k, mode="same")
        Emin, Emax = E.min(), E.max()
        if Emax - Emin < 1e-12:
            E = np.zeros_like(E)
        else:
            E = (E - Emin) / (Emax - Emin)
    return E

def normalize_01_per_protein(vals):
    """
    对一个一维数组做 per-protein 的 [0,1] 归一化。

    返回:
      v_norm: 归一化后的数组（只包含原来的有限值）
      vmin, vmax: 原数组的 (min, max)，便于调试/记录
    """
    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, np.nan, np.nan
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        # 全部几乎一样 → 归一化后设成全 0
        return np.zeros_like(arr), vmin, vmax
    v_norm = (arr - vmin) / (vmax - vmin)
    return v_norm, vmin, vmax

def choose_color_series_coord(df, ks, prefer="auto"):
    """
    从多尺度 CSV 中选择一列作为“色度场（Chromatic Field）”。
    - 仅使用 ResidueIndex > 0 的行；
    - prefer != 'auto' 时，直接用该列（若存在且有 finite 值）；
    - auto 时，按以下顺序尝试（优先最大 k）：
        integrability_err_k*, kahler_err_k*, ricci_proxy_k*, omega_norm_k*, logdetG_k*。
    返回： (series: np.ndarray, col_name or None)
    """
    mask = df["ResidueIndex"] > 0

    if prefer != "auto" and prefer in df.columns:
        s = pd.to_numeric(df.loc[mask, prefer], errors="coerce").to_numpy(float)
        s = s[np.isfinite(s)]
        if s.size > 0:
            return s, prefer

    # 从列名中解析出 ks
    all_cols = list(df.columns)
    cand_feats = ["integrability_err", "kahler_err", "ricci_proxy", "omega_norm", "logdetG"]

    # 从列名解析出所有 k
    found_ks = set()
    for c in all_cols:
        for f in cand_feats:
            if c.startswith(f + "_k"):
                try:
                    kval = int(c.split("_k", 1)[1])
                    found_ks.add(kval)
                except Exception:
                    pass
    if not found_ks:
        return np.array([]), None

    ks_sorted = sorted(found_ks, reverse=True)

    for f in cand_feats:
        for k in ks_sorted:
            col = f"{f}_k{k}"
            if col not in df.columns:
                continue
            s = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if s.size > 0:
                return s, col

    return np.array([]), None

def build_mesh_stats_from_series(series, mode="pos01"):
    """
    给定一条序列（score-source），构造 CY-style mesh 的全局统计字典。

    关键改动：先对该序列做 per-protein 的 min-max 归一化，再用
    归一化后的 mean / var 去驱动 CY_Score：
      - stats["lag_mean"] : 归一化后的均值（越小越“CY-like”）
      - stats["lag_p90"]  : 归一化后的 0.9 分位
      - stats["slvar"]    : 归一化后的方差（越小越“平滑”）

    同时保留原始序列的简单统计，便于之后分析 / 导出：
      - stats["raw_mean"], stats["raw_var"], stats["raw_min"], stats["raw_max"]
    """
    xs = np.asarray(series, float)
    xs = xs[np.isfinite(xs)]
    stats = {}

    if xs.size == 0:
        stats["lag_mean"] = np.nan
        stats["lag_p90"]  = np.nan
        stats["slvar"]    = np.nan
        stats["raw_mean"] = np.nan
        stats["raw_var"]  = np.nan
        stats["raw_min"]  = np.nan
        stats["raw_max"]  = np.nan
        stats["theta"]    = 0.0
        return stats

    # 先按蛋白内部 min-max 归一化，再算均值/方差
    xs_norm, vmin, vmax = normalize_01_per_protein(xs)

    stats["lag_mean"] = float(np.nanmean(xs_norm))
    stats["lag_p90"]  = float(np.quantile(xs_norm, 0.90))
    stats["slvar"]    = float(np.nanvar(xs_norm))

    # 顺便记录原始统计信息（可选，用不到也没关系）
    stats["raw_mean"] = float(np.nanmean(xs))
    stats["raw_var"]  = float(np.nanvar(xs))
    stats["raw_min"]  = float(vmin)
    stats["raw_max"]  = float(vmax)

    # 目前没有显式 theta，相当于 0
    stats["theta"] = 0.0
    return stats

def cy_score(stats, mode="pos01"):
    """
    根据 lag_mean 和 slvar 合成一个 CY_Score。
    - 这里假设“指标越小越好”（比如 error 类指标），
      所以用一个简单的 1 - x 映射（截断到 [0,1]）。
    """
    lm = stats.get("lag_mean", np.nan)
    sv = stats.get("slvar", np.nan)
    if not np.isfinite(lm) or not np.isfinite(sv):
        return np.nan

    def map01_small_is_better(x):
        if not np.isfinite(x):
            return np.nan
        if x <= 0:
            return 1.0
        if x >= 1.0:
            return 0.0
        return float(1.0 - x)

    a = map01_small_is_better(lm)
    b = map01_small_is_better(sv)
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan

    if mode == "neg":
        return -10.0 * ((1 - a) + (1 - b))
    else:
        s = (a + b) / 2.0
        return float(np.clip(s, 0.0, 1.0))


def render_mesh(title, stats, chrom_field, out_png, lw=0.7, cmap_name="viridis",
                tau_alpha=0.3, radius_alpha=0.08, pos_score=None,
                colorbar_label="Chromatic Field (0..1)"):
    """
    画 CY-like 网格：
    - R：由 lag_mean 控制整体半径；
    - r：由 lag_mean 控制管子粗细（这里也用 lag_mean；可按需改成其它全局量）；
    - tau_base：由 slvar 决定基本扭曲强度；
    - chrom_field：色度场（Chromatic Field）→ 相位能量 E → 调制 tau_u, r_u 与颜色。
    """
    lm = stats.get("lag_mean", np.nan)
    sv = stats.get("slvar", np.nan)
    th = stats.get("theta", 0.0)

    R = 1.6 + 0.6 * np.tanh(np.nan_to_num(lm, nan=0.2))
    r = 0.55 + 0.35 * np.exp(-np.nan_to_num(lm, nan=0.1) * 3.0)

    svn = np.nan_to_num(sv, nan=0.2)
    tau_base = 1.0 + 3.0 * (svn / (svn + 1.0)) + 0.5 * np.cos(np.nan_to_num(th, nan=0.0))

    U, V = 220, 90
    u = np.linspace(0, 2 * np.pi, U, endpoint=False)
    v = np.linspace(0, 2 * np.pi, V, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    if chrom_field is not None and len(chrom_field) > 0:
        E = local_phase_energy(chrom_field)
    else:
        E = np.array([])

    if E.size > 0:
        eU = np.interp(np.linspace(0, len(E) - 1, U), np.arange(len(E)), E)
        eU = eU - np.mean(eU)
        tau_u = tau_base + tau_alpha * eU
        r_u = r * (1.0 + radius_alpha * eU)
    else:
        tau_u = np.full(U, tau_base)
        r_u = np.full(U, r)

    w = 0.0
    if chrom_field is not None and len(chrom_field) > 0:
        cs = np.asarray(chrom_field, float)
        cs = cs[np.isfinite(cs)]
        if cs.size:
            cmin, cmax = cs.min(), cs.max()
            cstrip_src = (cs - cmin) / (cmax - cmin + 1e-9)
            w = 0.15 * (np.interp(np.linspace(0, len(cstrip_src) - 1, U),
                                  np.arange(len(cstrip_src)), cstrip_src) - 0.5)
        else:
            w = np.zeros(U)
    else:
        w = np.zeros(U)

    Rterm = r_u[:, None] * (1 + w[:, None]) * np.cos(vv + tau_u[:, None] * uu)
    X = (R + Rterm) * np.cos(uu)
    Y = (R + Rterm) * np.sin(uu)
    Z = r_u[:, None] * (1 + w[:, None]) * np.sin(vv + 0.65 * np.sin(tau_u[:, None] * uu))

    if chrom_field is not None and len(chrom_field) > 0:
        cs = np.asarray(chrom_field, float)
        cs = cs[np.isfinite(cs)]
        if cs.size:
            cmin, cmax = cs.min(), cs.max()
            cvals = (cs - cmin) / (cmax - cmin + 1e-9)
            cstrip = np.interp(np.linspace(0, len(cvals) - 1, U),
                               np.arange(len(cvals)), cvals)
        else:
            cstrip = np.linspace(0, 1, U)
    else:
        cstrip = np.linspace(0, 1, U)

    cmap = get_cmap(cmap_name)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    for i in range(U):
        col = cmap(cstrip[i])
        ax.plot(X[i, :], Y[i, :], Z[i, :], lw=lw, color=col, alpha=0.95)
    for j in range(0, V, 2):
        col = cmap(cstrip[j % U])
        ax.plot(X[:, j], Y[:, j], Z[:, j], lw=lw, color=col, alpha=0.6)

    ax.set_axis_off()
    ax.view_init(elev=24, azim=30)

    full_title = title
    if pos_score is not None and np.isfinite(pos_score):
        full_title += f"  |  CY_Score={pos_score:.3f}"
    plt.title(full_title, color="white")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0.0, 1.0)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(colorbar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

    plt.tight_layout()
    fig.savefig(out_png, facecolor="black", dpi=180)
    plt.close(fig)


# ---------- 主流程 ----------
def process_one_pdb(pdb_path, out_path, chain_id, ks, tan_dim, eps, args):
    residues, mode = parse_pdb_backbone_with_fallback(pdb_path, chain_id=chain_id)

    base = os.path.basename(pdb_path)

    if len(residues) == 0:
        print(f"[WARN] {base}: no backbone or CA atoms found (for chain filter), skipped.")
        return

    if mode == "ca":
        print(f"[WARN] {base}: no full N/CA/C/O backbone; using CA-only fallback.")

    X, B_avg = build_embedding_from_residues(residues)
    # 新增：基于 CA 点计算接触数和接触序
    contact_count, contact_order = compute_contacts_and_order(
        residues, cutoff=args.contacts_cutoff
    )

    features = compute_local_features_multiscale(X, ks, tan_dim=tan_dim, eps=eps)

    n = len(residues)
    ks_sorted = sorted(set(int(k) for k in ks if int(k) > 0))
    feat_names = ["integrability_err", "kahler_err", "omega_norm", "logdetG", "ricci_proxy"]

    base_cols = ["ResidueIndex", "Chain", "ResSeq", "ResName",
                 "B_avg", "ContactCount", "ContactOrder"]

    scale_cols = []
    for k in ks_sorted:
        for f in feat_names:
            scale_cols.append(f"{f}_k{k}")
    all_cols = base_cols + scale_cols

    rows = []

    # 全局行
    global_row = {
        "ResidueIndex": -1,
        "Chain": "GLOBAL",
        "ResSeq": -1,
        "ResName": "GLOBAL",
        "B_avg": float(np.nanmean(B_avg)),
        "ContactCount": float(np.nanmean(contact_count)) if contact_count.size else np.nan,
        "ContactOrder": float(np.nanmean(contact_order)) if contact_order.size else np.nan,
    }
    
    for k in ks_sorted:
        fdict = features[k]
        for f in feat_names:
            key = f"{f}_k{k}"
            global_row[key] = float(np.nanmean(fdict[f]))
    rows.append(global_row)

    # 残基层面
    for idx, res in enumerate(residues):
        row = {
            "ResidueIndex": idx + 1,
            "Chain": res["chain"],
            "ResSeq": res["resseq"],
            "ResName": res["resname"],
            "B_avg": float(B_avg[idx]),
            "ContactCount": float(contact_count[idx]),
            "ContactOrder": float(contact_order[idx]),
        }
    
        for k in ks_sorted:
            fdict = features[k]
            for f in feat_names:
                key = f"{f}_k{k}"
                row[key] = float(fdict[f][idx])
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] {base} ({mode}) -> {out_path} (n_res={n})")

    # ---------- 可选绘图 ----------
    if args.plot.lower() != "yes":
        return

    mask = df["ResidueIndex"] > 0
    feat_priority = ["integrability_err", "kahler_err", "ricci_proxy", "omega_norm", "logdetG"]

    def _pick_col_for_k(kval: int, prefer: str):
        """
        为指定 k 选取色度场列：
        - prefer != auto: 若用户给的是带 _k 的列名且存在，直接用；
                         若用户给的是不带 _k 的基名（如 integrability_err），则尝试补成 *_k{kval}；
        - prefer == auto: 按优先级在 *_k{kval} 中找第一个存在且有 finite 值的列。
        返回 (series, colname)；若失败返回 (empty, None)
        """
        # 1) prefer 非 auto
        if prefer != "auto":
            p = prefer.strip()
            cand = None
            if p in df.columns:
                cand = p
            else:
                # 允许用户写 integrability_err 这种“基名”
                p2 = f"{p}_k{kval}"
                if p2 in df.columns:
                    cand = p2

            if cand is not None:
                s = pd.to_numeric(df.loc[mask, cand], errors="coerce").to_numpy(float)
                s = s[np.isfinite(s)]
                if s.size > 0:
                    return s, cand

        # 2) auto：按优先级找 *_k{kval}
        for f in feat_priority:
            col = f"{f}_k{kval}"
            if col not in df.columns:
                continue
            s = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if s.size > 0:
                return s, col

        return np.array([]), None

    # 对每个 k 都画一张（核心改动就在这里）
    for k_draw in ks_sorted:
        chrom_series, chrom_col = _pick_col_for_k(k_draw, args.color_col)
        if chrom_series.size == 0 or chrom_col is None:
            print(f"[WARN] {base}: no usable Chromatic Field for k={k_draw}; skip plotting this k.")
            continue

        # score-source：auto 就跟随 chrom_col；否则同样允许“基名→补 _k{k}”
        if args.score_source == "auto" or not args.score_source.strip():
            score_col = chrom_col
        else:
            ssrc = args.score_source.strip()
            if ssrc in df.columns:
                score_col = ssrc
            else:
                ssrc2 = f"{ssrc}_k{k_draw}"
                score_col = ssrc2 if ssrc2 in df.columns else chrom_col
                if score_col == chrom_col:
                    print(f"[WARN] {base}: score-source '{ssrc}' not found for k={k_draw}; fallback to '{chrom_col}'.")

        score_series = pd.to_numeric(df.loc[mask, score_col], errors="coerce").to_numpy(float)
        score_series = score_series[np.isfinite(score_series)]
        if score_series.size == 0:
            print(f"[WARN] {base}: score-source '{score_col}' has no finite values for k={k_draw}; fallback to Chromatic Field.")
            score_col = chrom_col
            score_series = chrom_series.copy()

        stats = build_mesh_stats_from_series(score_series, mode=args.score_mode)
        score = cy_score(stats, mode=args.score_mode)

        # 文件名/标题显式带 k，避免覆盖
        name_no_ext, _ = os.path.splitext(base)
        chain_label = chain_id if chain_id is not None else "ALL"
        safe_score = score_col.replace("/", "_")
        safe_color = chrom_col.replace("/", "_")

        mesh_title = f"{name_no_ext}_{chain_label}_{score_col}"
        out_dir = os.path.dirname(out_path)
        mesh_png = os.path.join(
            out_dir,
            f"{name_no_ext}_{chain_label}_{safe_score}_{safe_color}_CYmesh.png"
        )

        colorbar_label = f"{chrom_col} (normalized)"
        render_mesh(
            mesh_title,
            stats,
            chrom_series,
            mesh_png,
            lw=args.lw,
            cmap_name=args.cmap,
            tau_alpha=args.tau_alpha,
            radius_alpha=args.radius_alpha,
            pos_score=score,
            colorbar_label=colorbar_label,
        )
        print(f"[OK] {base}: k={k_draw} CY-like mesh saved to {mesh_png}")

def main():
    args = parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    pdb_files = sorted(glob.glob(os.path.join(in_dir, "*.pdb")))
    if not pdb_files:
        print(f"[WARN] No *.pdb files found in {in_dir}")
        return

    print(f"[INFO] Found {len(pdb_files)} PDB files in {in_dir}")
    print(f"[INFO] ks = {ks}, tan_dim = {args.tan_dim}, plot = {args.plot}")

    for pdb_path in pdb_files:
        base = os.path.basename(pdb_path)
        name, _ = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"{name}.csv")
        process_one_pdb(
            pdb_path=pdb_path,
            out_path=out_path,
            chain_id=args.chain,
            ks=ks,
            tan_dim=args.tan_dim,
            eps=args.eps,
            args=args,
        )


if __name__ == "__main__":
    main()

