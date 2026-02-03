#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cy_conformation_pipeline.py
CY-context checks on protein conformational angle space

功能：
- 角序列提取：phi/psi/omega/chi1
- 滑窗拼角 + 缺失插补（圆均值）
- 残基白名单与比例过滤：--aa-allow --aa-min-frac
- 环面嵌入 + 局部切空间 PCA
- Lagrangian 检验（|ω|统计）
- CY-3 special-Lagrangian 检验（Ω 的精确复行列式）
- 相位校准：--sl-calibrate-phase（模式 l1/l2/grid；--grid-deg 步长）
- θ 扫描：--theta-scan PNG（mean |Im(e^{-iθ}Ω)| vs θ）
- 局部导出：--dump-local CSV（含校准列）
- 热力图：--heatmap PNG（可叠加 DSSP/接触数/接触阶/B 因子）

依赖：
  pip install biopython numpy matplotlib

示例：
  python cy_conformation_pipeline.py --pdb 7r58/7R58.pdb --chain A \\
    --angles phi,psi,chi1 --window 1 --k 30 \\
    --out 7r58/7R58_all_SL.json \\
    --dump-local 7r58/7R58_all_local.csv \\
    --heatmap 7r58/7R58_all_heatmap.png \\
    --sl-calibrate-phase --sl-calibrate-mode l1 --grid-deg 1.0 \\
    --theta-scan 7r58/7R58_theta_scan.png --theta-scan-steps 720 \\
    --annot-contacts --contacts-cutoff 8.0 --annot-bfactor
"""

import os
import sys
import math
import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, PDBIO, is_aa, calc_dihedral

try:
    from Bio.PDB.DSSP import DSSP
    _HAS_DSSP = True
except Exception:
    _HAS_DSSP = False

# ------------------ 常量 & 工具 ------------------

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

_DSSP_CODE = {
    "H": 3.0, "G": 3.0, "I": 3.0,  # helix-like
    "E": 2.0, "B": 2.0,           # strand-like
    # others -> 1.0
}

def _get_atom(res, name):
    try:
        return res[name]
    except KeyError:
        return None

def backbone_dihedrals(res_prev, res, res_next):
    """返回 (phi, psi, omega)，单位 rad；若不可算则为 NaN。"""
    phi = psi = omg = float("nan")
    # phi: C(i-1)-N-CA-C
    if res_prev is not None:
        C_prev = _get_atom(res_prev, "C")
        N = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        C = _get_atom(res, "C")
        if (C_prev is not None) and (N is not None) and (CA is not None) and (C is not None):
            phi = calc_dihedral(C_prev.get_vector(), N.get_vector(),
                                CA.get_vector(), C.get_vector())
    # psi: N-CA-C-N(i+1)
    if res_next is not None:
        N = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        C = _get_atom(res, "C")
        N_next = _get_atom(res_next, "N")
        if (N is not None) and (CA is not None) and (C is not None) and (N_next is not None):
            psi = calc_dihedral(N.get_vector(), CA.get_vector(),
                                C.get_vector(), N_next.get_vector())
    # omega: CA(i-1)-C(i-1)-N-CA
    if res_prev is not None:
        CA_prev = _get_atom(res_prev, "CA")
        C_prev  = _get_atom(res_prev, "C")
        N  = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        if (CA_prev is not None) and (C_prev is not None) and (N is not None) and (CA is not None):
            omg = calc_dihedral(CA_prev.get_vector(), C_prev.get_vector(),
                                N.get_vector(), CA.get_vector())
    return phi, psi, omg

CHI1_GAMMA = {
    "ARG":"CG","LYS":"CG","GLN":"CG","GLU":"CG","MET":"CG",
    "LEU":"CG","ILE":"CG1","VAL":"CG1","THR":"OG1","SER":"OG",
    "CYS":"SG","ASN":"CG","ASP":"CG","HIS":"CG","PHE":"CG",
    "TYR":"CG","TRP":"CG","PRO":"CG"
}
def chi1(res):
    name = res.get_resname()
    N  = _get_atom(res, "N")
    CA = _get_atom(res, "CA")
    CB = _get_atom(res, "CB")
    CGname = CHI1_GAMMA.get(name, None)
    if CGname is None:
        return float("nan")
    CG = _get_atom(res, CGname)
    if (N is None) or (CA is None) or (CB is None) or (CG is None):
        return float("nan")
    return calc_dihedral(N.get_vector(), CA.get_vector(),
                         CB.get_vector(), CG.get_vector())

def circular_mean(angles):
    """角度（rad）的圆均值（忽略 NaN）"""
    arr = np.asarray(angles, float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return math.atan2(np.mean(np.sin(arr)), np.mean(np.cos(arr)))

def torus_embed(angles_matrix):
    """
    angles_matrix: (N, m) rad
    返回 (N, 2m) 的 cos/sin 嵌入
    """
    A = np.asarray(angles_matrix, float)
    cos = np.cos(A)
    sin = np.sin(A)
    return np.concatenate([cos, sin], axis=1)

def standard_J(dim_even):
    assert dim_even % 2 == 0
    J = np.zeros((dim_even, dim_even), float)
    half = dim_even // 2
    for i in range(half):
        J[i, i+half] = -1.0
        J[i+half, i] = 1.0
    return J

# ------------------ 角序列提取：支持单链 & ALL ------------------

def extract_angle_series(pdb_path, chain_id, model_index=0, angles=("phi","psi"), include_omega=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    models = list(structure.get_models())
    if not models:
        raise ValueError("PDB 中没有模型")
    if model_index >= len(models):
        raise ValueError(f"model_index={model_index} 超出范围（共有 {len(models)} 个模型）")
    model = models[model_index]

    # 支持单链或 ALL：ALL 表示把模型中的所有链按顺序拼接，
    # 但在计算主链二面角时不会跨链相连。
    residues = []
    resid_chain_ids = []
    if (chain_id is None) or (str(chain_id).upper() == "ALL"):
        for ch in model:
            for res in ch:
                if is_aa(res, standard=True):
                    residues.append(res)
                    resid_chain_ids.append(ch.id)
        if not residues:
            raise ValueError("在任意链中都没有标准氨基酸残基")
        chain_label = "ALL"
    else:
        chain = None
        for ch in model:
            if ch.id == chain_id:
                chain = ch; break
        if chain is None:
            raise ValueError(f"找不到链: {chain_id}")
        for res in chain:
            if is_aa(res, standard=True):
                residues.append(res)
                resid_chain_ids.append(chain.id)
        chain_label = chain_id

    out = defaultdict(list)
    out["res_index"] = []
    out["res_aa1"]   = []
    out["ca_xyz"]    = []  # for contacts
    out["b_main"]    = []  # per-residue B-factor (CA or backbone mean)

    for a in angles:
        out[a] = []

    for i, res in enumerate(residues):
        # 前后残基仅在同一条链上才视为相邻；跨链则视为断点
        res_prev = None
        res_next = None
        if i > 0 and resid_chain_ids[i-1] == resid_chain_ids[i]:
            res_prev = residues[i-1]
        if i+1 < len(residues) and resid_chain_ids[i+1] == resid_chain_ids[i]:
            res_next = residues[i+1]
        phi, psi, omg = backbone_dihedrals(res_prev, res, res_next)
        if "phi" in angles: out["phi"].append(phi)
        if "psi" in angles: out["psi"].append(psi)
        if include_omega or ("omega" in angles): out["omega"].append(omg)
        if "chi1" in angles: out["chi1"].append(chi1(res))
        out["res_index"].append(i)
        out["res_chain"].append(resid_chain_ids[i])
        out["res_aa1"].append(AA3_TO_1.get(res.get_resname(), "X"))

        # CA xyz + B-factor
        CA = _get_atom(res, "CA")
        if CA is not None:
            out["ca_xyz"].append(CA.get_coord().astype(float))
            b = CA.get_bfactor()
        else:
            out["ca_xyz"].append(None)
            # average backbone B
            bb = []
            for nm in ("N","CA","C","O"):
                a = _get_atom(res, nm)
                if a is not None: bb.append(a.get_bfactor())
            b = float(np.mean(bb)) if bb else np.nan
        out["b_main"].append(b)

    out["res_count"] = len(residues)
    out["structure"] = structure
    out["model_index"] = model_index
    out["chain_id"] = chain_label
    out["pdb_path"] = pdb_path
    return out

# ------------------ 滑窗拼角 + 缺失处理 + 残基过滤 ------------------

def angle_window_matrix(angle_series, angles_order, window, step,
                        allow_missing=0, aa_allow_set=None, aa_min_frac=None):
    L = len(angle_series["res_index"])
    m = window * len(angles_order)
    X, starts, aa_windows = [], [], []

    total_candidates = max(0, L - window + 1)
    dropped_due_to_missing = 0
    dropped_due_to_all_missing_for_angle = 0

    if aa_allow_set is not None:
        if aa_min_frac is None:
            aa_min_frac = 1.0
        aa_min_frac = float(aa_min_frac)
    else:
        aa_min_frac = None

    angle_global_mean = {a: circular_mean(angle_series.get(a, [])) for a in angles_order}

    for start in range(0, L - window + 1, step):
        win_aa = angle_series["res_aa1"][start:start+window]

        # 白名单比例过滤
        if aa_allow_set is not None:
            cnt_ok = sum(1 for a in win_aa if a in aa_allow_set)
            frac = cnt_ok / float(window)
            if frac < aa_min_frac:
                if cnt_ok == window:
                    dropped_due_to_aa_filter += 1
                else:
                    dropped_due_to_aa_frac += 1
                continue

        vec = []
        missing_positions = []
        for t in range(window):
            idx = start + t
            for a in angles_order:
                vals = angle_series[a]
                v = vals[idx] if idx < len(vals) else float("nan")
                if np.isnan(v):
                    missing_positions.append((len(vec), a))
                vec.append(v)
        vec = np.asarray(vec, float)

        # 如果缺失超过允许数，跳过
        if len(missing_positions) > allow_missing:
            dropped_due_to_missing += 1
            continue

        # 替换 NaN 用全局圆均值
        for pos, a in missing_positions:
            vec[pos] = angle_global_mean[a]

        # 检查是否对某个角全部都是 NaN（极端情况）
        all_missing_flag = False
        for ai, a in enumerate(angles_order):
            vals = angle_series[a]
            sub = vals[start:start+window]
            if all(np.isnan(sub)):
                all_missing_flag = True
                break
        if all_missing_flag:
            dropped_due_to_all_missing_for_angle += 1
            continue

        X.append(vec)
        starts.append(start)
        aa_windows.append("".join(win_aa))

    X = np.array(X, dtype=float)
    stats = dict(
        residue_total=int(L),
        window_total=int(total_candidates),
        window_dropped_missing=int(dropped_due_to_missing),
        window_dropped_missing_no_global_mean=int(dropped_due_to_all_missing_for_angle),
        window_dropped_aa_filter=int(0),   # 已在外面统计
        window_dropped_aa_filter_by_frac=int(0)
    )
    return X, np.array(starts, dtype=int), stats, aa_windows

# ------------------ Lagrangian & Ω 计算 ------------------

def local_tangent_pca(Y, k, pca_var=0.9):
    """
    Y: (N, D) torus embedding
    返回：
      local_means: (N, D)
      bases_list:  长度 N 的列表，每项为 (d_i, D) 的正交基
    """
    N, D = Y.shape
    local_means = np.zeros_like(Y)
    bases_list = []

    for i in range(N):
        dists = np.linalg.norm(Y - Y[i], axis=1)
        idx = np.argsort(dists)[:k]
        neigh = Y[idx]
        mu = np.mean(neigh, axis=0)
        local_means[i] = mu
        Xc = neigh - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S**2) / np.sum(S**2) if np.sum(S**2) > 0 else np.zeros_like(S)
        d = np.searchsorted(np.cumsum(var), pca_var) + 1
        d = max(1, min(d, D))
        bases_list.append(Vt[:d].T)  # D x d

    return local_means, bases_list

def lagrangian_scores(Y, bases_list, J):
    """
    对每个局部切空间（基 = B, Dxd），计算 |ω| 的统计：
      ω(u,v) = <J u, v>
    返回：
      lag_local_mean, lag_local_max, lag_local_p90
    """
    N, D = Y.shape
    lag_local_mean = np.zeros((N,), float)
    lag_local_max  = np.zeros((N,), float)
    lag_local_p90  = np.zeros((N,), float)

    for i in range(N):
        B = bases_list[i]      # D x d
        JB = J @ B             # D x d
        d = B.shape[1]
        if d < 2:
            lag_local_mean[i] = 0.0
            lag_local_max[i]  = 0.0
            lag_local_p90[i]  = 0.0
            continue
        ws = []
        for a in range(d):
            for b in range(a+1, d):
                u = B[:,a]
                v = B[:,b]
                w = float(np.dot(J @ u, v))
                ws.append(abs(w))
        if not ws:
            lag_local_mean[i] = 0.0
            lag_local_max[i]  = 0.0
            lag_local_p90[i]  = 0.0
            continue
        ws = np.array(ws, float)
        lag_local_mean[i] = float(np.mean(ws))
        lag_local_max[i]  = float(np.max(ws))
        lag_local_p90[i]  = float(np.percentile(ws, 90.0))
    return lag_local_mean, lag_local_max, lag_local_p90

def special_lag_scores_m3(Y, bases_list):
    """
    m=3 情形：Ω = det(W)，W_{ij} 为局部切空间基在 C^3 坐标下的分量。
    Y: (N, 6) = (Re z1,Re z2,Re z3, Im z1,Im z2,Im z3)
    返回字典：
      - vals: 复数数组 Ω_i
      - im_mean, im_p90, abs_mean, phase_var 等
    """
    N, D = Y.shape
    assert D == 6
    vals = []
    for i in range(N):
        B = bases_list[i]   # 6 x d
        d = B.shape[1]
        if d < 3:
            vals.append(np.nan+1j*np.nan)
            continue
        B3 = B[:, :3]       # 6 x 3
        Re = B3[:3,:]
        Im = B3[3:,:]
        W = Re + 1j*Im      # 3 x 3
        detW = np.linalg.det(W)
        vals.append(detW)
    vals = np.array(vals, dtype=np.complex128)

    good = ~np.isnan(vals)
    used = int(np.sum(good))
    if used == 0:
        return dict(
            sl_im_mean=float("nan"),
            sl_im_p90=float("nan"),
            sl_phase_var=float("nan"),
            sl_abs_mean=float("nan"),
            used_points=0
        )

    v = vals[good]
    im_abs = np.abs(v.imag)
    abs_v  = np.abs(v)
    phases = np.angle(v)

    return dict(
        sl_im_mean=float(np.mean(im_abs)),
        sl_im_p90=float(np.percentile(im_abs, 90.0)),
        sl_phase_var=float(np.var(phases)),
        sl_abs_mean=float(np.mean(abs_v)),
        used_points=used,
        vals=vals
    )

# ------------------ 相位校准 & θ 扫描 ------------------

def calibrate_phase(vals, mode="l1", grid_deg=1.0):
    """
    vals: 复数数组 Ω_i
    mode: l1 / l2 / grid
    返回 (theta_best, vals_rot)
    """
    vals = np.asarray(vals, np.complex128)
    good = ~np.isnan(vals)
    if not np.any(good):
        return 0.0, vals

    def obj_l1(theta):
        v = vals[good] * np.exp(-1j*theta)
        return float(np.mean(np.abs(v.imag)))

    def obj_l2(theta):
        v = vals[good] * np.exp(-1j*theta)
        return float(np.mean((v.imag)**2))

    if mode == "grid":
        thetas = np.linspace(0.0, 2*np.pi, int(360.0/grid_deg), endpoint=False)
        vals_obj = [obj_l1(t) for t in thetas]
        idx = int(np.argmin(vals_obj))
        theta_best = float(thetas[idx])
    elif mode == "l2":
        # 粗 grid + 局部梯度下降
        thetas = np.linspace(0.0, 2*np.pi, 720, endpoint=False)
        vals_obj = [obj_l2(t) for t in thetas]
        theta_best = float(thetas[int(np.argmin(vals_obj))])
        for _ in range(50):
            h = 1e-3
            f1 = obj_l2(theta_best + h)
            f2 = obj_l2(theta_best - h)
            g = (f1 - f2) / (2*h)
            theta_best -= 0.1 * g
    else:
        # l1：粗 grid 搜索
        thetas = np.linspace(0.0, 2*np.pi, 720, endpoint=False)
        vals_obj = [obj_l1(t) for t in thetas]
        theta_best = float(thetas[int(np.argmin(vals_obj))])

    vals_rot = vals * np.exp(-1j*theta_best)
    return theta_best, vals_rot

def theta_scan_plot(vals, path_png, steps=720):
    vals = np.asarray(vals, np.complex128)
    good = ~np.isnan(vals)
    if not np.any(good):
        plt.figure(figsize=(6,3))
        plt.title("θ-scan (no valid Ω values)")
        plt.xlabel("theta (rad)")
        plt.ylabel("mean |Im(e^{-iθ}Ω)|")
        plt.tight_layout()
        plt.savefig(path_png, dpi=200)
        plt.close()
        return

    vals_np = vals[good]

    def mean_abs_im(theta):
        v = vals_np * np.exp(-1j*theta)
        return float(np.mean(np.abs(v.imag)))

    thetas = np.linspace(0.0, 2*np.pi, steps, endpoint=False)
    ys = [mean_abs_im(t) for t in thetas]

    plt.figure(figsize=(7,3))
    plt.plot(thetas, ys)
    plt.title("θ-scan")
    plt.xlabel("theta (rad)")
    plt.ylabel("mean |Im(e^{-iθ}Ω)|")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

# ------------------ 注释轨道：DSSP / Contacts / B-factor ------------------

def compute_dssp_track(angle_series, pdb_path):
    """返回长度 L 的数组：3/2/1（H/E/other），若失败则全 1"""
    L = angle_series["res_count"]
    if not _HAS_DSSP:
        print("[WARN] Biopython DSSP not available; skip --annot-ss.")
        return np.ones((L,), dtype=float)

    # 优先尝试用原始文件路径
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("tmp", pdb_path)
        model = list(structure.get_models())[angle_series["model_index"]]
        dssp = DSSP(model, pdb_path)
    except Exception as e:
        # 退路：把当前 structure 写到临时文件再跑 DSSP
        try:
            structure = angle_series["structure"]
            model = list(structure.get_models())[angle_series["model_index"]]
            io = PDBIO(); io.set_structure(structure)
            import tempfile, os
            with tempfile.TemporaryDirectory() as td:
                tmp_path = os.path.join(td, "tmp_dssp.pdb")
                io.save(tmp_path)
                dssp = DSSP(model, tmp_path)
        except Exception as e2:
            print(f"[WARN] DSSP failed ({e}); fallback coil.")
            return np.ones((L,), dtype=float)

    chain_id = angle_series["chain_id"]
    ss = np.ones((L,), dtype=float)
    # 映射：按 residues 顺序；支持单链或 ALL
    model = list(angle_series["structure"].get_models())[angle_series["model_index"]]

    if (chain_id is None) or (str(chain_id).upper() == "ALL"):
        # ALL: 遍历所有链，与 extract_angle_series 中的顺序保持一致
        res_i = 0
        for ch in model:
            cid = ch.id
            for r in ch:
                if not is_aa(r, standard=True): 
                    continue
                if res_i >= L:
                    break
                key = (cid, r.id)
                try:
                    sym = dssp[key][2]  # DSSP code
                    ss[res_i] = _DSSP_CODE.get(sym, 1)
                except Exception:
                    ss[res_i] = 1.0
                res_i += 1
        return ss
    else:
        chain = None
        for ch in model:
            if ch.id == chain_id:
                chain = ch; break
        if chain is None:
            return ss
        res_i = 0
        for r in chain:
            if not is_aa(r, standard=True): continue
            key = (chain_id, r.id)
            try:
                sym = dssp[key][2]  # DSSP code
                ss[res_i] = _DSSP_CODE.get(sym, 1)
            except Exception:
                ss[res_i] = 1.0
            res_i += 1
        return ss

def compute_contacts_tracks(angle_series, cutoff=8.0):
    L = angle_series["res_count"]
    xyz = angle_series["ca_xyz"]
    pts = []
    idx_map = []
    for i, p in enumerate(xyz):
        if p is not None:
            pts.append(p); idx_map.append(i)
    if not pts:
        return np.zeros((L,), float), np.zeros((L,), float)
    P = np.vstack(pts)  # (M,3)
    counts = np.zeros((L,), float)
    order  = np.zeros((L,), float)
    for a in range(P.shape[0]):
        i = idx_map[a]
        pa = P[a]
        local_orders = []
        c = 0
        for b in range(P.shape[0]):
            if a == b: continue
            j = idx_map[b]
            # 排除邻接序的直接相邻
            if abs(i-j) <= 1: continue
            if np.linalg.norm(pa - P[b]) <= cutoff:
                c += 1
                local_orders.append(abs(i-j))
        counts[i] = c
        order[i]  = (np.mean(local_orders) if local_orders else 0.0)
    return counts, order

def compute_bfactor_track(angle_series):
    L = angle_series["res_count"]
    b = np.asarray(angle_series["b_main"], float)
    if np.all(np.isnan(b)):
        return np.ones((L,), float)
    # 简单线性归一化
    m = np.nanmean(b)
    s = np.nanstd(b) if np.nanstd(b) > 1e-6 else 1.0
    z = (b - m) / s
    # 映射到 [0.5, 1.5]
    z = 1.0 + 0.5 * np.tanh(z)
    return z

# ------------------ dump-local & 热力图 ------------------

def dump_local_csv(path, Y, J, starts, window,
                   k, pca_var, do_sl_m3,
                   aa_windows=None,
                   sl_vals=None, sl_vals_calib=None,
                   lag_local_mean=None, lag_local_max=None, lag_local_p90=None):
    import csv
    N, D = Y.shape
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sample_id", "start", "window", "aa_seq",
                  "lag_local_mean", "lag_local_max", "lag_local_p90"]
        if do_sl_m3:
            header += ["sl_im", "sl_abs", "sl_im_calib"]
        w.writerow(header)
        for i in range(N):
            row = [i, int(starts[i]), int(window),
                   (aa_windows[i] if aa_windows is not None else "")]
            row += [float(lag_local_mean[i]), float(lag_local_max[i]), float(lag_local_p90[i])]
            if do_sl_m3:
                v = sl_vals[i]
                if np.isnan(v):
                    row += ["", "", ""]
                else:
                    row += [float(abs(v.imag)), float(abs(v)), float(abs(sl_vals_calib[i].imag))]
            w.writerow(row)

def plot_heatmap_from_dump(csv_path, png_path, is_m3, resid_tracks=None):
    import csv
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[WARN] no rows in dump-local; skip heatmap.")
        return

    metrics = ["lag_local_mean", "lag_local_max", "lag_local_p90"]
    if is_m3:
        metrics += ["sl_im", "sl_abs"]
        if resid_tracks is not None:
            metrics += ["sl_im_calib"]

    rows.sort(key=lambda r: int(r["sample_id"]))
    M = []
    for m in metrics:
        vals = []
        for r in rows:
            v = r.get(m, "")
            if v == "" or v.lower() == "none":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except:
                    vals.append(np.nan)
        M.append(vals)
    M = np.array(M, dtype=float)

    # 行均值填补 NaN
    for i in range(M.shape[0]):
        row = M[i]
        mask = np.isnan(row)
        if np.all(mask):
            M[i,:] = 0.0
        else:
            row[mask] = np.nanmean(row)

    plt.figure(figsize=(10, max(3, 0.4*M.shape[0])))
    plt.imshow(M, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="score")
    plt.yticks(range(len(metrics)), metrics)
    plt.xlabel("sample index")

    if resid_tracks is not None:
        ax = plt.gca()
        top_ax = ax.twiny()

        # dump-local 里的每一行对应一个 sample（窗口）
        L = len(rows)
        x = np.arange(L)

        # 从 CSV 里读出每个 sample 的窗口起点 & 窗口长度
        starts = np.array([int(r["start"]) for r in rows], dtype=int)
        # 所有行的 window 通常相同，这里取第一行；取不到就默认为 1
        try:
            win = int(rows[0].get("window", 1))
        except Exception:
            win = 1
        if win <= 0:
            win = 1

        offset = 0.0
        for name, track in resid_tracks.items():
            # track 是“按残基”的轨道，长度 = residue_total
            t_res = np.asarray(track, float)

            # 映射成“按 sample”的轨道：对 [start, start+window) 取平均
            t_samp = []
            n_res = t_res.shape[0]
            for s in starts:
                s0 = max(0, s)
                s1 = min(s + win, n_res)
                if s0 >= n_res or s0 >= s1:
                    t_samp.append(np.nan)
                else:
                    seg = t_res[s0:s1]
                    # 避免全 NaN
                    if np.all(np.isnan(seg)):
                        t_samp.append(np.nan)
                    else:
                        t_samp.append(np.nanmean(seg))
            t_samp = np.asarray(t_samp, float)

            # 用行均值填补 NaN，防止绘图再出问题
            mask = np.isnan(t_samp)
            if np.all(mask):
                continue
            t_samp[mask] = np.nanmean(t_samp[~mask])

            # 归一化到 [0,1] 后平移缩放到顶部
            t_norm = (t_samp - np.min(t_samp)) / (np.max(t_samp) - np.min(t_samp) + 1e-6)
            top_ax.plot(x, 1.1 + offset + 0.2 * t_norm, label=name)
            offset += 0.4

        top_ax.set_xlim(ax.get_xlim())
        top_ax.set_xticks([])
        top_ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

# ------------------ 主函数 ------------------

def main():
    ap = argparse.ArgumentParser(description="CY-context checks on protein conformational angle space")
    ap.add_argument("--pdb", required=True, help="PDB 文件路径")
    ap.add_argument("--chain", required=True, help="链标识，如 A，或 ALL 表示所有链按顺序拼接")
    ap.add_argument("--model-index", type=int, default=0, help="选择第几个模型（默认0）")
    ap.add_argument("--angles", default="phi,psi", help="角列表：phi,psi,omega,chi1（逗号分隔）")
    ap.add_argument("--window", type=int, default=3, help="滑窗残基数")
    ap.add_argument("--step", type=int, default=1, help="滑窗步长")
    ap.add_argument("--allow-missing", type=int, default=0, help="每窗口允许缺的角个数（0=不允许）")
    ap.add_argument("--k", type=int, default=20, help="kNN 邻域大小")
    ap.add_argument("--pca-var", type=float, default=0.90, help="PCA 累计方差阈值（0-1）")
    ap.add_argument("--out", default="cy_checks.json", help="输出 JSON 路径")
    ap.add_argument("--dump-local", default=None, help="导出每个窗口的局部分数 CSV 路径")
    ap.add_argument("--heatmap", default=None, help="基于 dump-local 的热力图 PNG 路径")

    # 过滤
    ap.add_argument("--aa-allow", default=None, help="允许的氨基酸一字母列表，如 L,I,V,M,F")
    ap.add_argument("--aa-min-frac", type=float, default=None, help="窗口内白名单残基最小比例（0-1）")

    # Special-Lagrangian & 相位
    ap.add_argument("--sl-calibrate-phase", action="store_true", help="是否做 Ω 相位校准")
    ap.add_argument("--sl-calibrate-mode", default="l1", choices=["l1","l2","grid"], help="相位校准模式")
    ap.add_argument("--grid-deg", type=float, default=1.0, help="grid 模式下的角度步长（度）")
    ap.add_argument("--theta-scan", default=None, help="θ 扫描输出 PNG 路径")
    ap.add_argument("--theta-scan-steps", type=int, default=720, help="θ 扫描步数（默认720）")

    # 注释
    ap.add_argument("--annot-ss", action="store_true", help="叠加 DSSP 二级结构轨道")
    ap.add_argument("--annot-contacts", action="store_true", help="叠加接触数/接触阶轨道")
    ap.add_argument("--contacts-cutoff", type=float, default=8.0, help="接触判定距离 Å")
    ap.add_argument("--annot-bfactor", action="store_true", help="叠加 B 因子轨道")

    args = ap.parse_args()

    angles_list = [s.strip() for s in args.angles.split(",") if s.strip()]
    include_omega = ("omega" in angles_list)

    aa_allow_set = None
    if args.aa_allow:
        aa_allow_set = set(a.strip() for a in args.aa_allow.split(",") if a.strip())
    aa_min_frac = args.aa_min_frac

    # 1) 提取角序列
    series = extract_angle_series(args.pdb, args.chain,
                                  model_index=args.model_index,
                                  angles=angles_list,
                                  include_omega=include_omega)

    # 2) 滑窗拼角 + 缺失处理 + 残基过滤
    X_angles, starts, wstats, aa_windows = angle_window_matrix(
        series, angles_order=angles_list, window=args.window, step=args.step,
        allow_missing=args.allow_missing,
        aa_allow_set=aa_allow_set, aa_min_frac=args.aa_min_frac
    )

    if X_angles.size == 0:
        res_empty = dict(
            n_samples=0,
            m_complex=args.window * len(angles_list),
            angles=list(angles_list),
            window=int(args.window),
            step=int(args.step),
            allow_missing=int(args.allow_missing),
            k=int(args.k),
            pca_var=float(args.pca_var),
            residue_total=int(wstats.get("residue_total", 0)),
            window_total=int(wstats.get("window_total", 0)),
            window_used=0,
            window_dropped_missing=int(wstats.get("window_dropped_missing", 0)),
            window_dropped_missing_no_global_mean=int(wstats.get("window_dropped_missing_no_global_mean", 0)),
            window_dropped_aa_filter=int(wstats.get("window_dropped_aa_filter", 0)),
            window_dropped_aa_filter_by_frac=int(wstats.get("window_dropped_aa_filter_by_frac", 0)),
            aa_allow=(sorted(list(aa_allow_set)) if aa_allow_set is not None else None),
            aa_min_frac=(float(args.aa_min_frac) if args.aa_min_frac is not None else None),
            lagrangian=None,
            special_lagrangian=None
        )
        with open(args.out, "w") as f:
            json.dump(res_empty, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved to {args.out}")
        print(json.dumps(res_empty, indent=2, ensure_ascii=False))
        return

    # 3) 环面嵌入
    Y = torus_embed(X_angles)
    N, D = Y.shape
    m = args.window * len(angles_list)
    J = standard_J(D)

    # 4) 局部切空间 PCA + Lagrangian 检验
    local_means, bases_list = local_tangent_pca(Y, k=args.k, pca_var=args.pca_var)
    lag_local_mean, lag_local_max, lag_local_p90 = lagrangian_scores(Y, bases_list, J)

    lag_mean = float(np.mean(lag_local_mean))
    lag_p90  = float(np.percentile(lag_local_mean, 90.0))

    lag_info = dict(
        lag_mean=lag_mean,
        lag_p90=lag_p90,
        used_points=int(N)
    )

    # 5) Special-Lagrangian (仅 m=3 时)
    do_sl_m3 = (m == 3)
    res = dict(
        n_samples=int(N),
        m_complex=int(m),
        angles=list(angles_list),
        window=int(args.window),
        step=int(args.step),
        allow_missing=int(args.allow_missing),
        k=int(args.k),
        pca_var=float(args.pca_var),
        residue_total=int(wstats.get("residue_total", 0)),
        window_total=int(wstats.get("window_total", 0)),
        window_used=int(N),
        window_dropped_missing=int(wstats.get("window_dropped_missing", 0)),
        window_dropped_missing_no_global_mean=int(wstats.get("window_dropped_missing_no_global_mean", 0)),
        window_dropped_aa_filter=int(wstats.get("window_dropped_aa_filter", 0)),
        window_dropped_aa_filter_by_frac=int(wstats.get("window_dropped_aa_filter_by_frac", 0)),
        aa_allow=(sorted(list(aa_allow_set)) if aa_allow_set is not None else None),
        aa_min_frac=(float(args.aa_min_frac) if args.aa_min_frac is not None else None),
        lagrangian=lag_info,
        special_lagrangian=None
    )

    sl = None
    sl_vals = None
    sl_vals_calib = None
    calib_theta = None

    if do_sl_m3:
        sl = special_lag_scores_m3(Y, bases_list)
        sl_vals = sl.pop("vals")
        res["special_lagrangian"] = sl

        if args.sl_calibrate_phase:
            theta, vals_rot = calibrate_phase(sl_vals, mode=args.sl_calibrate_mode, grid_deg=args.grid_deg)
            sl["sl_im_mean_calib"] = float(np.mean(np.abs(vals_rot[~np.isnan(vals_rot)].imag)))
            sl["sl_im_p90_calib"]  = float(np.percentile(np.abs(vals_rot[~np.isnan(vals_rot)].imag), 90.0))
            sl["sl_phase_var_calib"] = sl["sl_phase_var"]
            sl["calib_theta"] = float(theta)
            sl_vals_calib = vals_rot
            calib_theta = theta

        # θ-scan
        if args.theta_scan:
            theta_scan_plot(sl_vals, args.theta_scan, steps=args.theta_scan_steps)
    else:
        if args.theta_scan:
            plt.figure(figsize=(6,3))
            plt.title("θ-scan requires m=3 (angles*window=3)")
            plt.xlabel("theta (rad)"); plt.ylabel("mean |Im(e^{-iθ}Ω)|")
            plt.tight_layout(); plt.savefig(args.theta_scan, dpi=200); plt.close()
        res["special_lagrangian"] = None

    # 6) 局部分数导出
    if args.dump_local:
        dump_local_csv(args.dump_local, Y, J, starts, args.window,
                       k=args.k, pca_var=args.pca_var, do_sl_m3=do_sl_m3,
                       aa_windows=aa_windows,
                       sl_vals=sl_vals,
                       sl_vals_calib=sl_vals_calib,
                       lag_local_mean=lag_local_mean,
                       lag_local_max=lag_local_max,
                       lag_local_p90=lag_local_p90)

    # 7) 总体 JSON
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved to {args.out}")
    print(json.dumps(res, indent=2, ensure_ascii=False))

    # 8) 注释轨道准备（按 residue ）
    resid_tracks = {}
    if args.annot_ss:
        resid_tracks["SS(3/2/1)"] = compute_dssp_track(series, series["pdb_path"])
    if args.annot_contacts:
        counts, order = compute_contacts_tracks(series, cutoff=args.contacts_cutoff)
        resid_tracks[f"ContactCount({args.contacts_cutoff}Å)"] = counts
        resid_tracks["ContactOrder"] = order
    if args.annot_bfactor:
        resid_tracks["Bfactor(norm)"] = compute_bfactor_track(series)

    # 9) 热力图（叠加注释）
    if args.heatmap and args.dump_local:
        plot_heatmap_from_dump(args.dump_local, args.heatmap,
                               is_m3=do_sl_m3,
                               resid_tracks=resid_tracks if resid_tracks else None)

if __name__ == "__main__":
    main()

