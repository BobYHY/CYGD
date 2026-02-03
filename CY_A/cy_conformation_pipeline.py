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
  pip install biopython numpy scikit-learn matplotlib
  # 可选：系统需安装 mkdssp（仅 --annot-ss 时需要）
"""

import json
import math
import argparse
from collections import defaultdict
import numpy as np
import csv
from typing import Optional, List

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------ Biopython ------------------
from Bio.PDB import PDBParser, PDBIO, is_aa
from Bio.PDB.vectors import calc_dihedral

# DSSP（可选）
try:
    from Bio.PDB.DSSP import DSSP
    _HAS_DSSP = True
except Exception:
    _HAS_DSSP = False

# ------------------ 实用映射 ------------------
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

# DSSP 二级结构归并为：H(螺旋)=3, E(片)=2, 其他=1
_DSSP_CODE = {
    'H': 3, 'G': 3, 'I': 3,  # helices
    'E': 2, 'B': 2,          # sheet/bridge
    'T': 1, 'S': 1, '-': 1   # turn/bend/coil
}

# ------------------ 数学小工具 ------------------
def circular_mean(rad_array):
    xs, ys, cnt = 0.0, 0.0, 0
    for v in rad_array:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        xs += math.cos(v); ys += math.sin(v); cnt += 1
    if cnt == 0:
        return None
    return math.atan2(ys, xs)

def standard_J(dim_even):
    assert dim_even % 2 == 0
    J = np.zeros((dim_even, dim_even), dtype=float)
    for k in range(dim_even // 2):
        a, b = 2*k, 2*k+1
        J[a, b] = -1.0
        J[b, a] =  1.0
    return J

def omega(u, v, J):
    return float((J @ u).dot(v))

# ------------------ 原子抓取（含 altLoc） ------------------
def _get_atom(res, name):
    try:
        atom = res[name]
    except KeyError:
        return None
    try:
        if atom.is_disordered():
            best, best_occ = None, -1.0
            for _, child in atom.child_dict.items():
                occ = child.get_occupancy()
                occ = 0.0 if occ is None else float(occ)
                if occ > best_occ:
                    best_occ, best = occ, child
            atom = best
    except AttributeError:
        pass
    return atom

def backbone_dihedrals(res_prev, res, res_next):
    phi = psi = omg = None
    if res_prev is not None:
        C_prev = _get_atom(res_prev, "C")
        N  = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        C  = _get_atom(res, "C")
        if (C_prev is not None) and (N is not None) and (CA is not None) and (C is not None):
            phi = calc_dihedral(C_prev.get_vector(), N.get_vector(),
                                CA.get_vector(), C.get_vector())
    if res_next is not None:
        N  = _get_atom(res, "N")
        CA = _get_atom(res, "CA")
        C  = _get_atom(res, "C")
        N_next = _get_atom(res_next, "N")
        if (N is not None) and (CA is not None) and (C is not None) and (N_next is not None):
            psi = calc_dihedral(N.get_vector(), CA.get_vector(),
                                C.get_vector(), N_next.get_vector())
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
    CG = _get_atom(res, CGname) if CGname else None
    if (N is not None) and (CA is not None) and (CB is not None) and (CG is not None):
        return calc_dihedral(N.get_vector(), CA.get_vector(),
                             CB.get_vector(), CG.get_vector())
    return None

# ------------------ 角序列提取 ------------------
def extract_angle_series(pdb_path, chain_id, model_index=0, angles=("phi","psi"), include_omega=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    models = list(structure.get_models())
    if not models:
        raise ValueError("PDB 中没有模型")
    if model_index >= len(models):
        raise ValueError(f"model_index={model_index} 超出范围（共有 {len(models)} 个模型）")
    model = models[model_index]

    chain = None
    for ch in model:
        if (chain_id is None) or (ch.id == chain_id):
            chain = ch; break
    if chain is None:
        raise ValueError(f"找不到链: {chain_id}")

    residues = [res for res in chain if is_aa(res, standard=True)]
    out = defaultdict(list)
    out["res_index"] = []
    out["res_aa1"]   = []
    out["ca_xyz"]    = []  # for contacts
    out["b_main"]    = []  # per-residue B-factor (CA or backbone mean)

    for i, res in enumerate(residues):
        res_prev = residues[i-1] if i>0 else None
        res_next = residues[i+1] if i+1<len(residues) else None
        phi, psi, omg = backbone_dihedrals(res_prev, res, res_next)
        if "phi" in angles: out["phi"].append(phi)
        if "psi" in angles: out["psi"].append(psi)
        if include_omega or ("omega" in angles): out["omega"].append(omg)
        if "chi1" in angles: out["chi1"].append(chi1(res))
        out["res_index"].append(i)
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
    out["chain_id"] = chain_id
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
    dropped_due_to_aa_filter = 0
    dropped_due_to_aa_frac   = 0

    if aa_allow_set is None:
        aa_min_frac = None
    else:
        if aa_min_frac is None:
            aa_min_frac = 1.0
        aa_min_frac = float(aa_min_frac)

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
                v = vals[idx] if idx < len(vals) else None
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    missing_positions.append((len(vec), a))
                    vec.append(np.nan)
                else:
                    vec.append(v)

        if len(missing_positions) > allow_missing:
            dropped_due_to_missing += 1
            continue

        vec = np.array(vec, dtype=float)
        fill_failed = False
        for pos, a in missing_positions:
            mu = angle_global_mean.get(a, None)
            if mu is None:
                fill_failed = True; break
            vec[pos] = mu
        if fill_failed:
            dropped_due_to_all_missing_for_angle += 1
            continue

        X.append(vec); starts.append(start); aa_windows.append("".join(win_aa))

    stats = dict(
        residue_total=L,
        window_total=total_candidates,
        window_used=len(starts),
        window_dropped_missing=dropped_due_to_missing,
        window_dropped_missing_no_global_mean=dropped_due_to_all_missing_for_angle,
        window_dropped_aa_filter=dropped_due_to_aa_filter,
        window_dropped_aa_filter_by_frac=dropped_due_to_aa_frac
    )

    if not X:
        return np.empty((0, m), dtype=float), np.array([], dtype=int), stats, []
    return np.vstack(X), np.array(starts, dtype=int), stats, aa_windows

# ------------------ 环面嵌入与几何检验 ------------------
def torus_embed(angles_matrix):
    C = np.cos(angles_matrix)
    S = np.sin(angles_matrix)
    return np.concatenate([C, S], axis=1)

def local_tangent(Y, idx, k=20, pca_var=0.90, max_tangent_dim=None, nbrs=None):
    """更稳健：去掉自邻居；邻居不足或零方差时，返回空切空间"""
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
    n = len(Y)
    m = Y.shape[1] // 2
    vals, used = [], 0
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm="auto").fit(Y)
    for i in range(n):
        U, _, _ = local_tangent(Y, i, k=k, pca_var=pca_var, max_tangent_dim=m, nbrs=nbrs)
        d = U.shape[0]
        if d < 2: continue
        used += 1
        for a in range(d):
            for b in range(a+1, d):
                vals.append(abs(omega(U[a], U[b], J)))
    if not vals:
        return dict(lag_mean=0.0, lag_p90=0.0, used_points=0)
    arr = np.array(vals, float)
    return dict(lag_mean=float(arr.mean()),
                lag_p90=float(np.percentile(arr, 90)),
                used_points=int(used))

# ------------------ Ω：精确复分量实现（m=3） ------------------
def holomorphic_volume_m3_exact(u1, u2, u3):
    def to_complex_components(u):
        z1 = u[0] + 1j*u[1]
        z2 = u[2] + 1j*u[3]
        z3 = u[4] + 1j*u[5]
        return np.array([z1, z2, z3], dtype=np.complex128)
    W = np.vstack([to_complex_components(u1),
                   to_complex_components(u2),
                   to_complex_components(u3)])  # (3,3)
    return np.linalg.det(W)

def _sl_stats_from_vals(vals):
    """从一组复数 Ω 值统计 |ImΩ|、相位方差、|Ω|。vals 可以是 list 或 numpy.ndarray。"""
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

def compute_omega_vals_m3(Y, k=30, pca_var=0.95):
    """返回每个样本的 Ω 值（切向维=3 时），供 SL 与 θ 扫描使用"""
    n = len(Y); m = Y.shape[1] // 2
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

def special_lag_scores_m3(Y, J, k=30, pca_var=0.95,
                          calibrate=False, calibrate_mode='l1', grid_deg=1.0,
                          theta_scan_png: Optional[str]=None, theta_scan_steps: int=360):
    """
    返回:
      - 未校准: sl_im_mean, sl_im_p90, sl_phase_var, sl_abs_mean, used_points
      - 校准: sl_im_mean_calib, sl_im_p90_calib, sl_phase_var_calib, calib_theta
      - 若 theta_scan_png: 生成 mean|Im(e^{-iθ}Ω)| vs θ 的曲线
    """
    vals = compute_omega_vals_m3(Y, k=max(k, 10), pca_var=max(pca_var, 0.95))
    used = len(vals)
    stats = _sl_stats_from_vals(vals)
    stats["used_points"] = int(used)
    stats.update(dict(sl_im_mean_calib=None, sl_im_p90_calib=None,
                      sl_phase_var_calib=None, calib_theta=None))

    # θ 扫描图
    if theta_scan_png:
        if used == 0:
            plt.figure(figsize=(6,3))
            plt.title("θ-scan (no valid Ω samples)")
            plt.xlabel("theta (rad)")
            plt.ylabel("mean |Im(e^{-iθ}Ω)|")
            plt.tight_layout(); plt.savefig(theta_scan_png, dpi=200); plt.close()
        else:
            vals_np = np.asarray(vals, np.complex128)
            def mean_abs_im(theta):
                v = vals_np * np.exp(-1j*theta)
                return float(np.mean(np.abs(v.imag)))
            if calibrate_mode == 'grid':
                thetas = np.linspace(0.0, 2*np.pi, max(3, theta_scan_steps), endpoint=False)
            else:
                step = math.radians(grid_deg)
                nstep = int(np.ceil(2*np.pi/step))
                thetas = np.linspace(0.0, 2*np.pi, max(3, nstep), endpoint=False)
            ys = [mean_abs_im(t) for t in thetas]
            plt.figure(figsize=(7,3))
            plt.plot(thetas, ys)
            plt.xlabel("theta (rad)"); plt.ylabel("mean |Im(e^{-iθ}Ω)|")
            plt.title("θ-scan")
            plt.tight_layout(); plt.savefig(theta_scan_png, dpi=200); plt.close()

    if used == 0 or not calibrate:
        return stats

    # 做相位校准
    vals = np.asarray(vals, np.complex128)
    if calibrate_mode == 'l2':
        theta = float(np.angle(np.sum(vals)))
    elif calibrate_mode in ('l1','grid'):
        step = math.radians(grid_deg)
        nstep = int(np.ceil(2*np.pi/step))
        thetas = np.linspace(0.0, 2*np.pi, max(3, nstep), endpoint=False)
        best = (1e9, 0.0)
        for t in thetas:
            v = vals * np.exp(-1j*t)
            m_im = float(np.mean(np.abs(v.imag)))
            if m_im < best[0]:
                best = (m_im, t)
        theta = float(best[1])
    else:
        theta = float(circular_mean(np.angle(vals)) or 0.0)

    vcal = vals * np.exp(-1j*theta)
    cstats = _sl_stats_from_vals(vcal)
    stats.update(dict(
        sl_im_mean_calib=cstats["sl_im_mean"],
        sl_im_p90_calib=cstats["sl_im_p90"],
        sl_phase_var_calib=cstats["sl_phase_var"],
        calib_theta=float(theta)
    ))
    return stats

# ------------------ 局部分数导出 ------------------
def dump_local_csv(path, Y, J, starts, window, k, pca_var, do_sl_m3=False, aa_windows=None,
                   calibrate=False, calib_theta=None):
    n = len(Y); m = Y.shape[1] // 2
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm="auto").fit(Y)

    header = ["sample_id","res_start","res_end","aa_window","tangent_dim",
              "lag_local_mean","lag_local_max","lag_local_p90","lag_pairs","k_used"]
    if do_sl_m3:
        header += ["sl_im","sl_phase","sl_abs"]
        if calibrate:
            header += ["sl_im_calib","sl_phase_calib"]

    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for i in range(n):
            U, inds, _ = local_tangent(Y, i, k=k, pca_var=pca_var, max_tangent_dim=m, nbrs=nbrs)
            d = U.shape[0]
            # |ω|
            lag_vals = []
            if d >= 2:
                for a in range(d):
                    for b in range(a+1, d):
                        lag_vals.append(abs(omega(U[a], U[b], J)))
            lag_mean = float(np.mean(lag_vals)) if lag_vals else None
            lag_max  = float(np.max(lag_vals))  if lag_vals else None
            lag_p90  = float(np.percentile(lag_vals, 90)) if lag_vals else None
            lag_pairs= int(len(lag_vals))

            row = [i, int(starts[i]), int(starts[i] + window - 1),
                   (aa_windows[i] if aa_windows else None),
                   int(d), lag_mean, lag_max, lag_p90, lag_pairs, int(len(inds))]

            if do_sl_m3:
                if d == 3:
                    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
                    v = holomorphic_volume_m3_exact(U_norm[0], U_norm[1], U_norm[2])
                    row += [float(abs(v.imag)), float(np.angle(v)), float(abs(v))]
                    if calibrate and (calib_theta is not None):
                        v2 = v * np.exp(-1j*calib_theta)
                        row += [float(abs(v2.imag)), float(np.angle(v2))]
                    elif calibrate:
                        row += [None, None]
                else:
                    row += [None, None, None]
                    if calibrate:
                        row += [None, None]
            w.writerow(row)

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
    # 映射：按 residues 顺序
    model = list(angle_series["structure"].get_models())[angle_series["model_index"]]
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
    b = np.array(angle_series["b_main"], dtype=float)
    v = b.copy()
    bad = ~np.isfinite(v)
    if np.all(bad):
        return np.zeros_like(b)
    v[bad] = np.nan
    m, M = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(m) or not np.isfinite(M) or M<=m:
        return np.zeros_like(b)
    v = (v - m) / (M - m)
    v[bad] = 0.0
    return v

# ------------------ 热力图绘制（叠加注释） ------------------
def plot_heatmap_from_dump(csv_path, png_path, is_m3=False, calibrated=False,
                           resid_tracks: Optional[dict]=None, starts: Optional[np.ndarray]=None):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        print("[WARN] dump-local CSV 为空，跳过热力图。")
        return

    metrics = ["lag_local_mean", "lag_local_max", "lag_local_p90"]
    if is_m3:
        metrics += ["sl_im", "sl_abs"]
        if calibrated:
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
            M[i, :] = 0.0
        else:
            mv = np.nanmean(row)
            row[mask] = mv
            M[i] = row

    # 叠加 residue 级轨道（按 res_start 对齐）
    if resid_tracks and starts is not None and len(rows)>0:
        res_starts = np.array([int(r["res_start"]) for r in rows], dtype=int)
        annot_rows = []
        annot_labels = []
        for name, track in resid_tracks.items():
            arr = []
            for s in res_starts:
                if s < 0 or s >= len(track):
                    arr.append(0.0)
                else:
                    arr.append(float(track[s]))
            annot_rows.append(arr)
            annot_labels.append(name)
        if annot_rows:
            A = np.array(annot_rows, dtype=float)
            M = np.vstack([M, A])
            metrics += annot_labels

    plt.figure(figsize=(max(8, M.shape[1]*0.02), 3 + 0.5*len(metrics)))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(metrics)), metrics)
    plt.xlabel("sample_id (window index)")
    plt.colorbar(label="score")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"[OK] Heatmap saved to {png_path}")

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser(description="CY-context checks on protein conformational angle space")
    ap.add_argument("--pdb", required=True, help="PDB 文件路径")
    ap.add_argument("--chain", required=True, help="链标识，如 A")
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
    ap.add_argument("--aa-allow", default=None, help="一字母残基白名单，逗号分隔，如 L,I,V")
    ap.add_argument("--aa-min-frac", type=float, default=None, help="白名单比例阈值（0..1）；默认 None=严格(=1.0)")

    # special-Lagrangian 相位校准与 θ 扫描
    ap.add_argument("--sl-calibrate-phase", action="store_true", help="对 Ω 做全局相位校准并输出校准后的统计")
    ap.add_argument("--sl-calibrate-mode", default="l1", choices=["l1","l2","grid"],
                    help="相位校准策略：l1(最小化均值|Im|)、l2(最小化均方Im)、grid(纯网格)")
    ap.add_argument("--grid-deg", type=float, default=1.0, help="相位网格步长（度），用于 l1/grid 模式")
    ap.add_argument("--theta-scan", default=None, help="保存 θ 扫描曲线 PNG")
    ap.add_argument("--theta-scan-steps", type=int, default=360, help="θ 扫描采样数（仅 grid 模式）")

    # 注释叠加
    ap.add_argument("--annot-ss", action="store_true", help="叠加 DSSP 二级结构")
    ap.add_argument("--annot-contacts", action="store_true", help="叠加接触数/接触阶（CA–CA）")
    ap.add_argument("--contacts-cutoff", type=float, default=8.0, help="接触阈值（Å）")
    ap.add_argument("--annot-bfactor", action="store_true", help="叠加 B 因子")
    args = ap.parse_args()

    angles_list = tuple([x.strip() for x in args.angles.split(",") if x.strip()])
    include_omega = ("omega" in angles_list)

    aa_allow_set = None
    if args.aa_allow is not None:
        aa_allow_set = set([x.strip().upper() for x in args.aa_allow.split(",") if x.strip()])

    # 1) 提取角序列 + residue 附加信息
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
            aa_min_frac=args.aa_min_frac,
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
    m = X_angles.shape[1]
    J = standard_J(2*m)

    # 4) 全局评分
    lag = lagrangian_scores(Y, J, k=args.k, pca_var=args.pca_var)
    res = dict(
        n_samples=int(Y.shape[0]),
        m_complex=int(m),
        angles=list(angles_list),
        window=int(args.window),
        step=int(args.step),
        allow_missing=int(args.allow_missing),
        k=int(args.k),
        pca_var=float(args.pca_var),
        residue_total=int(wstats.get("residue_total", 0)),
        window_total=int(wstats.get("window_total", 0)),
        window_used=int(wstats.get("window_used", 0)),
        window_dropped_missing=int(wstats.get("window_dropped_missing", 0)),
        window_dropped_missing_no_global_mean=int(wstats.get("window_dropped_missing_no_global_mean", 0)),
        window_dropped_aa_filter=int(wstats.get("window_dropped_aa_filter", 0)),
        window_dropped_aa_filter_by_frac=int(wstats.get("window_dropped_aa_filter_by_frac", 0)),
        aa_allow=(sorted(list(aa_allow_set)) if aa_allow_set is not None else None),
        aa_min_frac=args.aa_min_frac,
        lagrangian=lag
    )

    # 5) SL 评分与 θ 扫描
    do_sl_m3 = (m == 3)
    calib_theta = None
    if do_sl_m3:
        sl = special_lag_scores_m3(
            Y, J, k=max(args.k, 20), pca_var=max(args.pca_var, 0.95),
            calibrate=args.sl_calibrate_phase,
            calibrate_mode=args.sl_calibrate_mode,
            grid_deg=args.grid_deg,
            theta_scan_png=args.theta_scan,
            theta_scan_steps=args.theta_scan_steps
        )
        res["special_lagrangian"] = sl
        calib_theta = sl.get("calib_theta", None)
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
                       calibrate=args.sl_calibrate_phase, calib_theta=calib_theta)

    # 7) 输出 JSON
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
                               is_m3=do_sl_m3, calibrated=args.sl_calibrate_phase,
                               resid_tracks=(resid_tracks if resid_tracks else None),
                               starts=starts)
    elif args.heatmap and not args.dump_local:
        print("[WARN] --heatmap 需要配合 --dump-local；已跳过。")

if __name__ == "__main__":
    main()

