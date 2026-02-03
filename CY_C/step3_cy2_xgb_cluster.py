#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGB + clustering on aggregated CY-like features.

输入: 由 cath_cylike_eval.py 产生的 aggregated_cylike_features.csv
格式类似:
  Domain,Label,L1,L2,L3,
  all_integrability_err_k10_mean, ...,
  all_kahler_err_k10_mean, ...,
  ...

输出:
  - classification_metrics_cylike_xgb.csv
  - (可选) confusion_matrix_Level{1,2,3}.csv
  - clustering_metrics_cylike.csv
"""

import os, argparse, time
from typing import List

import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    normalized_mutual_info_score, adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ---------- 小工具 ----------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def as_bool(s) -> bool:
    return str(s).strip().lower() in ("yes", "y", "true", "1", "on")


# ---------- XGB helpers（基本照抄 plus_addCY） ----------

def _xgb_ge_20():
    try:
        return int(str(xgb.__version__).split(".")[0]) >= 2
    except Exception:
        return False

def get_xgb_device_params(choice: str):
    """
    choice: 'auto'|'cpu'|'gpu'
    对 XGBoost 2.0+ 使用 device='cuda'/'cpu' + tree_method='hist'
    对旧版使用 tree_method='gpu_hist' / predictor='gpu_predictor'
    """
    ge20 = _xgb_ge_20()
    if choice == "cpu":
        return {"device": "cpu", "tree_method": "hist"} if ge20 else {"tree_method": "hist"}
    if choice == "gpu":
        return {"device": "cuda", "tree_method": "hist"} if ge20 else {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor"
        }
    # auto: prefer GPU
    return {"device": "cuda", "tree_method": "hist"} if ge20 else {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor"
    }

def compute_class_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return {c: w[i] for i, c in enumerate(classes)}

def _fit_with_fallback(model, Xtr, ytr, sw, device_choice: str):
    """
    先按当前 device 尝试训练；若失败且 device_choice=='auto'，自动退回 CPU。
    """
    try:
        model.fit(Xtr, ytr, sample_weight=sw, verbose=False)
        return model, False
    except Exception as e:
        if device_choice == "auto":
            # fallback to CPU
            log(f"[XGB] GPU fit failed, fallback to CPU. Reason: {e}")
            ge20 = _xgb_ge_20()
            cpu_params = {"device": "cpu", "tree_method": "hist"} if ge20 else {"tree_method": "hist"}
            model.set_params(**cpu_params)
            model.fit(Xtr, ytr, sample_weight=sw, verbose=False)
            return model, True
        else:
            raise

def xgb_cv(X, y, num_class: int, seed=0, xgb_device_params=None,
           grid_trials=24, device_choice="auto"):
    """
    简化版 5-fold CV + 随机搜索，返回 (acc, macro-F1)
    """
    if xgb_device_params is None:
        xgb_device_params = {"device": "cpu", "tree_method": "hist"} if _xgb_ge_20() else {"tree_method": "hist"}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 超参搜索空间（和 plus_addCY 一致）
    grid = {
        "n_estimators":     [600, 900, 1500],
        "learning_rate":    [0.05, 0.1],
        "max_depth":        [6, 8],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda":       [1.0, 5.0],
    }
    combos = [
        (ne, lr, md, ss, cs, rl)
        for ne in grid["n_estimators"]
        for lr in grid["learning_rate"]
        for md in grid["max_depth"]
        for ss in grid["subsample"]
        for cs in grid["colsample_bytree"]
        for rl in grid["reg_lambda"]
    ][:grid_trials]

    def _fit(model, Xtr, ytr, sw):
        return _fit_with_fallback(model, Xtr, ytr, sw, device_choice=device_choice)

    y_true_all, y_pred_all = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        cw = compute_class_weights(ytr)
        sw = np.array([cw[c] for c in ytr], dtype=np.float32)

        best_params = None
        best_score = -1.0

        for (ne, lr, md, ss, cs, rl) in combos:
            clf = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=num_class,
                eval_metric="mlogloss",
                random_state=0,
                n_estimators=ne,
                learning_rate=lr,
                max_depth=md,
                subsample=ss,
                colsample_bytree=cs,
                reg_lambda=rl,
                **xgb_device_params,
            )
            Xtr_i, Xva_i, ytr_i, yva_i, sw_tr, sw_va = train_test_split(
                Xtr, ytr, sw, test_size=0.2,
                stratify=ytr, random_state=0
            )
            clf, _ = _fit(clf, Xtr_i, ytr_i, sw_tr)
            ypv = clf.predict(Xva_i)
            score = f1_score(yva_i, ypv, average="macro")
            if score > best_score:
                best_score = score
                best_params = {
                    "n_estimators": ne,
                    "learning_rate": lr,
                    "max_depth": md,
                    "subsample": ss,
                    "colsample_bytree": cs,
                    "reg_lambda": rl,
                }

        # 用 best_params 重新在折内训练 + 验证
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            random_state=0,
            **xgb_device_params,
            **best_params,
        )
        clf, _ = _fit(clf, Xtr, ytr, sw)
        yp = clf.predict(Xte)
        y_true_all.append(yte)
        y_pred_all.append(yp)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    f1m = f1_score(y_true_all, y_pred_all, average="macro")
    return acc, f1m


# ---------- clustering helpers（照抄 plus_addCY） ----------

def cluster_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    s = 0
    for k in np.unique(y_pred):
        m = (y_pred == k)
        if m.sum() == 0:
            continue
        _, cnt = np.unique(y_true[m], return_counts=True)
        s += cnt.max()
    return s / N

def try_agg(X, k, linkage="average", metric="euclidean"):
    try:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
    except TypeError:
        # 兼容旧版 sklearn 的 affinity 参数
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=metric)
    return model.fit_predict(X)

def spectral_cluster(X, k, pca_dim=32, gamma=1.0, seed=0, mode="nearest", n_neighbors=20):
    Xr = X
    if pca_dim and X.shape[1] > pca_dim:
        Xr = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    if mode == "rbf":
        sc = SpectralClustering(
            n_clusters=k,
            eigen_solver="arpack",
            assign_labels="kmeans",
            gamma=gamma,
            random_state=seed,
            n_init=10,
            affinity="rbf",
        )
    else:
        sc = SpectralClustering(
            n_clusters=k,
            eigen_solver="arpack",
            assign_labels="kmeans",
            random_state=seed,
            n_init=10,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
        )
    return sc.fit_predict(Xr)


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="XGB + clustering on aggregated CY-like features")
    ap.add_argument("--input", required=True, help="aggregated_cylike_features.csv")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--xgb_device",  default="auto", choices=["auto","cpu","gpu"])
    ap.add_argument("--grid_trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--save_cm", default="yes")

    ap.add_argument("--clu_max_n", type=int, default=5000)
    ap.add_argument("--run_spectral", default="yes")
    ap.add_argument("--spectral_mode", default="auto", choices=["auto","nearest","rbf"])
    ap.add_argument("--spectral_neighbors", type=int, default=20)
    ap.add_argument("--spectral_pca", type=int, default=32)
    ap.add_argument("--spectral_gamma", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if not set(["Domain","Label","L1","L2","L3"]).issubset(df.columns):
        raise ValueError("input CSV must contain columns: Domain, Label, L1, L2, L3")

    # 选特征列：所有数值列，去掉 Domain/Label 层级
    exclude = {"Domain","Label","L1","L2","L3"}
    feat_cols = [c for c in df.columns
                 if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found in input.")

    log(f"[FEATS] d={len(feat_cols)}")

    # 标准化
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, copy=False)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    if len(df) < 10:
        log("[WARN] too few domains; skip XGB / clustering.")
        return

    # ---------- XGB 监督 ----------
    xgb_dev_params = get_xgb_device_params(args.xgb_device)
    log(f"[XGB] device={args.xgb_device} -> {xgb_dev_params}")

    cls_rows = []
    for level, labcol in [("Level1","L1"), ("Level2","L2"), ("Level3","L3")]:
        if labcol not in df.columns:
            continue
        y = df[labcol].astype(str).to_numpy()
        uniq = sorted(np.unique(y))
        if len(uniq) < 2:
            log(f"[XGB] {level}: only one class, skip.")
            continue

        label2id = {lab: i for i, lab in enumerate(uniq)}
        yi = np.array([label2id[v] for v in y], dtype=int)

        acc, f1m = xgb_cv(
            X, yi, num_class=len(uniq),
            seed=args.seed,
            xgb_device_params=xgb_dev_params,
            grid_trials=args.grid_trials,
            device_choice=args.xgb_device,
        )
        log(f"[XGB] {level}: acc={acc:.4f}, F1_macro={f1m:.4f}")
        cls_rows.append({
            "Representation": "CYLIKE",
            "Level": level,
            "n_class": len(uniq),
            "Accuracy": acc,
            "F1_macro": f1m,
        })

        # 全数据再 fit 一次，用于 confusion matrix
        cw = compute_class_weights(yi)
        sw = np.array([cw[c] for c in yi], dtype=np.float32)
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(uniq),
            eval_metric="mlogloss",
            random_state=0,
            **xgb_dev_params,
        )
        clf, _ = _fit_with_fallback(clf, X, yi, sw, device_choice=args.xgb_device)
        y_pred = clf.predict(X)

        if as_bool(args.save_cm):
            cm = confusion_matrix(yi, y_pred, labels=list(range(len(uniq))))
            cm_df = pd.DataFrame(cm, index=uniq, columns=uniq)
            cm_path = os.path.join(args.outdir, f"confusion_matrix_{level}.csv")
            cm_df.to_csv(cm_path)
            log(f"[XGB] {level} confusion matrix -> {cm_path}")

    if cls_rows:
        pd.DataFrame(cls_rows).to_csv(
            os.path.join(args.outdir, "classification_metrics_cylike_xgb.csv"),
            index=False,
        )
        log("[OK] classification_metrics_cylike_xgb.csv saved")

    # ---------- 聚类（k-means / HAC / Spectral） ----------
    rng = np.random.default_rng(args.seed)
    subN = min(args.clu_max_n, len(df))
    idx = np.sort(rng.choice(np.arange(len(df)), size=subN, replace=False))
    Xc = X[idx]
    sub_df = df.iloc[idx].reset_index(drop=True)

    clu_rows = []
    for level, labcol in [("Level1","L1"), ("Level2","L2"), ("Level3","L3")]:
        if labcol not in sub_df.columns:
            continue
        y = sub_df[labcol].astype(str).to_numpy()
        uniq = sorted(np.unique(y))
        if len(uniq) < 2:
            continue
        k = len(uniq)

        # k-means
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        ykm = kmeans.fit_predict(Xc)
        nmi = normalized_mutual_info_score(y, ykm)
        ari = adjusted_rand_score(y, ykm)
        _, _, v = homogeneity_completeness_v_measure(y, ykm)
        pur = cluster_purity(y, ykm)
        clu_rows.append({
            "Representation": "CYLIKE",
            "Level": level,
            "Method": "KMeans",
            "NMI": nmi,
            "ARI": ari,
            "V_measure": v,
            "Purity": pur,
        })

        # HAC-ward
        yw = try_agg(Xc, k, linkage="ward", metric="euclidean")
        nmi = normalized_mutual_info_score(y, yw)
        ari = adjusted_rand_score(y, yw)
        _, _, v = homogeneity_completeness_v_measure(y, yw)
        pur = cluster_purity(y, yw)
        clu_rows.append({
            "Representation": "CYLIKE",
            "Level": level,
            "Method": "HAC-ward",
            "NMI": nmi,
            "ARI": ari,
            "V_measure": v,
            "Purity": pur,
        })

        # HAC-cosine
        Xn = normalize(Xc)
        yc = try_agg(Xn, k, linkage="average", metric="cosine")
        nmi = normalized_mutual_info_score(y, yc)
        ari = adjusted_rand_score(y, yc)
        _, _, v = homogeneity_completeness_v_measure(y, yc)
        pur = cluster_purity(y, yc)
        clu_rows.append({
            "Representation": "CYLIKE",
            "Level": level,
            "Method": "HAC-cosine",
            "NMI": nmi,
            "ARI": ari,
            "V_measure": v,
            "Purity": pur,
        })

        # Spectral
        if as_bool(args.run_spectral):
            mode = args.spectral_mode
            if mode == "auto":
                mode = "nearest"
            ys = spectral_cluster(
                Xc, k,
                pca_dim=args.spectral_pca,
                gamma=args.spectral_gamma,
                seed=args.seed,
                mode=mode,
                n_neighbors=args.spectral_neighbors,
            )
            nmi = normalized_mutual_info_score(y, ys)
            ari = adjusted_rand_score(y, ys)
            _, _, v = homogeneity_completeness_v_measure(y, ys)
            pur = cluster_purity(y, ys)
            clu_rows.append({
                "Representation": "CYLIKE",
                "Level": level,
                "Method": f"Spectral({mode})",
                "NMI": nmi,
                "ARI": ari,
                "V_measure": v,
                "Purity": pur,
            })

    if clu_rows:
        pd.DataFrame(clu_rows).to_csv(
            os.path.join(args.outdir, "clustering_metrics_cylike.csv"),
            index=False,
        )
        log("[OK] clustering_metrics_cylike.csv saved")


if __name__ == "__main__":
    main()

