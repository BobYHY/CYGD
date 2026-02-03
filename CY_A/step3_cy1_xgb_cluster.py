#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CATH CY-like XGB + clustering evaluation

输入：第 1 步脚本生成的 aggregated_cylike_features.csv，包含：
  Domain, name, Label, L1, L2, L3, <大量特征列>

输出：
  - classification_metrics_xgb.csv
  - clustering_metrics_plus.csv
  - (可选) confusion_matrix_Level1/2/3.csv
"""

import os, argparse, time, warnings
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    normalized_mutual_info_score, adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

# 可按需修改 GPU 号
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import xgboost as xgb


# ----------------- utils -----------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def as_bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


# ----------------- XGB helpers -----------------
def _xgb_ge_20():
    try:
        return int(str(xgb.__version__).split(".")[0]) >= 2
    except Exception:
        return False

def get_xgb_device_params(choice: str) -> Dict:
    ge20 = _xgb_ge_20()
    if choice == "cpu":
        return {"device": "cpu", "tree_method": "hist"} if ge20 else {"tree_method": "hist"}
    if choice == "gpu":
        return {"device": "cuda", "tree_method": "hist"} if ge20 else {
            "tree_method": "gpu_hist", "predictor": "gpu_predictor"
        }
    # auto
    return {"device": "cuda", "tree_method": "hist"} if ge20 else {
        "tree_method": "gpu_hist", "predictor": "gpu_predictor"
    }

def compute_class_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return {c: w[i] for i, c in enumerate(classes)}

def _fit_with_fallback(model, Xtr, ytr, sw, device_choice: str):
    try:
        model.fit(Xtr, ytr, sample_weight=sw, verbose=False)
        return model, False
    except Exception as e:
        if device_choice == "auto":
            log(f"[XGB] GPU fit failed, fallback to CPU. Reason: {e}")
            ge20 = _xgb_ge_20()
            cpu_params = {"device": "cpu", "tree_method": "hist"} if ge20 else {"tree_method": "hist"}
            base = {k: v for k, v in model.get_params().items()
                    if k not in ["tree_method", "predictor", "device"]}
            m2 = xgb.XGBClassifier(**base, **cpu_params)
            m2.fit(Xtr, ytr, sample_weight=sw, verbose=False)
            return m2, True
        else:
            raise

def xgb_cv(X, y, num_class: int, seed=0, xgb_device_params=None,
           grid_trials=24, device_choice="auto"):
    if xgb_device_params is None:
        xgb_device_params = {"device": "cpu", "tree_method": "hist"} if _xgb_ge_20() else {"tree_method": "hist"}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []

    grid = {
        "n_estimators": [600, 900, 1500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1.0, 5.0],
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

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        cw = compute_class_weights(ytr)
        sw = np.array([cw[c] for c in ytr], dtype=np.float32)

        best_params = None
        best_score = -1.0

        # 内部 val 选最优超参
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
            try:
                clf, _ = _fit(clf, Xtr_i, ytr_i, sw_tr)
            except Exception:
                continue
            ypv = clf.predict(Xva_i)
            score = f1_score(yva_i, ypv, average="macro")
            if score > best_score:
                best_score = score
                best_params = dict(
                    n_estimators=ne,
                    learning_rate=lr,
                    max_depth=md,
                    subsample=ss,
                    colsample_bytree=cs,
                    reg_lambda=rl,
                )

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
        y_true_all.append(y[te])
        y_pred_all.append(yp)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    f1m = f1_score(y_true_all, y_pred_all, average="macro")
    return acc, f1m


# ----------------- clustering helpers -----------------
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
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=metric)
    return model.fit_predict(X)

def spectral_cluster(X, k, pca_dim=32, gamma=1.0, seed=0, mode="nearest", n_neighbors=20):
    Xr = X
    if pca_dim and X.shape[1] > pca_dim:
        Xr = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    if mode == "rbf":
        sc = SpectralClustering(
            n_clusters=k,
            eigen_solver='arpack',
            assign_labels='kmeans',
            gamma=gamma,
            random_state=seed,
            n_init=10,
            affinity='rbf',
        )
    else:
        sc = SpectralClustering(
            n_clusters=k,
            eigen_solver='arpack',
            assign_labels='kmeans',
            random_state=seed,
            n_init=10,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
        )
    return sc.fit_predict(Xr)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="CATH CY-like XGB + clustering eval")
    ap.add_argument("--input_csv", required=True,
                    help="aggregated_cylike_features.csv from features-only script")
    ap.add_argument("--outdir", required=True)
    # XGB 部分
    ap.add_argument("--xgb_device",  default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--grid_trials", type=int, default=24)
    ap.add_argument("--seed",        type=int, default=3407)
    ap.add_argument("--save_cm",     default="yes")
    # 聚类部分
    ap.add_argument("--clu_max_n",   type=int, default=6000)
    ap.add_argument("--run_hac_ward",  default="yes")
    ap.add_argument("--run_hac_cos",   default="yes")
    ap.add_argument("--run_spectral",  default="yes")
    ap.add_argument("--spectral_mode", default="nearest", choices=["nearest", "rbf"])
    ap.add_argument("--spectral_neighbors", type=int, default=20)
    ap.add_argument("--spectral_pca",  type=int, default=32)
    ap.add_argument("--spectral_gamma", type=float, default=1.0)
    ap.add_argument("--kmeans_inits",  type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 读 aggregated_cylike_features.csv
    df = pd.read_csv(args.input_csv)
    if df.empty:
        log("[ERR] input_csv is empty.")
        return

    # 基本检查
    for col in ["Domain", "name", "Label", "L1", "L2", "L3"]:
        if col not in df.columns:
            log(f"[ERR] input_csv missing column: {col}")
            return

    # 选择特征列：排除 Domain/name/Label/L1/L2/L3
    feat_cols = [c for c in df.columns if c not in ["Domain", "name", "Label", "L1", "L2", "L3"]]
    if not feat_cols:
        log("[ERR] no feature columns detected.")
        return

    # 中位数填补（保守处理）
    df[feat_cols] = df[feat_cols].apply(lambda s: s.fillna(s.median()), axis=0)

    # 标准化
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, copy=False)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    xgb_dev_params = get_xgb_device_params(args.xgb_device)
    log(f"[XGB] device={args.xgb_device} -> {xgb_dev_params}")

    # ----------------- 监督：三层 XGB -----------------
    cls_rows = []
    for level, labcol in [("Level1", "L1"), ("Level2", "L2"), ("Level3", "L3")]:
        y = df[labcol].astype(str).to_numpy()
        uniq = pd.Index(sorted(np.unique(y)))
        lut = {k: i for i, k in enumerate(uniq)}
        yi = np.array([lut[v] for v in y], dtype=np.int32)

        acc, f1m = xgb_cv(
            X, yi, num_class=len(uniq), seed=args.seed,
            xgb_device_params=xgb_dev_params,
            grid_trials=args.grid_trials,
            device_choice=args.xgb_device,
        )
        cls_rows.append({
            "Representation": "CY+rich",
            "Level": level,
            "Accuracy": acc,
            "MacroF1": f1m,
        })
        log(f"[XGB] {level}: Acc={acc:.4f}, MacroF1={f1m:.4f}")

        # 混淆矩阵（sanity）
        Xtr, Xte, ytr, yte = train_test_split(
            X, yi, test_size=0.2,
            stratify=yi, random_state=args.seed
        )
        cw = compute_class_weights(ytr)
        sw = np.array([cw[c] for c in ytr], dtype=np.float32)

        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(uniq),
            eval_metric="mlogloss",
            random_state=0,
            **xgb_dev_params,
        )
        clf.fit(Xtr, ytr, sample_weight=sw, verbose=False)
        yp = clf.predict(Xte)

        cm = pd.DataFrame(
            confusion_matrix(yte, yp, labels=np.arange(len(uniq))),
            index=[f"T:{c}" for c in uniq],
            columns=[f"P:{c}" for c in uniq],
        )
        if as_bool(args.save_cm):
            cm.to_csv(os.path.join(args.outdir, f"confusion_matrix_{level}.csv"))

    pd.DataFrame(cls_rows).to_csv(
        os.path.join(args.outdir, "classification_metrics_xgb.csv"),
        index=False
    )
    log("[OK] classification_metrics_xgb.csv saved")

    # ----------------- 无监督聚类（子采样） -----------------
    rng = np.random.default_rng(args.seed)
    subN = min(args.clu_max_n, len(df))
    idx = np.sort(rng.choice(np.arange(len(df)), size=subN, replace=False))
    Xc = X[idx]
    sub_df = df.iloc[idx].reset_index(drop=True)

    clu_rows = []
    for level, labcol in [("Level1", "L1"), ("Level2", "L2"), ("Level3", "L3")]:
        y = sub_df[labcol].astype(str).to_numpy()
        k = len(np.unique(y))

        # KMeans
        km = KMeans(n_clusters=k, n_init=args.kmeans_inits, random_state=args.seed)
        ykm = km.fit_predict(Xc)
        nmi = normalized_mutual_info_score(y, ykm)
        ari = adjusted_rand_score(y, ykm)
        _, _, v = homogeneity_completeness_v_measure(y, ykm)
        pur = cluster_purity(y, ykm)
        clu_rows.append({
            "Representation": "CY+rich",
            "Level": level,
            "Method": "KMeans",
            "NMI": nmi,
            "ARI": ari,
            "V_measure": v,
            "Purity": pur,
        })

        # HAC-ward
        if as_bool(args.run_hac_ward):
            yw = try_agg(Xc, k, linkage="ward", metric="euclidean")
            nmi = normalized_mutual_info_score(y, yw)
            ari = adjusted_rand_score(y, yw)
            _, _, v = homogeneity_completeness_v_measure(y, yw)
            pur = cluster_purity(y, yw)
            clu_rows.append({
                "Representation": "CY+rich",
                "Level": level,
                "Method": "HAC-ward",
                "NMI": nmi,
                "ARI": ari,
                "V_measure": v,
                "Purity": pur,
            })

        # HAC-cosine
        if as_bool(args.run_hac_cos):
            Xn = normalize(Xc)
            yc = try_agg(Xn, k, linkage="average", metric="cosine")
            nmi = normalized_mutual_info_score(y, yc)
            ari = adjusted_rand_score(y, yc)
            _, _, v = homogeneity_completeness_v_measure(y, yc)
            pur = cluster_purity(y, yc)
            clu_rows.append({
                "Representation": "CY+rich",
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
            if mode == "rbf" and len(sub_df) > 5000:
                log("[Spectral] RBF on >5k is heavy; switch to nearest.")
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
                "Representation": "CY+rich",
                "Level": level,
                "Method": f"Spectral({mode})",
                "NMI": nmi,
                "ARI": ari,
                "V_measure": v,
                "Purity": pur,
            })

    pd.DataFrame(clu_rows).to_csv(
        os.path.join(args.outdir, "clustering_metrics_plus.csv"),
        index=False
    )
    log("[OK] clustering_metrics_plus.csv saved")


if __name__ == "__main__":
    main()

