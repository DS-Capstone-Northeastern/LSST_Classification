#!/usr/bin/env python3
"""
Hybrid PLAsTiCC-style classifier:
1) MLP embeddings + feature importance on tabular metadata
2) Kalman filter (state-space) feature extraction on light curves
3) Fusion into one row/object
4) Final classifier on fused features

Default expected files in data/:
- data/plasticc_train_metadata.csv
- data/plasticc_test_metadata.csv
- data/plasticc_train_lightcurves.csv
- data/plasticc_test_set_batch*.csv (all batches are automatically concatenated)

You can override paths with CLI arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from metrics import (
    macro_f1,
    macro_pr_auc,
    multiclass_brier_score,
    plasticc_log_loss,
)
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def log_step(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def identity(x: np.ndarray) -> np.ndarray:
    return x


_ACTIVATION_MAP = {
    "relu": relu,
    "tanh": tanh,
    "logistic": logistic,
    "identity": identity,
}


def mlp_last_hidden_embedding(mlp: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden activations from sklearn MLP weights.
    Returns the last hidden layer activation as embedding.
    """
    # Self-contained activations (no reliance on module-level helpers or _ACTIVATION_MAP),
    # so notebook cells can run out of order without NameError.
    act_fn = {
        "relu": lambda t: np.maximum(t, 0.0),
        "tanh": np.tanh,
        "logistic": lambda t: 1.0 / (1.0 + np.exp(-t)),
        "identity": lambda t: t,
    }[mlp.activation]
    h = x.copy()
    n_layers = len(mlp.coefs_)
    for i in range(n_layers - 1):  # skip output layer
        h = h @ mlp.coefs_[i] + mlp.intercepts_[i]
        h = act_fn(h)
    return h


def _safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def kalman_features_one_series(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    q_level: float = 1e-3,
    q_trend: float = 1e-4,
    r_floor: float = 1e-3,
) -> Dict[str, float]:
    """
    2D local linear trend state-space model:
    state = [level, trend]
    observation = level + noise
    """
    n = len(y)
    if n == 0:
        return {
            "n_obs": 0.0,
            "ll_mean": 0.0,
            "ll_sum": 0.0,
            "innov_mean": 0.0,
            "innov_std": 0.0,
            "norm_innov_abs_mean": 0.0,
            "final_level": 0.0,
            "final_trend": 0.0,
            "level_std": 0.0,
            "trend_std": 0.0,
        }
    if n == 1:
        val = float(y[0])
        return {
            "n_obs": 1.0,
            "ll_mean": 0.0,
            "ll_sum": 0.0,
            "innov_mean": 0.0,
            "innov_std": 0.0,
            "norm_innov_abs_mean": 0.0,
            "final_level": val,
            "final_trend": 0.0,
            "level_std": 0.0,
            "trend_std": 0.0,
        }

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    yerr = np.maximum(yerr[order], 1e-6)

    # Initialization from first two points
    dt0 = max(float(t[1] - t[0]), 1e-3)
    x = np.array([y[0], (y[1] - y[0]) / dt0], dtype=float)
    p = np.diag([10.0, 10.0]).astype(float)
    h = np.array([[1.0, 0.0]])  # observe level

    innovations: List[float] = []
    norm_innov_abs: List[float] = []
    loglik: List[float] = []
    levels: List[float] = []
    trends: List[float] = []

    for i in range(n):
        if i == 0:
            dt = 1e-3
        else:
            dt = max(float(t[i] - t[i - 1]), 1e-3)

        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        q = np.array(
            [[q_level * dt, 0.0], [0.0, q_trend * max(dt, 1e-3)]], dtype=float
        )
        r = float(yerr[i] ** 2 + r_floor)

        # Predict
        x_pred = f @ x
        p_pred = f @ p @ f.T + q

        # Update
        y_pred = float((h @ x_pred.reshape(-1, 1))[0, 0])
        innov = float(y[i] - y_pred)
        s = float((h @ p_pred @ h.T)[0, 0] + r)
        s = max(s, 1e-9)
        k = (p_pred @ h.T) / s
        x = x_pred + k.flatten() * innov
        p = (np.eye(2) - k @ h) @ p_pred

        innovations.append(innov)
        norm_innov_abs.append(abs(innov) / np.sqrt(s))
        loglik.append(-0.5 * (np.log(2.0 * np.pi * s) + (innov**2) / s))
        levels.append(float(x[0]))
        trends.append(float(x[1]))

    return {
        "n_obs": float(n),
        "ll_mean": float(np.mean(loglik)),
        "ll_sum": float(np.sum(loglik)),
        "innov_mean": float(np.mean(innovations)),
        "innov_std": float(np.std(innovations)),
        "norm_innov_abs_mean": float(np.mean(norm_innov_abs)),
        "final_level": float(levels[-1]),
        "final_trend": float(trends[-1]),
        "level_std": float(np.std(levels)),
        "trend_std": float(np.std(trends)),
    }


def kalman_features_for_object(df_obj: pd.DataFrame) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    per_pb_feats: List[Dict[str, float]] = []

    for pb, df_pb in df_obj.groupby("passband", sort=True):
        f = kalman_features_one_series(
            t=df_pb["mjd"].to_numpy(dtype=float),
            y=df_pb["flux"].to_numpy(dtype=float),
            yerr=df_pb["flux_err"].to_numpy(dtype=float),
        )
        for k, v in f.items():
            feat[f"pb{pb}_{k}"] = v
        per_pb_feats.append(f)

    if per_pb_feats:
        keys = per_pb_feats[0].keys()
        for k in keys:
            vals = np.array([d[k] for d in per_pb_feats], dtype=float)
            feat[f"agg_{k}_mean"] = float(vals.mean())
            feat[f"agg_{k}_std"] = float(vals.std())
            feat[f"agg_{k}_min"] = float(vals.min())
            feat[f"agg_{k}_max"] = float(vals.max())
    else:
        for k in [
            "n_obs",
            "ll_mean",
            "ll_sum",
            "innov_mean",
            "innov_std",
            "norm_innov_abs_mean",
            "final_level",
            "final_trend",
            "level_std",
            "trend_std",
        ]:
            feat[f"agg_{k}_mean"] = 0.0
            feat[f"agg_{k}_std"] = 0.0
            feat[f"agg_{k}_min"] = 0.0
            feat[f"agg_{k}_max"] = 0.0

    # Extra non-SSM light curve summary features
    flux = df_obj["flux"].to_numpy(dtype=float)
    flux_err = df_obj["flux_err"].to_numpy(dtype=float)
    feat.update(_safe_stats(flux, "flux"))
    feat.update(_safe_stats(flux_err, "flux_err"))
    feat["snr_mean"] = float(np.mean(np.abs(flux) / np.maximum(flux_err, 1e-6))) if len(flux) else 0.0
    feat["n_total_obs"] = float(len(df_obj))
    feat["n_passbands"] = float(df_obj["passband"].nunique())
    return feat


def build_lightcurve_feature_table(lightcurve_df: pd.DataFrame) -> pd.DataFrame:
    log_step("Starting Kalman/SSM feature extraction per object.")
    rows: List[Dict[str, float]] = []
    ids: List[int] = []
    total_objects = int(lightcurve_df["object_id"].nunique())
    for idx, (object_id, grp) in enumerate(
        lightcurve_df.groupby("object_id", sort=False), start=1
    ):
        rows.append(kalman_features_for_object(grp))
        ids.append(object_id)
        if idx == 1 or idx % 1000 == 0 or idx == total_objects:
            log_step(
                f"Kalman features progress: {idx}/{total_objects} objects processed."
            )
    out = pd.DataFrame(rows)
    out.insert(0, "object_id", ids)
    log_step("Completed Kalman/SSM feature extraction.")
    return out


def sample_metadata(
    df: pd.DataFrame,
    id_col: str,
    max_objects: int | None,
    sample_random_state: int,
    dataset_name: str,
) -> pd.DataFrame:
    if max_objects is None or max_objects <= 0 or len(df) <= max_objects:
        log_step(f"{dataset_name}: using all {len(df)} objects.")
        return df
    log_step(
        f"{dataset_name}: sampling {max_objects} objects out of {len(df)} "
        f"(random_state={sample_random_state})."
    )
    sampled = df.sample(n=max_objects, random_state=sample_random_state)
    sampled = sampled.sort_values(id_col).reset_index(drop=True)
    return sampled


def filter_lightcurve_by_object_ids(
    lc_df: pd.DataFrame,
    id_col: str,
    object_ids: set,
    dataset_name: str,
) -> pd.DataFrame:
    before = len(lc_df)
    lc_df = lc_df[lc_df[id_col].isin(object_ids)].copy()
    after = len(lc_df)
    log_step(f"{dataset_name}: filtered rows {before} -> {after} using object subset.")
    return lc_df


def load_test_lightcurves_subset(
    test_lc: Path | None,
    test_lc_glob: str,
    id_col: str,
    selected_object_ids: set | None,
    max_test_batches: int | None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Memory-friendly loader for very large test lightcurve sets.
    - Can load from a single file OR many batch files.
    - Optional object_id filtering during chunk reads.
    - Optional cap on number of batch files for faster experimentation.
    """
    if test_lc is not None:
        log_step(f"Loading single test lightcurve file with chunking: {test_lc}")
        frames: List[pd.DataFrame] = []
        total_rows = 0
        kept_rows = 0
        for chunk in pd.read_csv(test_lc, chunksize=chunksize):
            total_rows += len(chunk)
            if selected_object_ids is not None:
                chunk = chunk[chunk[id_col].isin(selected_object_ids)]
            kept_rows += len(chunk)
            frames.append(chunk)
        log_step(f"Single-file test LC rows read={total_rows}, kept={kept_rows}.")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    batch_paths = sorted(Path().glob(test_lc_glob))
    if not batch_paths:
        raise FileNotFoundError(
            f"No test lightcurve files found for pattern: {test_lc_glob}"
        )
    if max_test_batches is not None and max_test_batches > 0:
        batch_paths = batch_paths[:max_test_batches]
        log_step(f"Using first {len(batch_paths)} test batch files for fast run.")
    else:
        log_step(f"Using all {len(batch_paths)} test batch files.")

    all_frames: List[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    for batch_idx, batch_path in enumerate(batch_paths, start=1):
        log_step(f"Reading test batch {batch_idx}/{len(batch_paths)}: {batch_path}")
        for chunk_idx, chunk in enumerate(
            pd.read_csv(batch_path, chunksize=chunksize), start=1
        ):
            total_rows += len(chunk)
            if selected_object_ids is not None:
                chunk = chunk[chunk[id_col].isin(selected_object_ids)]
            kept_rows += len(chunk)
            all_frames.append(chunk)
            if chunk_idx == 1 or chunk_idx % 10 == 0:
                log_step(
                    f"  batch {batch_idx}, chunk {chunk_idx}: total_rows={total_rows}, kept_rows={kept_rows}"
                )
    log_step(f"Finished loading test LC. rows read={total_rows}, kept={kept_rows}.")
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def build_lightcurve_feature_table_from_csv_stream(
    csv_path: Path,
    id_col: str,
    allowed_object_ids: set | None = None,
    chunksize: int = 1_000_000,
    dataset_name: str = "lightcurves",
) -> pd.DataFrame:
    """
    Stream a large lightcurve CSV and compute per-object Kalman features without
    loading the full file in memory.
    Assumes rows are mostly grouped by object_id; handles chunk boundaries safely.
    """
    log_step(f"{dataset_name}: streaming feature extraction from {csv_path}")
    carryover = pd.DataFrame()
    rows: List[Dict[str, float]] = []
    ids: List[int] = []
    processed_objects = 0
    seen_rows = 0

    usecols = [id_col, "mjd", "passband", "flux", "flux_err"]
    for chunk_idx, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize), start=1
    ):
        seen_rows += len(chunk)
        if not carryover.empty:
            chunk = pd.concat([carryover, chunk], ignore_index=True)
            carryover = pd.DataFrame()

        if chunk.empty:
            continue

        last_obj = chunk[id_col].iloc[-1]
        process_mask = chunk[id_col] != last_obj
        process_chunk = chunk[process_mask]
        carryover = chunk[~process_mask].copy()

        for object_id, grp in process_chunk.groupby(id_col, sort=False):
            if allowed_object_ids is not None and object_id not in allowed_object_ids:
                continue
            feat = kalman_features_for_object(grp)
            rows.append(feat)
            ids.append(object_id)
            processed_objects += 1

        if chunk_idx == 1 or chunk_idx % 5 == 0:
            log_step(
                f"{dataset_name}: chunk {chunk_idx}, rows_seen={seen_rows}, "
                f"objects_done={processed_objects}"
            )

    if not carryover.empty:
        for object_id, grp in carryover.groupby(id_col, sort=False):
            if allowed_object_ids is not None and object_id not in allowed_object_ids:
                continue
            feat = kalman_features_for_object(grp)
            rows.append(feat)
            ids.append(object_id)
            processed_objects += 1

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[id_col])
    else:
        out.insert(0, id_col, ids)
    log_step(f"{dataset_name}: completed. objects_with_features={processed_objects}")
    return out


def build_test_lightcurve_features_microbatched(
    test_lc: Path | None,
    test_lc_glob: str,
    id_col: str,
    allowed_object_ids: set | None,
    max_test_batches: int | None,
    chunksize: int,
) -> pd.DataFrame:
    """
    RAM-safe micro-batched feature extraction across test files.
    """
    if test_lc is not None:
        return build_lightcurve_feature_table_from_csv_stream(
            csv_path=test_lc,
            id_col=id_col,
            allowed_object_ids=allowed_object_ids,
            chunksize=chunksize,
            dataset_name="Test lightcurves(single file)",
        )

    batch_paths = sorted(Path().glob(test_lc_glob))
    if not batch_paths:
        raise FileNotFoundError(
            f"No test lightcurve files found for pattern: {test_lc_glob}"
        )
    if max_test_batches is not None and max_test_batches > 0:
        batch_paths = batch_paths[:max_test_batches]
    log_step(
        f"Commencing RAM-safe micro-batched inference across {len(batch_paths)} test batch files."
    )
    feat_tables: List[pd.DataFrame] = []
    for i, p in enumerate(batch_paths, start=1):
        log_step(f"Processing test batch {i}/{len(batch_paths)}: {p.name}")
        feat_tables.append(
            build_lightcurve_feature_table_from_csv_stream(
                csv_path=p,
                id_col=id_col,
                allowed_object_ids=allowed_object_ids,
                chunksize=chunksize,
                dataset_name=f"Test {p.name}",
            )
        )
    if feat_tables:
        return pd.concat(feat_tables, ignore_index=True)
    return pd.DataFrame(columns=[id_col])


def compute_mlp_oof_embeddings(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
    - train_oof_proba (n_train, n_classes)
    - train_oof_embed (n_train, emb_dim)
    - test_proba_avg (n_test, n_classes)
    - test_embed_avg (n_test, emb_dim)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_classes = int(np.max(y_train)) + 1

    mlp_template = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )

    train_oof_proba = np.zeros((x_train.shape[0], n_classes), dtype=float)
    train_oof_embed = np.zeros((x_train.shape[0], 32), dtype=float)

    test_proba_folds: List[np.ndarray] = []
    test_embed_folds: List[np.ndarray] = []

    log_step("Starting OOF MLP training for embeddings.")
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_train), start=1):
        log_step(
            f"MLP fold {fold_idx}/{n_splits}: train={len(tr_idx)} val={len(va_idx)}"
        )
        model = clone(mlp_template)
        model.fit(x_train[tr_idx], y_train[tr_idx])
        log_step(f"MLP fold {fold_idx}: training complete, generating predictions.")

        train_oof_proba[va_idx] = model.predict_proba(x_train[va_idx])
        train_oof_embed[va_idx] = mlp_last_hidden_embedding(model, x_train[va_idx])

        test_proba_folds.append(model.predict_proba(x_test))
        test_embed_folds.append(mlp_last_hidden_embedding(model, x_test))
        log_step(f"MLP fold {fold_idx}: done.")

    test_proba_avg = np.mean(test_proba_folds, axis=0)
    test_embed_avg = np.mean(test_embed_folds, axis=0)
    log_step("Completed OOF MLP training and embedding extraction.")
    return train_oof_proba, train_oof_embed, test_proba_avg, test_embed_avg


def main() -> None:
    log_step("Parsing command-line arguments.")
    parser = argparse.ArgumentParser(description="Hybrid PLAsTiCC-style pipeline")
    parser.add_argument(
        "--train-meta",
        type=Path,
        default=Path("data/plasticc_train_metadata.csv"),
        help="Train metadata CSV path.",
    )
    parser.add_argument(
        "--test-meta",
        type=Path,
        default=Path("data/plasticc_test_metadata.csv"),
        help="Test metadata CSV path.",
    )
    parser.add_argument(
        "--train-lc",
        type=Path,
        default=Path("data/plasticc_train_lightcurves.csv"),
        help="Train lightcurve CSV path.",
    )
    parser.add_argument(
        "--test-lc",
        type=Path,
        default=None,
        help="Optional single test lightcurve CSV path.",
    )
    parser.add_argument(
        "--test-lc-glob",
        type=str,
        default="data/plasticc_test_set_batch*.csv",
        help="Glob pattern for test lightcurve batch files when --test-lc is not set.",
    )
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--id-col", type=str, default="object_id")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-max-objects",
        type=int,
        default=0,
        help="Max number of train metadata objects (for faster iteration). Use <=0 for all.",
    )
    parser.add_argument(
        "--test-max-objects",
        type=int,
        default=0,
        help="Max number of test metadata objects (for faster iteration). Use <=0 for all.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=0,
        help="Max number of test lightcurve batch files to read. Use <=0 for all batches.",
    )
    parser.add_argument(
        "--lc-chunksize",
        type=int,
        default=1_000_000,
        help="Chunk size for streaming lightcurve CSV processing.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random state for train/test object sampling.",
    )
    parser.add_argument(
        "--exclude-leakage-features",
        action="store_true",
        default=True,
        help="Exclude leakage-prone simulation truth columns (true_* and tflux_*).",
    )
    parser.add_argument(
        "--include-all-features",
        action="store_true",
        default=False,
        help="Override leakage exclusion and include all metadata columns except id/target.",
    )
    args = parser.parse_args()
    log_step(f"Arguments parsed. Output dir: {args.outdir}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_step("Output directory is ready.")

    log_step(f"Loading train metadata: {args.train_meta}")
    train_meta = pd.read_csv(args.train_meta)
    log_step(f"Loading test metadata: {args.test_meta}")
    test_meta = pd.read_csv(args.test_meta)

    if args.id_col not in train_meta.columns or args.id_col not in test_meta.columns:
        raise ValueError(f"ID column '{args.id_col}' must exist in train_meta and test_meta.")
    if args.target_col not in train_meta.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in train_meta.")

    # Fast subset mode for large data
    train_meta = sample_metadata(
        train_meta,
        id_col=args.id_col,
        max_objects=args.train_max_objects,
        sample_random_state=args.sample_random_state,
        dataset_name="Train metadata",
    )
    test_meta = sample_metadata(
        test_meta,
        id_col=args.id_col,
        max_objects=args.test_max_objects,
        sample_random_state=args.sample_random_state,
        dataset_name="Test metadata",
    )
    train_object_ids = set(train_meta[args.id_col].tolist())
    test_object_ids = set(test_meta[args.id_col].tolist())

    log_step(f"Loading train lightcurves: {args.train_lc}")
    train_lc = pd.read_csv(args.train_lc)
    train_lc = filter_lightcurve_by_object_ids(
        train_lc,
        id_col=args.id_col,
        object_ids=train_object_ids,
        dataset_name="Train lightcurves",
    )

    log_step("Building test lightcurve features (RAM-safe streaming mode).")
    test_lc_feats = build_test_lightcurve_features_microbatched(
        test_lc=args.test_lc,
        test_lc_glob=args.test_lc_glob,
        id_col=args.id_col,
        allowed_object_ids=test_object_ids,
        max_test_batches=args.max_test_batches if args.max_test_batches > 0 else None,
        chunksize=args.lc_chunksize,
    )
    log_step(
        "Loaded all inputs. "
        f"train_meta={train_meta.shape}, test_meta={test_meta.shape}, "
        f"train_lc={train_lc.shape}, test_lc_feats={test_lc_feats.shape}"
    )

    required_lc_cols = {args.id_col, "mjd", "passband", "flux", "flux_err"}
    missing_train_lc = required_lc_cols - set(train_lc.columns)
    if missing_train_lc:
        raise ValueError(
            f"Missing light curve columns in train_lc. missing={missing_train_lc}"
        )

    # ----- Tabular preprocessing -----
    log_step("Starting tabular preprocessing.")
    tabular_cols = [c for c in train_meta.columns if c not in [args.id_col, args.target_col]]
    if args.exclude_leakage_features and not args.include_all_features:
        tabular_cols = [
            c for c in tabular_cols if not (c.startswith("true_") or c.startswith("tflux_"))
        ]
        log_step(
            "Leakage-safe mode enabled: excluded metadata columns starting with true_ and tflux_."
        )
    x_tab_train_df = train_meta[tabular_cols].copy()
    x_tab_test_df = test_meta[tabular_cols].copy()

    # Numeric-only default. If you have categoricals, encode before this script.
    x_tab_train_df = x_tab_train_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_tab_test_df = x_tab_test_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_raw = train_meta[args.target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_labels = list(le.classes_)

    scaler = StandardScaler()
    x_tab_train = scaler.fit_transform(x_tab_train_df.values)
    x_tab_test = scaler.transform(x_tab_test_df.values)
    log_step(
        f"Tabular preprocessing complete. feature_count={x_tab_train.shape[1]}, "
        f"train_samples={x_tab_train.shape[0]}, test_samples={x_tab_test.shape[0]}"
    )

    # ----- MLP OOF embeddings -----
    log_step("Starting MLP branch (OOF embeddings + probabilities).")
    oof_proba, oof_embed, test_proba, test_embed = compute_mlp_oof_embeddings(
        x_train=x_tab_train,
        y_train=y,
        x_test=x_tab_test,
        n_splits=args.n_splits,
        random_state=args.seed,
    )

    mlp_oof_logloss = log_loss(y, np.clip(oof_proba, 1e-8, 1.0))
    print(f"MLP OOF logloss: {mlp_oof_logloss:.6f}")
    mlp_oof_plasticc = plasticc_log_loss(y_raw, np.clip(oof_proba, 1e-8, 1.0), class_labels)
    print(f"MLP OOF PLAsTiCC weighted logloss: {mlp_oof_plasticc:.6f}")

    # Fit one full MLP for permutation importance report
    log_step("Training full-data MLP for permutation feature importance.")
    mlp_full = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    mlp_full.fit(x_tab_train, y)
    imp = permutation_importance(
        mlp_full,
        x_tab_train,
        y,
        scoring="neg_log_loss",
        n_repeats=5,
        random_state=args.seed,
        n_jobs=-1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": tabular_cols,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(args.outdir / "mlp_tabular_feature_importance.csv", index=False)
    log_step("Saved MLP tabular feature importance.")

    # ----- Kalman/SSM light-curve features -----
    log_step("Starting Kalman/SSM branch for train lightcurves.")
    train_lc_feats = build_lightcurve_feature_table(train_lc)
    log_step("Starting Kalman/SSM branch for test lightcurves.")
    log_step(
        f"Kalman feature tables ready. train={train_lc_feats.shape}, test={test_lc_feats.shape}"
    )

    # ----- Merge branches -----
    log_step("Merging metadata IDs with Kalman features.")
    train_merged = train_meta[[args.id_col]].merge(
        train_lc_feats, on=args.id_col, how="left"
    )
    test_merged = test_meta[[args.id_col]].merge(test_lc_feats, on=args.id_col, how="left")
    train_merged = train_merged.fillna(0.0)
    test_merged = test_merged.fillna(0.0)

    x_lc_train = train_merged.drop(columns=[args.id_col]).to_numpy(dtype=float)
    x_lc_test = test_merged.drop(columns=[args.id_col]).to_numpy(dtype=float)
    log_step(
        f"Merged lightcurve feature matrices ready. train={x_lc_train.shape}, test={x_lc_test.shape}"
    )

    # Hybrid vector:
    # - original tabular (scaled)
    # - MLP embedding (OOF for train, averaged folds for test)
    # - Kalman lightcurve features
    x_hybrid_train = np.concatenate([x_tab_train, oof_embed, x_lc_train], axis=1)
    x_hybrid_test = np.concatenate([x_tab_test, test_embed, x_lc_test], axis=1)
    log_step(
        f"Hybrid matrices built. train={x_hybrid_train.shape}, test={x_hybrid_test.shape}"
    )

    # ----- Final classifier -----
    log_step("Starting final hybrid classifier training.")
    final_model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=args.seed,
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_final = np.zeros((x_hybrid_train.shape[0], len(le.classes_)), dtype=float)
    test_folds: List[np.ndarray] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(x_hybrid_train, y), start=1):
        log_step(
            f"Final model fold {fold_idx}/{args.n_splits}: "
            f"train={len(tr_idx)} val={len(va_idx)}"
        )
        model = clone(final_model)
        model.fit(x_hybrid_train[tr_idx], y[tr_idx])
        oof_final[va_idx] = model.predict_proba(x_hybrid_train[va_idx])
        test_folds.append(model.predict_proba(x_hybrid_test))
        log_step(f"Final model fold {fold_idx}: done.")

    final_oof_logloss = log_loss(y, np.clip(oof_final, 1e-8, 1.0))
    print(f"Final hybrid OOF logloss: {final_oof_logloss:.6f}")
    final_oof_plasticc = plasticc_log_loss(
        y_raw,
        np.clip(oof_final, 1e-8, 1.0),
        class_labels,
    )
    y_pred_idx = np.argmax(oof_final, axis=1)
    y_pred_labels = le.inverse_transform(y_pred_idx)
    final_macro_f1 = macro_f1(y_raw, y_pred_labels)
    final_macro_pr_auc = macro_pr_auc(
        y_raw,
        np.clip(oof_final, 1e-8, 1.0),
        class_labels,
    )
    final_brier = multiclass_brier_score(
        y_raw,
        np.clip(oof_final, 1e-8, 1.0),
        class_labels,
    )
    print(f"Final hybrid OOF PLAsTiCC weighted logloss: {final_oof_plasticc:.6f}")
    print(f"Final hybrid OOF macro F1: {final_macro_f1:.6f}")
    print(f"Final hybrid OOF macro PR-AUC: {final_macro_pr_auc:.6f}")
    print(f"Final hybrid OOF multiclass Brier: {final_brier:.6f}")

    test_pred = np.mean(test_folds, axis=0)
    log_step("Generated averaged test predictions from all folds.")

    # Optional genuine test-set evaluation when labels are available in test metadata
    test_set_metrics: Dict[str, float] = {}
    if args.target_col in test_meta.columns:
        log_step("Test metadata contains target labels. Computing final test-set metrics.")
        y_test_raw_all = test_meta[args.target_col].values
        # Use pandas isin for robust dtype handling (e.g., int vs float/object columns).
        valid_test_mask = pd.Series(y_test_raw_all).isin(set(class_labels)).to_numpy()
        if not np.all(valid_test_mask):
            dropped = int((~valid_test_mask).sum())
            log_step(
                f"Dropping {dropped} test rows with labels unseen in train classes."
            )
        y_test_raw = y_test_raw_all[valid_test_mask]
        test_pred_eval = test_pred[valid_test_mask]

        if len(y_test_raw) == 0:
            unseen_unique = pd.Series(y_test_raw_all).dropna().unique()[:20].tolist()
            log_step(
                "No valid test rows remain after class filtering. "
                "Skipping test metric computation."
            )
            test_set_metrics = {
                "test_eval_rows": 0,
                "test_metrics_skipped": True,
                "test_metrics_skip_reason": "No overlap between train classes and test labels.",
                "test_unique_labels_sample": unseen_unique,
            }
        else:
            y_test_pred_idx = np.argmax(test_pred_eval, axis=1)
            y_test_pred_labels = le.inverse_transform(y_test_pred_idx)

            test_set_metrics = {
                "test_plasticc_weighted_logloss": float(
                    plasticc_log_loss(y_test_raw, np.clip(test_pred_eval, 1e-8, 1.0), class_labels)
                ),
                "test_macro_f1": float(macro_f1(y_test_raw, y_test_pred_labels)),
                "test_macro_pr_auc": float(
                    macro_pr_auc(y_test_raw, np.clip(test_pred_eval, 1e-8, 1.0), class_labels)
                ),
                "test_multiclass_brier": float(
                    multiclass_brier_score(
                        y_test_raw, np.clip(test_pred_eval, 1e-8, 1.0), class_labels
                    )
                ),
                "test_eval_rows": int(len(y_test_raw)),
                "test_metrics_skipped": False,
            }
            print("\n--- FINAL TEST SET METRICS ---")
            print(
                f"PLAsTiCC Weighted Log-Loss: {test_set_metrics['test_plasticc_weighted_logloss']:.4f}"
            )
            print(f"Macro F1-Score:             {test_set_metrics['test_macro_f1']:.4f}")
            print(f"Macro PR-AUC:               {test_set_metrics['test_macro_pr_auc']:.4f}")
            print(f"Multiclass Brier Score:     {test_set_metrics['test_multiclass_brier']:.4f}")

    # Save outputs
    log_step("Saving metrics and output artifacts.")
    metrics = {
        "mlp_oof_logloss": float(mlp_oof_logloss),
        "mlp_oof_plasticc_weighted_logloss": float(mlp_oof_plasticc),
        "hybrid_oof_logloss": float(final_oof_logloss),
        "hybrid_oof_plasticc_weighted_logloss": float(final_oof_plasticc),
        "hybrid_oof_macro_f1": float(final_macro_f1),
        "hybrid_oof_macro_pr_auc": float(final_macro_pr_auc),
        "hybrid_oof_multiclass_brier": float(final_brier),
        "n_train": int(x_hybrid_train.shape[0]),
        "n_test": int(x_hybrid_test.shape[0]),
        "hybrid_dim": int(x_hybrid_train.shape[1]),
        "num_classes": int(len(le.classes_)),
        "class_labels": [int(c) if isinstance(c, (np.integer, int)) else c for c in class_labels],
        "exclude_leakage_features": bool(args.exclude_leakage_features and not args.include_all_features),
    }
    metrics.update(test_set_metrics)
    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log_step("Saved metrics.json.")

    embed_cols = [f"mlp_emb_{i}" for i in range(oof_embed.shape[1])]
    train_embed_df = pd.DataFrame(oof_embed, columns=embed_cols)
    train_embed_df.insert(0, args.id_col, train_meta[args.id_col].values)
    train_embed_df.to_csv(args.outdir / "train_mlp_embeddings.csv", index=False)
    log_step("Saved train_mlp_embeddings.csv.")

    test_embed_df = pd.DataFrame(test_embed, columns=embed_cols)
    test_embed_df.insert(0, args.id_col, test_meta[args.id_col].values)
    test_embed_df.to_csv(args.outdir / "test_mlp_embeddings.csv", index=False)
    log_step("Saved test_mlp_embeddings.csv.")

    train_lc_feats.to_csv(args.outdir / "train_kalman_features.csv", index=False)
    test_lc_feats.to_csv(args.outdir / "test_kalman_features.csv", index=False)
    log_step("Saved Kalman feature CSVs.")

    proba_cols = [f"class_{c}" for c in le.classes_]
    sub = pd.DataFrame(test_pred, columns=proba_cols)
    sub.insert(0, args.id_col, test_meta[args.id_col].values)
    sub.to_csv(args.outdir / "test_predictions.csv", index=False)
    log_step("Saved test_predictions.csv.")

    print(f"Saved artifacts to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
