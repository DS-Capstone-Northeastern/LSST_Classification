"""
Train a classical SSM feature extractor + classifier using explicit split ledgers.

Mirrors TFT strategy:
  - Use train_ids.csv and val_ids.csv ledgers for data separation (no random split).
  - Scan multiple lightcurve CSVs with memory-safe chunk filtering to those IDs.
  - Use inverse-frequency sample weights (analogue to WeightedRandomSampler).
  - Exclude leaky/static columns via static_metadata.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

from ssm.ssm_features import extract_ssm_features_matrix
from ssm.static_metadata import StaticMetadataPreprocessor


def parse_csv_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def compute_global_dt_scale(df_lc: pd.DataFrame) -> float:
    df_sorted = df_lc[["object_id", "mjd"]].sort_values(["object_id", "mjd"])
    dt = df_sorted.groupby("object_id")["mjd"].diff()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1.0
    return float(max(np.median(dt.to_numpy(dtype=np.float64)), 1e-6))


def load_metadata_for_ids(
    *,
    base_dir: Path,
    metadata_files: Sequence[str],
    object_ids: Sequence[int],
    label_col: str,
    chunksize: int = 0,
) -> pd.DataFrame:
    obj_set = set(map(int, object_ids))
    frames: List[pd.DataFrame] = []
    # Read full metadata (we still filter later); missing optional columns handled by preprocessor.
    for name in metadata_files:
        path = base_dir / name
        if chunksize and chunksize > 0:
            reader = pd.read_csv(path, chunksize=chunksize)
            for chunk in reader:
                sub = chunk[chunk["object_id"].isin(obj_set)]
                if not sub.empty:
                    frames.append(sub)
        else:
            df = pd.read_csv(path)
            sub = df[df["object_id"].isin(obj_set)]
            if not sub.empty:
                frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["object_id"], keep="first")


def load_lightcurves_for_ids(
    *,
    base_dir: Path,
    lightcurve_files: Sequence[str],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    chunksize: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["object_id", "mjd", "passband", "flux", "flux_err"]
    dtypes = {
        "object_id": "int64",
        "mjd": "float64",
        "passband": "int16",
        "flux": "float64",
        "flux_err": "float64",
    }
    train_set = set(map(int, train_ids))
    val_set = set(map(int, val_ids))

    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []

    for name in lightcurve_files:
        path = base_dir / name
        print(f"Loading lightcurves: {path}")
        reader = pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=chunksize)
        for chunk in reader:
            if chunk.empty:
                continue
            # Filter rows for each split
            if train_set:
                sub_train = chunk[chunk["object_id"].isin(train_set)]
                if not sub_train.empty:
                    train_frames.append(sub_train)
            if val_set:
                sub_val = chunk[chunk["object_id"].isin(val_set)]
                if not sub_val.empty:
                    val_frames.append(sub_val)
 

    df_train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=cols)
    df_val = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame(columns=cols)
    return df_train, df_val


def compute_kpis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes_physical: np.ndarray,
) -> Dict[str, float]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    n_classes = int(y_proba.shape[1])

    # TFT-style weighted log-loss computed for two different special sets:
    #  - short:   {64, 99}
    #  - extended:{64, 99, 991, 993}
    class_double = 2.0

    special_short = {64, 99}
    class_weights_short = np.array(
        [class_double if int(cls) in special_short else 1.0 for cls in classes_physical],
        dtype=np.float64,
    )
    sample_weight_short = class_weights_short[y_true]
    weighted_log_loss_short = float(
        log_loss(
            y_true,
            y_proba,
            sample_weight=sample_weight_short,
            labels=np.arange(n_classes),
        )
    )

    special_extended = {64, 99, 991, 993}
    class_weights_extended = np.array(
        [class_double if int(cls) in special_extended else 1.0 for cls in classes_physical],
        dtype=np.float64,
    )
    sample_weight_extended = class_weights_extended[y_true]
    weighted_log_loss_extended = float(
        log_loss(
            y_true,
            y_proba,
            sample_weight=sample_weight_extended,
            labels=np.arange(n_classes),
        )
    )

    y_onehot = np.zeros_like(y_proba, dtype=np.float64)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    # macro PR-AUC (one-vs-rest)
    from sklearn.metrics import average_precision_score

    macro_pr_auc = float(average_precision_score(y_onehot, y_proba, average="macro"))

    multiclass_brier = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))

    return {
        "weighted_log_loss_64_99": weighted_log_loss_short,
        "weighted_log_loss_64_99_991_993": weighted_log_loss_extended,
        # For option-1 "w64w99" runs we keep the metric on the short special-set.
        "weighted_log_loss": weighted_log_loss_short,
        "macro_f1": macro_f1,
        "macro_pr_auc": macro_pr_auc,
        "multiclass_brier_score": multiclass_brier,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, required=True, help="Data folder containing plasticc_*lightcurves*.csv + metadata csvs")
    ap.add_argument("--splits-dir", type=str, required=True, help="Folder containing train_ids.csv / val_ids.csv / test_ids.csv")
    ap.add_argument("--train-ids-file", type=str, default="train_ids.csv")
    ap.add_argument("--val-ids-file", type=str, default="val_ids.csv")
    ap.add_argument("--label-col", type=str, default="true_target")
    ap.add_argument("--metadata-files", type=str, default="plasticc_train_metadata.csv,plasticc_test_metadata.csv")
    ap.add_argument(
        "--lightcurve-files",
        type=str,
        default=(
            "plasticc_train_lightcurves.csv,"
            "plasticc_test_set_batch1.csv,plasticc_test_set_batch2.csv,plasticc_test_set_batch3.csv,"
            "plasticc_test_set_batch4.csv,plasticc_test_set_batch5.csv,plasticc_test_set_batch6.csv,"
            "plasticc_test_set_batch7.csv,plasticc_test_set_batch8.csv,plasticc_test_set_batch9.csv,"
            "plasticc_test_set_batch10.csv,plasticc_test_set_batch11.csv"
        ),
    )
    ap.add_argument("--chunksize", type=int, default=250000)
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--r-min", type=float, default=1e-3)
    ap.add_argument(
        "--dt-scale",
        type=float,
        default=0.0,
        help="Override dt_scale used by Kalman/SSM. If >0, skips auto-computation.",
    )
    ap.add_argument("--rf-n-estimators", type=int, default=200)
    ap.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="RandomForest parallelism (set to 1 to avoid OpenMP shared-memory issues).",
    )
    ap.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "xgboost"])
    ap.add_argument("--xgb-n-estimators", type=int, default=600)
    ap.add_argument("--xgb-max-depth", type=int, default=8)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.05)
    ap.add_argument("--xgb-subsample", type=float, default=0.8)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ap.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=1,
        help="XGBoost parallelism (set to 1 to avoid OpenMP shared-memory issues).",
    )
    ap.add_argument("--max-train-ids", type=int, default=0, help="If >0, limit to first N train IDs (speed smoke-test).")
    ap.add_argument("--max-val-ids", type=int, default=0, help="If >0, limit to first N val IDs (speed smoke-test).")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = pd.read_csv(splits_dir / args.train_ids_file)["object_id"].astype(np.int64).tolist()
    val_ids = pd.read_csv(splits_dir / args.val_ids_file)["object_id"].astype(np.int64).tolist()
    if args.max_train_ids and args.max_train_ids > 0:
        train_ids = train_ids[: int(args.max_train_ids)]
    if args.max_val_ids and args.max_val_ids > 0:
        val_ids = val_ids[: int(args.max_val_ids)]
    print(f"Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)}")

    metadata_files = parse_csv_list(args.metadata_files)
    lightcurve_files = parse_csv_list(args.lightcurve_files)

    print("Loading metadata for train split...")
    train_meta = load_metadata_for_ids(
        base_dir=base_dir,
        metadata_files=metadata_files,
        object_ids=train_ids,
        label_col=args.label_col,
    )
    print("Loading metadata for val split...")
    val_meta = load_metadata_for_ids(
        base_dir=base_dir,
        metadata_files=metadata_files,
        object_ids=val_ids,
        label_col=args.label_col,
    )

    print("Loading lightcurves for train/val IDs (chunked)...")
    train_lc, val_lc = load_lightcurves_for_ids(
        base_dir=base_dir,
        lightcurve_files=lightcurve_files,
        train_ids=train_ids,
        val_ids=val_ids,
        chunksize=args.chunksize,
    )
    print(f"Train LC rows: {len(train_lc)} | Val LC rows: {len(val_lc)}")

    dt_scale_auto = compute_global_dt_scale(train_lc)
    dt_scale = float(args.dt_scale) if float(args.dt_scale) > 0.0 else dt_scale_auto
    print(f"Using dt_scale: {dt_scale} (auto={dt_scale_auto})")

    print("Fitting static metadata preprocessor (no-leak)...")
    static_pre = StaticMetadataPreprocessor.fit(
        train_meta,
        label_col=args.label_col,
        object_id_col="object_id",
        feature_cols=None,
        exclude_exact={"target", "true_target", "object_id"},
        exclude_prefixes=("true_", "sim_"),
    )

    train_obj_ids_static, X_static_train = static_pre.transform(train_meta)
    val_obj_ids_static, X_static_val = static_pre.transform(val_meta)

    # Label encoder based on train labels only
    train_id_to_label = dict(zip(train_meta["object_id"].astype(np.int64), train_meta[args.label_col]))
    y_train_raw = np.array([train_id_to_label.get(int(oid), -1) for oid in train_ids], dtype=np.int64)

    # Some IDs might be missing from metadata; drop them consistently
    valid_train_mask = y_train_raw != -1
    train_ids_filtered = np.array(train_ids, dtype=np.int64)[valid_train_mask]
    y_train_raw = y_train_raw[valid_train_mask]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    print(f"Encoded {len(le.classes_)} classes.")

    # For val: build arrays aligned to val_ids
    val_id_to_label = dict(zip(val_meta["object_id"].astype(np.int64), val_meta[args.label_col]))
    y_val_raw = np.array([val_id_to_label.get(int(oid), -1) for oid in val_ids], dtype=np.int64)
    valid_val_mask = y_val_raw != -1
    val_ids_filtered = np.array(val_ids, dtype=np.int64)[valid_val_mask]
    y_val_raw = y_val_raw[valid_val_mask]
    # Filter out val labels not seen in train encoder
    known = set(le.classes_.tolist())
    known_mask = np.array([int(v) in known for v in y_val_raw], dtype=bool)
    val_ids_filtered = val_ids_filtered[known_mask]
    y_val_raw = y_val_raw[known_mask]
    y_val = le.transform(y_val_raw)

    # Extract SSM features
    print("Extracting SSM features for train objects...")
    X_ssm_train = extract_ssm_features_matrix(
        train_lc,
        object_ids=train_ids_filtered,
        dt_scale=dt_scale,
        q=args.q,
        r_min=args.r_min,
    )
    print("Extracting SSM features for val objects...")
    X_ssm_val = extract_ssm_features_matrix(
        val_lc,
        object_ids=val_ids_filtered,
        dt_scale=dt_scale,
        q=args.q,
        r_min=args.r_min,
    )

    # Align static features to filtered object orders
    static_train_map = {int(oid): i for i, oid in enumerate(train_obj_ids_static.tolist())}
    static_val_map = {int(oid): i for i, oid in enumerate(val_obj_ids_static.tolist())}

    X_static_train_aligned = np.stack([X_static_train[static_train_map[int(oid)]] for oid in train_ids_filtered], axis=0)
    X_static_val_aligned = np.stack([X_static_val[static_val_map[int(oid)]] for oid in val_ids_filtered], axis=0)

    X_train = np.hstack([X_static_train_aligned, X_ssm_train]).astype(np.float64, copy=False)
    X_val = np.hstack([X_static_val_aligned, X_ssm_val]).astype(np.float64, copy=False)

    # Training-time class emphasis (Option 1, "w64w99" style).
    # - Metric is short special-set {64, 99}.
    # - Training sample weights are also short special-set {64, 99}.
    physical_labels_for_train = le.classes_[y_train]
    special = {64, 99}
    sample_weight = np.array(
        [2.0 if int(lbl) in special else 1.0 for lbl in physical_labels_for_train],
        dtype=np.float64,
    )

    models = {}
    if args.model_type == "random_forest":
        models["random_forest"] = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=args.rf_n_jobs,
            random_state=42,
        )
    elif args.model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Install it with: ./myenv/bin/pip install xgboost"
            ) from e
        # sample_weight handles class imbalance; objective is multiclass softprob.
        models["xgboost"] = XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=args.xgb_n_jobs,
        )

    best_name = None
    best_model = None
    best_kpis = None
    best_report = None
    best_acc = None

    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        kpis = compute_kpis(y_val, y_pred, y_proba, classes_physical=le.classes_)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        acc = float(accuracy_score(y_val, y_pred))

        print(
            f"{name} -> Acc {acc:.5f}, Macro-F1 {kpis['macro_f1']:.5f}, "
            f"WLogLoss {kpis['weighted_log_loss']:.5f}"
        )

        if best_kpis is None or kpis["macro_f1"] > best_kpis["macro_f1"]:
            best_name = name
            best_model = clf
            best_kpis = kpis
            best_report = report
            best_acc = acc

    bundle = {
        "classifier": best_model,
        "model_name": best_name,
        "static_preprocessor": static_pre,
        "label_encoder": le,
        "dt_scale": dt_scale,
        "q": args.q,
        "r_min": args.r_min,
        "feature_dim_static": int(X_static_train_aligned.shape[1]),
        "feature_dim_ssm": int(X_ssm_train.shape[1]),
    }
    joblib.dump(bundle, out_dir / "ssm_classifier_bundle.joblib")

    payload = {
        "best_model_name": best_name,
        "accuracy": best_acc,
        "macro_f1": best_kpis["macro_f1"],
        "weighted_log_loss": best_kpis["weighted_log_loss"],
        "weighted_log_loss_64_99": best_kpis["weighted_log_loss_64_99"],
        "weighted_log_loss_64_99_991_993": best_kpis["weighted_log_loss_64_99_991_993"],
        "macro_pr_auc": best_kpis["macro_pr_auc"],
        "multiclass_brier_score": best_kpis["multiclass_brier_score"],
        "n_train_samples": int(X_train.shape[0]),
        "n_val_samples": int(X_val.shape[0]),
        "dt_scale": dt_scale,
        "q": args.q,
        "r_min": args.r_min,
        "classification_report": best_report,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved bundle + metrics to: {out_dir}")


if __name__ == "__main__":
    main()

