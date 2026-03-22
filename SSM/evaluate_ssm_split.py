"""
Evaluate an SSM feature + classifier bundle on test_ids.csv ledgers.

Mirrors TFT Phase 6 intent:
  - Use test_ids.csv to restrict which objects to evaluate.
  - Scan the provided test lightcurve batch CSVs, filter rows to test IDs.
  - Extract SSM features, merge static features, predict, compute KPIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, log_loss

from ssm.ssm_features import extract_ssm_features_matrix


def parse_csv_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


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
    macro_pr_auc = float(average_precision_score(y_onehot, y_proba, average="macro"))

    multiclass_brier = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))

    return {
        "macro_f1": macro_f1,
        "weighted_log_loss_64_99": weighted_log_loss_short,
        "weighted_log_loss_64_99_991_993": weighted_log_loss_extended,
        # For option-1 "w64w99" runs we keep the main metric on the short special-set.
        "weighted_log_loss": weighted_log_loss_short,
        "macro_pr_auc": macro_pr_auc,
        "multiclass_brier_score": multiclass_brier,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-path", type=str, required=True)
    ap.add_argument("--base-dir", type=str, required=True, help="Data folder (contains plasticc_* csv files)")
    ap.add_argument("--splits-dir", type=str, required=True, help="Folder containing test_ids.csv")
    ap.add_argument("--ids-file", type=str, default="test_ids.csv")
    ap.add_argument("--metadata-files", type=str, default="plasticc_train_metadata.csv,plasticc_test_metadata.csv")
    ap.add_argument(
        "--lightcurve-files",
        type=str,
        default=(
            "plasticc_test_set_batch1.csv,plasticc_test_set_batch2.csv,plasticc_test_set_batch3.csv,"
            "plasticc_test_set_batch4.csv,plasticc_test_set_batch5.csv,plasticc_test_set_batch6.csv,"
            "plasticc_test_set_batch7.csv,plasticc_test_set_batch8.csv,plasticc_test_set_batch9.csv,"
            "plasticc_test_set_batch10.csv,plasticc_test_set_batch11.csv"
        ),
    )
    ap.add_argument("--label-col", type=str, default="true_target")
    ap.add_argument("--chunksize", type=int, default=0, help="If >0, reads lightcurves with chunksize and filters.")
    ap.add_argument(
        "--max-test-ids",
        type=int,
        default=0,
        help="Optional speed mode: limit number of object_ids from test_ids.csv (0 = all).",
    )
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle_path)
    clf = bundle["classifier"]
    static_pre = bundle["static_preprocessor"]
    label_encoder = bundle["label_encoder"]
    dt_scale = float(bundle["dt_scale"])
    q = float(bundle["q"])
    r_min = float(bundle["r_min"])

    base_dir = Path(args.base_dir)
    splits_dir = Path(args.splits_dir)

    test_ids_df = pd.read_csv(splits_dir / args.ids_file)
    ids_list = test_ids_df["object_id"].astype(np.int64).tolist()
    if args.max_test_ids and args.max_test_ids > 0:
        ids_list = ids_list[: int(args.max_test_ids)]
    test_ids: Set[int] = set(map(int, ids_list))
    print(f"Loaded test_ids: {len(test_ids)}")

    metadata_files = parse_csv_list(args.metadata_files)
    lightcurve_files = parse_csv_list(args.lightcurve_files)

    # Load metadata for evaluation IDs
    meta_frames: List[pd.DataFrame] = []
    for name in metadata_files:
        path = base_dir / name
        df = pd.read_csv(path, usecols=None)
        df = df[df["object_id"].isin(test_ids)]
        if not df.empty:
            meta_frames.append(df)
    if not meta_frames:
        raise RuntimeError("No metadata rows found for test_ids.")
    df_meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(subset=["object_id"])

    # Restrict and extract label + static features once; later align per batch.
    meta_obj_ids = df_meta["object_id"].astype(np.int64).to_numpy()
    y_meta_raw = df_meta[args.label_col].to_numpy()

    # Only labels seen during training
    known_labels = set(label_encoder.classes_.tolist())
    known_mask = np.array([int(v) in known_labels for v in y_meta_raw], dtype=bool)
    df_meta = df_meta[known_mask]
    meta_obj_ids = df_meta["object_id"].astype(np.int64).to_numpy()
    y_meta_raw = df_meta[args.label_col].to_numpy()

    # label_encoder.transform works on numpy arrays for sklearn LabelEncoder
    y_meta = label_encoder.transform(y_meta_raw)
    # For static preprocessing, fit-transform already stored in bundle.
    eval_obj_ids_static, X_static_all = static_pre.transform(df_meta)
    static_map = {int(oid): i for i, oid in enumerate(eval_obj_ids_static.tolist())}
    y_map = {int(df_meta.iloc[i]["object_id"]): int(y_meta[i]) for i in range(len(df_meta))}

    seen_obj_ids: Set[int] = set()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_proba_all: List[np.ndarray] = []

    for lc_name in lightcurve_files:
        path = base_dir / lc_name
        print(f"Processing lightcurves: {path.name}")

        cols = ["object_id", "mjd", "passband", "flux", "flux_err"]
        dtypes = {
            "object_id": "int64",
            "mjd": "float64",
            "passband": "int16",
            "flux": "float64",
            "flux_err": "float64",
        }

        if args.chunksize and args.chunksize > 0:
            frames: List[pd.DataFrame] = []
            reader = pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=args.chunksize)
            for chunk in reader:
                chunk = chunk[chunk["object_id"].isin(test_ids)]
                if not chunk.empty:
                    frames.append(chunk)
            if not frames:
                continue
            df_lc = pd.concat(frames, ignore_index=True)
        else:
            df_lc = pd.read_csv(path, usecols=cols, dtype=dtypes)
            df_lc = df_lc[df_lc["object_id"].isin(test_ids)]

        if df_lc.empty:
            continue

        obj_ids_in_batch = df_lc["object_id"].astype(np.int64).unique().tolist()
        obj_ids_new = [oid for oid in obj_ids_in_batch if int(oid) not in seen_obj_ids]
        if not obj_ids_new:
            continue

        # Mark seen
        for oid in obj_ids_new:
            seen_obj_ids.add(int(oid))

        # Extract SSM features for objects in this batch
        X_ssm = extract_ssm_features_matrix(
            df_lc,
            object_ids=obj_ids_new,
            dt_scale=dt_scale,
            q=q,
            r_min=r_min,
        )

        # Align static features and labels
        # Some object_ids in lightcurve batches may have been dropped during the
        # metadata filtering step (e.g., label not present in training label_encoder).
        # Filter to only those that exist in both maps.
        obj_ids_keep = [int(oid) for oid in obj_ids_new if int(oid) in static_map and int(oid) in y_map]
        if not obj_ids_keep:
            continue

        # Slice X_ssm to keep alignment with obj_ids_keep.
        keep_indices = [i for i, oid in enumerate(obj_ids_new) if int(oid) in static_map and int(oid) in y_map]

        X_static = np.stack([X_static_all[static_map[oid]] for oid in obj_ids_keep], axis=0)
        y_true = np.array([y_map[oid] for oid in obj_ids_keep], dtype=np.int64)

        X_ssm_keep = X_ssm[keep_indices]
        X = np.hstack([X_static, X_ssm_keep]).astype(np.float64, copy=False)

        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_proba_all.append(y_proba)

    if not y_true_all:
        raise RuntimeError("No predictions were generated (empty eval set after filtering).")

    y_true_arr = np.asarray(y_true_all, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred_all, dtype=np.int64)
    y_proba_arr = np.vstack(y_proba_all)

    kpis = compute_kpis(y_true_arr, y_pred_arr, y_proba_arr, classes_physical=label_encoder.classes_)
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)

    payload = {
        "n_samples": int(y_true_arr.shape[0]),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": kpis["macro_f1"],
        "weighted_log_loss": kpis["weighted_log_loss"],
        "macro_pr_auc": kpis["macro_pr_auc"],
        "multiclass_brier_score": kpis["multiclass_brier_score"],
        "classification_report": report,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {args.output_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

