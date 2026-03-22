"""
Static metadata preprocessing with anti-leakage rules.

We select numeric metadata columns and exclude:
  - id/label columns: object_id, true_target (label), target
  - all columns starting with "true_"
  - columns starting with "sim_"

Then:
  - median imputation
  - StandardScaler
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class StaticMetadataPreprocessor:
    label_col: str
    object_id_col: str
    feature_cols: List[str]
    medians_: np.ndarray
    scaler_: StandardScaler

    @classmethod
    def fit(
        cls,
        df_meta: pd.DataFrame,
        *,
        label_col: str,
        object_id_col: str = "object_id",
        feature_cols: Optional[Sequence[str]] = None,
        exclude_exact: Optional[Iterable[str]] = None,
        exclude_prefixes: Optional[Iterable[str]] = None,
    ) -> "StaticMetadataPreprocessor":
        if label_col not in df_meta.columns:
            raise KeyError(f"label_col={label_col} not found in metadata.")
        if object_id_col not in df_meta.columns:
            raise KeyError(f"object_id_col={object_id_col} not found in metadata.")

        if feature_cols is None:
            numeric_cols = df_meta.select_dtypes(include=[np.number]).columns.tolist()
            exact_exclude = {object_id_col, label_col, "target", "true_target"}
            if exclude_exact is not None:
                exact_exclude.update(exclude_exact)

            prefixes = ("true_", "sim_")
            if exclude_prefixes is not None:
                prefixes = tuple(exclude_prefixes)

            numeric_cols = [
                c
                for c in numeric_cols
                if c not in exact_exclude and not any(c.startswith(p) for p in prefixes)
            ]
            numeric_cols = sorted(numeric_cols)
            feature_cols = numeric_cols
        else:
            feature_cols = list(feature_cols)

        X_raw = df_meta[feature_cols]
        medians = X_raw.median(numeric_only=True).to_numpy(dtype=np.float64)
        X_filled = X_raw.fillna(dict(zip(feature_cols, medians)))

        scaler = StandardScaler()
        scaler.fit(X_filled.to_numpy(dtype=np.float64))

        return cls(
            label_col=label_col,
            object_id_col=object_id_col,
            feature_cols=list(feature_cols),
            medians_=medians,
            scaler_=scaler,
        )

    def transform(self, df_meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.object_id_col not in df_meta.columns:
            raise KeyError(f"object_id_col={self.object_id_col} not found in metadata.")
        for c in self.feature_cols:
            if c not in df_meta.columns:
                raise KeyError(f"Missing static feature column: {c}")

        X_raw = df_meta[self.feature_cols]
        X_filled = X_raw.fillna(dict(zip(self.feature_cols, self.medians_)))
        X_scaled = self.scaler_.transform(X_filled.to_numpy(dtype=np.float64))
        obj_ids = df_meta[self.object_id_col].to_numpy(dtype=np.int64)
        return obj_ids, X_scaled.astype(np.float64, copy=False)

