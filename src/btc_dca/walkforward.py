from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from .features import FEATURE_COLUMNS


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    feature_importance: pd.DataFrame


def classification_metrics(y: np.ndarray, p: np.ndarray, decision_threshold: float = 0.35) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"n": 0, "positive_rate": np.nan, "unsafe_recall": np.nan,
                "unsafe_precision": np.nan, "decision_threshold": decision_threshold,
                "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan}
    predicted = p > decision_threshold
    true_positive = int(((y == 1) & predicted).sum())
    positives = int((y == 1).sum())
    predicted_positive = int(predicted.sum())
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "unsafe_recall": true_positive / positives if positives else np.nan,
        "unsafe_precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "decision_threshold": float(decision_threshold),
        "pr_auc": float(average_precision_score(y, p)) if positives else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
    }


def hysteresis(probability: pd.Series, on_threshold: float = 0.25,
               off_threshold: float = 0.35, confirmation_bars: int = 2) -> pd.Series:
    state = False
    low_count = 0
    states: list[bool] = []
    for value in probability.to_numpy(dtype=float):
        if not np.isfinite(value):
            state = False
            low_count = 0
        elif value > off_threshold:
            state = False
            low_count = 0
        elif value < on_threshold:
            low_count += 1
            if low_count >= confirmation_bars:
                state = True
        else:
            low_count = 0
        states.append(state)
    return pd.Series(states, index=probability.index, name="ml_on")


def _make_estimator(kind: str):
    if kind == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    if kind == "lightgbm":
        return LGBMClassifier(
            objective="binary", n_estimators=180, learning_rate=0.05,
            num_leaves=15, max_depth=5, min_child_samples=50,
            reg_lambda=1.0, verbosity=-1, n_jobs=-1, random_state=271828,
        )
    raise ValueError(f"Unknown estimator: {kind}")


def _transform(kind: str, train_x: np.ndarray, other_x: np.ndarray):
    if kind != "logistic":
        return train_x, other_x, None
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(other_x), scaler


def _fit_predict(kind: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, object]:
    tx, vx, scaler = _transform(kind, train_x, test_x)
    model = _make_estimator(kind)
    model.fit(tx, train_y)
    probability = model.predict_proba(vx)[:, 1]
    return probability, (model, scaler)


def _purged_oof(kind: str, x: np.ndarray, y: np.ndarray, purge: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered expanding predictions used only to fit a sigmoid calibrator."""
    n = len(y)
    initial = max(300, int(n * 0.45))
    remaining = n - initial
    if remaining < 300:
        return np.array([], dtype=float), np.array([], dtype=int)
    edges = np.linspace(initial, n, 4, dtype=int)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        fit_end = max(0, lo - purge)
        if fit_end < 200 or hi <= lo or len(np.unique(y[:fit_end])) < 2:
            continue
        p, _ = _fit_predict(kind, x[:fit_end], y[:fit_end], x[lo:hi])
        probs.append(p)
        labels.append(y[lo:hi])
    if not probs:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(probs), np.concatenate(labels)


def _sigmoid_calibrator(oof_p: np.ndarray, oof_y: np.ndarray):
    if len(oof_y) < 200 or len(np.unique(oof_y)) < 2:
        return None
    logits = np.log(np.clip(oof_p, 1e-6, 1 - 1e-6) / np.clip(1 - oof_p, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    calibrator.fit(logits.reshape(-1, 1), oof_y)
    return calibrator


def _calibrate(p: np.ndarray, calibrator) -> np.ndarray:
    if calibrator is None:
        return np.asarray(p, dtype=float)
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def walk_forward(
    dataset: pd.DataFrame,
    test_start: str = "2022-04-01",
    train_months: int = 24,
    validation_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    purge_bars: int = 42,
) -> WalkForwardResult:
    """Run fixed-parameter logistic and LightGBM expanding OOF calibration and quarterly tests."""
    clean = dataset.dropna(subset=FEATURE_COLUMNS).copy()
    labels_available = clean["unsafe"].notna()
    end_time = clean.index.max()
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    offset = pd.DateOffset(months=step_months)
    rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    fold_id = 0

    while test_start_ts <= end_time:
        fold_id += 1
        test_end_ts = min(test_start_ts + pd.DateOffset(months=test_months), end_time + pd.Timedelta(seconds=1))
        validation_start = test_start_ts - pd.DateOffset(months=validation_months)
        train_start = max(clean.index.min(), validation_start - pd.DateOffset(months=train_months))

        train_mask = (clean.index >= train_start) & (clean.index < validation_start - pd.Timedelta(hours=4 * purge_bars)) & labels_available
        val_mask = (clean.index >= validation_start) & (clean.index < test_start_ts - pd.Timedelta(hours=4 * purge_bars)) & labels_available
        test_mask = (clean.index >= test_start_ts) & (clean.index < test_end_ts)
        train = clean.loc[train_mask]
        val = clean.loc[val_mask]
        test = clean.loc[test_mask]
        if len(train) < 1000 or len(val) < 200 or test.empty or train["unsafe"].nunique() < 2:
            test_start_ts += offset
            continue

        x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train["unsafe"].to_numpy(dtype=int)
        x_val = val[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_val = val["unsafe"].to_numpy(dtype=int)
        x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
        labeled_test = test["unsafe"].notna().to_numpy()
        y_test = test.loc[labeled_test, "unsafe"].to_numpy(dtype=int)

        for kind in ("logistic", "lightgbm"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_p, oof_y = _purged_oof(kind, x_train, y_train, purge_bars)
                calibrator = _sigmoid_calibrator(oof_p, oof_y)
                val_raw, _ = _fit_predict(kind, x_train, y_train, x_val)
                test_raw, fitted = _fit_predict(kind, x_train, y_train, x_test)
            val_p = _calibrate(val_raw, calibrator)
            test_p = _calibrate(test_raw, calibrator)
            val_metrics = classification_metrics(y_val, val_p)
            test_metrics = classification_metrics(y_test, test_p[labeled_test])
            row = {
                "fold": fold_id, "model": kind, "train_start": train_start.isoformat(),
                "validation_start": validation_start.isoformat(), "test_start": test_start_ts.isoformat(),
                "test_end": test_end_ts.isoformat(), "train_rows": len(train), "validation_rows": len(val),
                "test_rows": len(test), "purge_bars": purge_bars, "calibrator": "purged_train_oof_sigmoid" if calibrator else "identity_fallback",
            }
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row.update({f"test_{k}": v for k, v in test_metrics.items()})
            fold_rows.append(row)
            pred = pd.DataFrame({
                "timestamp": test.index, "fold": fold_id, "model": kind,
                "p_unsafe": test_p, "unsafe": test["unsafe"].to_numpy(),
            })
            rows.append(pred)
            model, _ = fitted
            if kind == "lightgbm":
                importance_rows.extend({"fold": fold_id, "feature": name, "importance": int(value)}
                                       for name, value in zip(FEATURE_COLUMNS, model.feature_importances_))
        test_start_ts += offset

    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["timestamp", "fold", "model", "p_unsafe", "unsafe"]
    )
    folds = pd.DataFrame(fold_rows)
    importance = pd.DataFrame(importance_rows)
    return WalkForwardResult(predictions, folds, importance)
