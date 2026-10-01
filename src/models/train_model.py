#!/usr/bin/env python3
"""Tune tree models with chronological validation and quantify small-sample uncertainty."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.balanced_xgb import BalancedXGBClassifier

DATA_FILE = ROOT / "data" / "processed" / "flood_training_24h.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "flood_model.pkl"
REPORT_FILE = MODEL_DIR / "evaluation.json"
DIAGNOSTICS_FILE = MODEL_DIR / "feature_diagnostics.json"

CATEGORICAL_FEATURES = ["station_id", "current_risk_label"]
NUMERIC_FEATURES = [
    "level_minus_danger_m",
    "water_level_change_24h_m",
    "rainfall_24h_mm",
    "rainfall_24_48h_mm",
    "rainfall_48_72h_mm",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "risk_label_24h"
ALL_LABELS = ["Normal", "Above Normal", "Severe", "Extreme"]
PARAM_GRID = {
    "classifier__n_estimators": [100, 250],
    "classifier__max_depth": [None, 5],
    "classifier__min_samples_leaf": [2, 4],
    "classifier__max_features": ["sqrt", 0.8],
}
XGB_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [1, 2, 3, 4],
    "classifier__learning_rate": [0.02, 0.05, 0.1],
    "classifier__min_child_weight": [1, 3, 5, 8],
    "classifier__gamma": [0.0, 0.25, 1.0],
    "classifier__subsample": [0.7, 0.85, 1.0],
    "classifier__colsample_bytree": [0.7, 0.85, 1.0],
    "classifier__reg_lambda": [1.0, 5.0, 15.0],
    "classifier__reg_alpha": [0.0, 0.1, 1.0],
    "classifier__class_weight_strength": [0.0, 0.25, 0.5, 1.0],
}
XGB_SEARCH_ITERATIONS = 32


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and express river level relative to station thresholds."""
    frame = data.copy()
    # Today's observed category is known at prediction time and is a valid persistence/context feature.
    frame["current_risk_label"] = frame["risk_label"]
    frame["level_minus_danger_m"] = frame["water_level_m"] - frame["danger_level_m"]
    # Use disjoint rainfall blocks rather than nested sums to reduce collinearity.
    frame["rainfall_24_48h_mm"] = frame["rainfall_48h_mm"] - frame["rainfall_24h_mm"]
    frame["rainfall_48_72h_mm"] = frame["rainfall_72h_mm"] - frame["rainfall_48h_mm"]
    return frame


def make_model(use_pca: bool = False) -> Pipeline:
    if use_pca:
        numeric = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
            ]
        )
    else:
        numeric = "passthrough"
    transform = ColumnTransformer(
        [
            ("station", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_xgb_model() -> Pipeline:
    transform = ColumnTransformer(
        [
            ("station", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", transform),
            ("classifier", BalancedXGBClassifier(random_state=42, n_jobs=1)),
        ]
    )


def vif_report(values: pd.DataFrame) -> dict[str, float | None]:
    """Calculate numeric VIFs; return None for a perfectly redundant feature."""
    x = values.astype(float).to_numpy()
    result: dict[str, float | None] = {}
    for index, name in enumerate(values.columns):
        y = x[:, index]
        others = np.delete(x, index, axis=1)
        design = np.column_stack([np.ones(len(y)), others])
        predicted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = float(np.sum((y - y.mean()) ** 2))
        residual = float(np.sum((y - predicted) ** 2))
        if total == 0 or residual <= total * 1e-12:
            result[name] = None
        else:
            result[name] = round(1.0 / (1.0 - (1.0 - residual / total)), 3)
    return result


def make_date_cv(train: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Create expanding-window folds, keeping every station on a date together."""
    dates = np.array(sorted(train["date"].unique()))
    splitter = TimeSeriesSplit(n_splits=4, test_size=14)
    folds = []
    for train_date_idx, valid_date_idx in splitter.split(dates):
        train_dates = dates[train_date_idx]
        valid_dates = dates[valid_date_idx]
        train_rows = np.flatnonzero(train["date"].isin(train_dates).to_numpy())
        valid_rows = np.flatnonzero(train["date"].isin(valid_dates).to_numpy())
        folds.append((train_rows, valid_rows))
    return folds, len(dates)


def score_cv(model: Pipeline, x: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    cv_labels = [label for label in ALL_LABELS if label in set(y)]
    metrics = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": make_scorer(f1_score, average="macro", labels=cv_labels, zero_division=0),
    }
    scores = cross_validate(model, x, y, cv=folds, scoring=metrics, n_jobs=1)
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
            "fold_scores": [float(value) for value in scores[f"test_{metric}"]],
        }
        for metric in metrics
    }


def tune(x: pd.DataFrame, y: pd.Series, folds, use_pca: bool) -> GridSearchCV:
    cv_labels = [label for label in ALL_LABELS if label in set(y)]
    search = GridSearchCV(
        estimator=make_model(use_pca=use_pca),
        param_grid=PARAM_GRID,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": make_scorer(f1_score, average="macro", labels=cv_labels, zero_division=0),
        },
        refit="accuracy",
        cv=folds,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x, y)
    return search


def tune_xgboost(x: pd.DataFrame, y: pd.Series, folds) -> RandomizedSearchCV:
    cv_labels = [label for label in ALL_LABELS if label in set(y)]
    search = RandomizedSearchCV(
        estimator=make_xgb_model(),
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=XGB_SEARCH_ITERATIONS,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": make_scorer(f1_score, average="macro", labels=cv_labels, zero_division=0),
        },
        refit="accuracy",
        cv=folds,
        n_jobs=1,
        random_state=42,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x, y)
    return search


def candidate_metrics(search) -> dict:
    index = search.best_index_
    n_folds = 0
    while f"split{n_folds}_test_accuracy" in search.cv_results_:
        n_folds += 1
    fold_scores = [
        float(search.cv_results_[f"split{fold}_test_accuracy"][index])
        for fold in range(n_folds)
    ]
    return {
        "cv_accuracy_mean": float(np.mean(fold_scores)),
        "cv_accuracy_std": float(np.std(fold_scores)),
        "cv_accuracy_95_ci": cv_accuracy_interval(fold_scores),
        "cv_accuracy_fold_scores": fold_scores,
        "cv_macro_f1_mean": float(search.cv_results_["mean_test_macro_f1"][index]),
        "best_params": search.best_params_,
    }


def cv_accuracy_interval(scores: list[float], confidence: float = 0.95) -> list[float]:
    """Approximate t interval over chronological fold scores (descriptive only)."""
    values = np.asarray(scores, dtype=float)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    margin = critical * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return [max(0.0, float(values.mean() - margin)), min(1.0, float(values.mean() + margin))]


def date_block_bootstrap_accuracy_interval(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    *,
    block_days: int = 3,
    replicates: int = 5000,
    seed: int = 42,
) -> list[float]:
    """Bootstrap holdout accuracy by contiguous date blocks to respect clustering."""
    date_values = pd.to_datetime(dates).dt.normalize().to_numpy()
    unique_dates = np.array(sorted(pd.unique(date_values)))
    row_indices = [np.flatnonzero(date_values == date) for date in unique_dates]
    block_days = min(block_days, len(unique_dates))
    blocks_per_sample = int(np.ceil(len(unique_dates) / block_days))
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    boot_scores = np.empty(replicates)
    for replicate in range(replicates):
        starts = rng.integers(0, len(unique_dates) - block_days + 1, size=blocks_per_sample)
        sampled_dates = np.concatenate([
            np.arange(start, start + block_days) for start in starts
        ])[: len(unique_dates)]
        sampled_rows = np.concatenate([row_indices[index] for index in sampled_dates])
        boot_scores[replicate] = float(correct[sampled_rows].mean())
    lower, upper = np.quantile(boot_scores, [0.025, 0.975])
    return [float(lower), float(upper)]


def main() -> None:
    data = pd.read_csv(DATA_FILE, parse_dates=["date", "target_date"])
    data = data.dropna(subset=[
        "station_id", "water_level_m", "water_level_change_24h_m", "rainfall_24h_mm",
        "rainfall_48h_mm", "rainfall_72h_mm", "danger_level_m", "hfl_m", "risk_label", TARGET,
    ]).sort_values(["date", "station_id"]).reset_index(drop=True)
    data = make_features(data)

    # Keep the latest 20% of unique dates untouched until final evaluation.
    unique_dates = np.array(sorted(data["date"].unique()))
    holdout_start_index = int(len(unique_dates) * 0.8)
    holdout_start = pd.Timestamp(unique_dates[holdout_start_index])
    train = data[data["date"] < holdout_start].copy().reset_index(drop=True)
    test = data[data["date"] >= holdout_start].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("The chronological split produced an empty train or test set.")

    cv_folds, cv_date_count = make_date_cv(train)
    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]

    # Compare the prior-style baseline to a small, explicit parameter grid.
    baseline = make_model(use_pca=False).set_params(
        classifier__n_estimators=400,
        classifier__min_samples_leaf=3,
        classifier__max_features="sqrt",
    )
    baseline_cv = score_cv(baseline, x_train, y_train, cv_folds)

    print("Tuning Random Forest without PCA across four chronological folds...")
    tuned = tune(x_train, y_train, cv_folds, use_pca=False)
    print("Tuning PCA + Random Forest across the same chronological folds...")
    pca_tuned = tune(x_train, y_train, cv_folds, use_pca=True)
    print("Tuning class-weighted XGBoost across the same chronological folds...")
    xgb_tuned = tune_xgboost(x_train, y_train, cv_folds)

    candidates = [
        ("tuned_random_forest", tuned),
        ("pca_random_forest", pca_tuned),
        ("tuned_xgboost", xgb_tuned),
    ]
    # Use the one-standard-error rule: if scores are within fold variation,
    # prefer the simpler, more interpretable no-PCA forest.
    cv_accuracy = {
        name: float(search.cv_results_["mean_test_accuracy"][search.best_index_])
        for name, search in candidates
    }
    best_name = max(cv_accuracy, key=cv_accuracy.get)
    best_search = dict(candidates)[best_name]
    best_se = float(best_search.cv_results_["std_test_accuracy"][best_search.best_index_]) / np.sqrt(len(cv_folds))
    eligible = [name for name, score in cv_accuracy.items() if score >= cv_accuracy[best_name] - best_se]
    selected_name = "tuned_random_forest" if "tuned_random_forest" in eligible else best_name
    selected = dict(candidates)[selected_name]
    model = selected.best_estimator_  # GridSearchCV refit uses all pre-holdout dates.
    predictions = model.predict(x_test)
    labels = [label for label in ALL_LABELS if label in set(y_test) | set(predictions)]
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0))
    persistence_predictions = test["risk_label"]
    persistence_accuracy = float(accuracy_score(y_test, persistence_predictions))
    persistence_macro_f1 = float(f1_score(y_test, persistence_predictions, labels=labels, average="macro", zero_division=0))

    candidate_diag = {
        "tuned_random_forest": candidate_metrics(tuned),
        "pca_random_forest": candidate_metrics(pca_tuned),
        "tuned_xgboost": candidate_metrics(xgb_tuned),
    }

    holdout_comparison = {}
    for name, search in candidates:
        candidate_prediction = search.best_estimator_.predict(x_test)
        holdout_comparison[name] = {
            "accuracy": float(accuracy_score(y_test, candidate_prediction)),
            "accuracy_95_ci_date_block_bootstrap": date_block_bootstrap_accuracy_interval(
                y_test, candidate_prediction, test["date"]
            ),
            "macro_f1": float(f1_score(y_test, candidate_prediction, labels=labels, average="macro", zero_division=0)),
        }

    persistence_cv_scores = [
        float(accuracy_score(train.iloc[valid_idx][TARGET], train.iloc[valid_idx]["risk_label"]))
        for _, valid_idx in cv_folds
    ]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    evaluation = {
        "model": "BalancedXGBClassifier" if selected_name == "tuned_xgboost" else "RandomForestClassifier",
        "selected_candidate": selected_name,
        "pca_used": selected_name == "pca_random_forest",
        "best_params": selected.best_params_,
        "prediction_horizon_hours": 24,
        "split_type": "chronological 80/20 by unique date; all station rows on one date stay together",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "cv": {
            "method": "expanding-window TimeSeriesSplit, 4 folds, 14 validation dates per fold",
            "dates_available_for_cv": int(cv_date_count),
            "baseline": baseline_cv,
            "persistence_baseline_accuracy": {
                "fold_scores": persistence_cv_scores,
                "mean": float(np.mean(persistence_cv_scores)),
                "std": float(np.std(persistence_cv_scores)),
                "approximate_95_ci": cv_accuracy_interval(persistence_cv_scores),
            },
            "candidate_comparison": candidate_diag,
            "selection_rule": "Choose by chronological CV accuracy; within one standard error prefer the non-PCA Random Forest for simplicity. XGBoost uses randomized hyperparameter search.",
            "confidence_interval_note": "Approximate 95% Student-t intervals use only four chronological fold scores. The folds are temporally ordered and share training history, so intervals are descriptive, not independent-sample guarantees.",
        },
        "holdout_candidate_comparison": holdout_comparison,
        "holdout_interval_note": "Approximate 95% moving date-block bootstrap intervals use 3-day blocks over only the 23 later dates. This holdout has already been viewed in prior model iterations and is exploratory, not a fresh final test.",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "persistence_baseline_holdout": {
            "accuracy": persistence_accuracy,
            "accuracy_95_ci_date_block_bootstrap": date_block_bootstrap_accuracy_interval(
                y_test, persistence_predictions.to_numpy(), test["date"]
            ),
            "macro_f1": persistence_macro_f1,
            "description": "Predict the next-day category equals the observed current-day risk_label.",
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": report,
        "target_classes_in_data": sorted(data[TARGET].unique().tolist()),
        "classes_absent_from_data": sorted(set(ALL_LABELS) - set(data[TARGET])),
        "feature_columns": FEATURES,
        "caveat": "Only 228 examples from two stations in one short period. Scores are not operational flood-warning validation.",
        "class_score_note": "Model predict_proba outputs are uncalibrated class scores, not calibrated probabilities of flooding.",
    }
    REPORT_FILE.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    correlations = data[NUMERIC_FEATURES].corr().fillna(0)
    diagnostics = {
        "rows": int(len(data)),
        "numeric_features": NUMERIC_FEATURES,
        "removed_exact_duplicate": "rainfall_mm was omitted because rainfall_24h_mm is exactly the same for daily data.",
        "relative_level_features": "water_level_m is expressed relative to danger_level_m. Raw danger/HFL thresholds and absolute water level are not separate model inputs.",
        "rainfall_window_features": "Nested 24/48/72-hour totals are converted to non-overlapping 0-24, 24-48, and 48-72-hour blocks to reduce collinearity.",
        "pearson_correlation": {
            row: {column: round(float(correlations.loc[row, column]), 3) for column in correlations.columns}
            for row in correlations.index
        },
        "numeric_vif": vif_report(data[NUMERIC_FEATURES]),
        "vif_note": "VIF is mainly a linear-model diagnostic. Random Forests do not require independent inputs; the rainfall blocks here are non-overlapping and remaining correlations are small.",
        "leakage_audit": {
            "target": TARGET,
            "target_derived_from": "next consecutive day's river level compared with configured warning/danger/HFL thresholds",
            "future_fields_excluded_from_inputs": ["target_date", "target_water_level_m", "risk_label_24h"],
            "current_risk_label": "Included as current_risk_label, calculated only from the selected date's observed river level and configured thresholds; it is known at prediction time.",
        },
    }
    DIAGNOSTICS_FILE.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Selected: {selected_name}; PCA used: {evaluation['pca_used']}")
    print(f"Best parameters: {selected.best_params_}")
    print(f"Holdout: {len(test)} rows from {evaluation['test_start']} through {evaluation['test_end']}")
    print(f"Test accuracy: {accuracy:.1%}; macro F1: {macro_f1:.3f}; balanced accuracy: {evaluation['balanced_accuracy']:.3f}")
    print(f"Persistence baseline accuracy: CV {evaluation['cv']['persistence_baseline_accuracy']['mean']:.1%}; holdout {persistence_accuracy:.1%}")
    print("CV candidate comparison:")
    for name, values in candidate_diag.items():
        ci_low, ci_high = values["cv_accuracy_95_ci"]
        print(f"  {name}: accuracy={values['cv_accuracy_mean']:.3f} +/- {values['cv_accuracy_std']:.3f}; approximate CV 95% CI [{ci_low:.3f}, {ci_high:.3f}]; macro_f1={values['cv_macro_f1_mean']:.3f}")
    print("Later-date holdout candidate comparison:")
    for name, values in holdout_comparison.items():
        ci_low, ci_high = values["accuracy_95_ci_date_block_bootstrap"]
        print(f"  {name}: accuracy={values['accuracy']:.3f}; date-block bootstrap 95% CI [{ci_low:.3f}, {ci_high:.3f}]; macro_f1={values['macro_f1']:.3f}")
    print(f"Classes absent from the full dataset: {', '.join(evaluation['classes_absent_from_data']) or 'none'}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved evaluation: {REPORT_FILE}")
    print(f"Saved feature diagnostics: {DIAGNOSTICS_FILE}")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
