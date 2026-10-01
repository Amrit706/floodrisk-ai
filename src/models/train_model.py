#!/usr/bin/env python3
"""Audit features, tune Random Forests with date-based CV, and evaluate once on a later holdout."""

from __future__ import annotations

import json
from pathlib import Path

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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
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

    candidates = [
        ("tuned_random_forest", tuned),
        ("pca_random_forest", pca_tuned),
    ]
    # Use the one-standard-error rule: if the CV difference is within noise,
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
    probabilities = model.predict_proba(x_test)
    classes = list(model.named_steps["classifier"].classes_)
    labels = [label for label in ALL_LABELS if label in set(y_test) | set(predictions)]
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0))
    persistence_predictions = test["risk_label"]
    persistence_accuracy = float(accuracy_score(y_test, persistence_predictions))
    persistence_macro_f1 = float(f1_score(y_test, persistence_predictions, labels=labels, average="macro", zero_division=0))

    pca_diag = {
        "tuned_random_forest": {
            "cv_accuracy_mean": float(tuned.cv_results_["mean_test_accuracy"][tuned.best_index_]),
            "cv_accuracy_std": float(tuned.cv_results_["std_test_accuracy"][tuned.best_index_]),
            "cv_macro_f1_mean": float(tuned.cv_results_["mean_test_macro_f1"][tuned.best_index_]),
            "best_params": tuned.best_params_,
        },
        "pca_random_forest": {
            "cv_accuracy_mean": float(pca_tuned.cv_results_["mean_test_accuracy"][pca_tuned.best_index_]),
            "cv_accuracy_std": float(pca_tuned.cv_results_["std_test_accuracy"][pca_tuned.best_index_]),
            "cv_macro_f1_mean": float(pca_tuned.cv_results_["mean_test_macro_f1"][pca_tuned.best_index_]),
            "best_params": pca_tuned.best_params_,
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    evaluation = {
        "model": "RandomForestClassifier",
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
                "fold_scores": [
                    float(accuracy_score(train.iloc[valid_idx][TARGET], train.iloc[valid_idx]["risk_label"]))
                    for _, valid_idx in cv_folds
                ],
                "mean": float(np.mean([
                    accuracy_score(train.iloc[valid_idx][TARGET], train.iloc[valid_idx]["risk_label"])
                    for _, valid_idx in cv_folds
                ])),
                "std": float(np.std([
                    accuracy_score(train.iloc[valid_idx][TARGET], train.iloc[valid_idx]["risk_label"])
                    for _, valid_idx in cv_folds
                ])),
            },
            "candidate_comparison": pca_diag,
            "selection_rule": "One-standard-error rule for CV accuracy; prefer no PCA when its score is statistically close to PCA.",
        },
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "persistence_baseline_holdout": {
            "accuracy": persistence_accuracy,
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
        "class_score_note": "Random Forest predict_proba outputs are not calibrated probabilities of flooding.",
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
    for name, values in pca_diag.items():
        print(f"  {name}: accuracy={values['cv_accuracy_mean']:.3f} +/- {values['cv_accuracy_std']:.3f}; macro_f1={values['cv_macro_f1_mean']:.3f}")
    print(f"Classes absent from the full dataset: {', '.join(evaluation['classes_absent_from_data']) or 'none'}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved evaluation: {REPORT_FILE}")
    print(f"Saved feature diagnostics: {DIAGNOSTICS_FILE}")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
