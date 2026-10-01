#!/usr/bin/env python3
"""Train and evaluate the FloodRiskAI 24-hour pilot model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "flood_training_24h.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "flood_model.pkl"
REPORT_FILE = MODEL_DIR / "evaluation.json"

CATEGORICAL_FEATURES = ["station_id"]
NUMERIC_FEATURES = [
    "water_level_m",
    "water_level_change_24h_m",
    "rainfall_mm",
    "rainfall_24h_mm",
    "rainfall_48h_mm",
    "rainfall_72h_mm",
    "danger_level_m",
    "hfl_m",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "risk_label_24h"


def main() -> None:
    data = pd.read_csv(DATA_FILE, parse_dates=["date", "target_date"])
    data = data.dropna(subset=FEATURES + [TARGET]).sort_values("date")
    unique_dates = sorted(data["date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("At least 10 unique dates are needed for a time-based holdout.")

    split_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.8)))
    cutoff = pd.Timestamp(unique_dates[split_index])
    train = data[data["date"] < cutoff]
    test = data[data["date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("The time-based split produced an empty train or test set.")

    transform = ColumnTransformer(
        [
            ("station", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("features", transform),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    # Single-process training also works in restricted Windows sandboxes.
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(train[FEATURES], train[TARGET])
    predictions = model.predict(test[FEATURES])
    classes = list(model.named_steps["classifier"].classes_)
    labels = list(dict.fromkeys(classes + sorted(set(test[TARGET]))))
    report = classification_report(
        test[TARGET], predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(test[TARGET], predictions, labels=labels).tolist()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    summary = {
        "model": "RandomForestClassifier",
        "prediction_horizon_hours": 24,
        "split_type": "chronological",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "target_classes_in_data": sorted(data[TARGET].unique().tolist()),
        "classes_absent_from_data": sorted(
            {"Normal", "Above Normal", "Severe", "Extreme"} - set(data[TARGET])
        ),
        "macro_f1": float(f1_score(test[TARGET], predictions, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": matrix,
        "classification_report": report,
        "feature_columns": FEATURES,
        "caveat": "Small, short-period pilot data from two matched stations; not an operational warning model.",
    }
    REPORT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Time split: train {len(train)} rows; test {len(test)} rows from {cutoff.date()} onward")
    print(f"Test macro F1: {summary['macro_f1']:.3f}")
    print(f"Classes not represented in data: {', '.join(summary['classes_absent_from_data']) or 'none'}")
    print(f"Saved evaluation: {REPORT_FILE}")
    print(classification_report(test[TARGET], predictions, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
