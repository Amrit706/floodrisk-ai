"""Streamlit historical replay for the FloodRiskAI pilot model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "flood_training_24h.csv"
MODEL_FILE = ROOT / "models" / "flood_model.pkl"
REPORT_FILE = ROOT / "models" / "evaluation.json"
CONFIG_FILE = ROOT / "configs" / "bihar.yaml"

FEATURES = [
    "station_id",
    "current_risk_label",
    "level_minus_danger_m",
    "water_level_change_24h_m",
    "rainfall_24h_mm",
    "rainfall_24_48h_mm",
    "rainfall_48_72h_mm",
]
RISK_STYLE = {
    "Normal": ("🟢", "#17803d"),
    "Above Normal": ("🟡", "#947000"),
    "Severe": ("🟠", "#bd5b00"),
    "Extreme": ("🔴", "#bd2222"),
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict, dict]:
    data = pd.read_csv(DATA_FILE, parse_dates=["date", "target_date"])
    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    return data, config, report


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def measured_context(row: pd.Series) -> list[str]:
    change = float(row["water_level_change_24h_m"])
    if change > 0.05:
        trend = f"River level rose {change:.2f} m over the previous 24 hours."
    elif change < -0.05:
        trend = f"River level fell {abs(change):.2f} m over the previous 24 hours."
    else:
        trend = "River level was nearly steady over the previous 24 hours."
    gap = float(row["danger_level_m"]) - float(row["water_level_m"])
    if gap >= 0:
        level = f"Observed level was {gap:.2f} m below the configured danger level."
    else:
        level = f"Observed level was {abs(gap):.2f} m above the configured danger level."
    rain = f"Rainfall summed to {float(row['rainfall_72h_mm']):.1f} mm over the previous 72 hours."
    return [trend, level, rain]


def shap_explanations(pipeline, frame: pd.DataFrame, class_name: str) -> list[str]:
    """Return local tree-SHAP explanations, translated to citizen-facing labels."""
    import numpy as np
    import shap

    transform = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    encoded = transform.transform(frame)
    shap_model = getattr(classifier, "estimator_", classifier)
    values = shap.TreeExplainer(shap_model).shap_values(encoded)
    class_index = list(classifier.classes_).index(class_name)
    if isinstance(values, list):
        contributions = np.asarray(values[class_index])[0]
    else:
        values = np.asarray(values)
        contributions = values[0, :, class_index] if values.ndim == 3 else values[0]
    names = transform.get_feature_names_out()
    friendly = {
        "numeric__level_minus_danger_m": "current river level relative to danger threshold",
        "numeric__water_level_change_24h_m": "24-hour river-level change",
        "numeric__rainfall_24h_mm": "24-hour rainfall",
        "numeric__rainfall_24_48h_mm": "rainfall from 24 to 48 hours earlier",
        "numeric__rainfall_48_72h_mm": "rainfall from 48 to 72 hours earlier",
    }
    ranked = np.argsort(np.abs(contributions))[::-1]
    result = []
    for index in ranked:
        name = str(names[index])
        if name.startswith("station__"):
            continue
        value = float(contributions[index])
        if abs(value) < 0.001:
            continue
        direction = "increased" if value > 0 else "reduced"
        label = friendly.get(name, name.replace("numeric__", "").replace("_", " "))
        result.append(f"{label} {direction} the {class_name} model score")
        if len(result) == 3:
            break
    return result


st.set_page_config(page_title="FloodRiskAI | Bihar", page_icon="🌊", layout="centered")
st.title("🌊 FloodRiskAI")
st.caption("A citizen-friendly, 24-hour flood-risk research demo for Bihar")
st.warning(
    "Historical replay only. The source river readings end on 29 Sep 2026; this app has no live feed "
    "and must not be used for emergency decisions. Follow official local alerts."
)

if not (DATA_FILE.exists() and MODEL_FILE.exists() and REPORT_FILE.exists()):
    st.error("Training data or model is missing. Follow the setup steps in README.md first.")
    st.stop()

data, config, report = load_data()
model = load_model()
stations = {s["station_id"]: s for s in config["stations"]}
available_ids = sorted(data["station_id"].unique(), key=lambda sid: stations[sid]["name"])

with st.sidebar:
    st.header("Choose a historical observation")
    station_id = st.selectbox(
        "Station",
        available_ids,
        format_func=lambda sid: f"{stations[sid]['name']} · {stations[sid]['river']}",
    )
    choices = data.loc[data["station_id"] == station_id].sort_values("date")
    selected_date = st.selectbox(
        "Observation date",
        choices["date"].tolist(),
        format_func=lambda d: d.strftime("%d %b %Y"),
    )

row = choices.loc[choices["date"] == selected_date].iloc[0]
features = pd.DataFrame([{
    "station_id": row["station_id"],
    "current_risk_label": row["risk_label"],
    "level_minus_danger_m": row["water_level_m"] - row["danger_level_m"],
    "water_level_change_24h_m": row["water_level_change_24h_m"],
    "rainfall_24h_mm": row["rainfall_24h_mm"],
    "rainfall_24_48h_mm": row["rainfall_48h_mm"] - row["rainfall_24h_mm"],
    "rainfall_48_72h_mm": row["rainfall_72h_mm"] - row["rainfall_48h_mm"],
}])
probabilities = model.predict_proba(features)[0]
classes = model.named_steps["classifier"].classes_
ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
predicted_class, score = ranked[0]
icon, _ = RISK_STYLE.get(predicted_class, ("⚪", "#444444"))

st.subheader(f"{stations[station_id]['name']}, {stations[station_id]['district']}")
st.write(f"**Prediction for:** {row['target_date']:%d %b %Y} (about 24 hours after the selected observation)")
st.markdown(f"## {icon} {predicted_class}")
st.metric("Model score for this class", f"{score:.0%}")
st.caption("This is the model's uncalibrated class probability, not a verified probability of flooding.")

if "Extreme" in report.get("classes_absent_from_data", []):
    st.info("No Extreme examples were present in training data, so the model cannot learn or predict that class reliably.")

st.markdown("### What the selected observations show")
for context in measured_context(row):
    st.write(f"• {context}")
try:
    local_reasons = shap_explanations(model, features, predicted_class)
except Exception:
    local_reasons = []
if local_reasons:
    st.markdown("### Model factors (SHAP)")
    for reason in local_reasons:
        st.write(f"• {reason}")
    st.caption("SHAP describes how inputs influenced this model output; it does not establish flood causation.")
else:
    st.caption("Measured context is shown above. SHAP explanations are unavailable in this environment.")

with st.expander("Compare with the recorded next-day category"):
    st.write(f"Recorded next-day category: **{row['risk_label_24h']}**")
    st.caption("Shown only because this is a historical replay. The recorded category is the training target.")

st.markdown("### Pilot model evaluation")
metric_cols = st.columns(3)
metric_cols[0].metric("Later-date holdout accuracy", f"{report['accuracy']:.1%}")
metric_cols[1].metric("Holdout macro F1", f"{report['macro_f1']:.3f}")
metric_cols[2].metric("Holdout rows", str(report["test_rows"]))
persistence_cv = report["cv"]["persistence_baseline_accuracy"]
persistence_test = report["persistence_baseline_holdout"]
comparison = report["cv"].get("candidate_comparison", {})
holdout_comparison = report.get("holdout_candidate_comparison", {})
if comparison:
    model_names = {
        "tuned_random_forest": "Random Forest",
        "pca_random_forest": "PCA + Random Forest",
        "tuned_xgboost": "Class-weighted XGBoost",
    }
    comparison_rows = []
    for key, values in comparison.items():
        holdout = holdout_comparison.get(key, {})
        comparison_rows.append({
            "Model": model_names.get(key, key),
            "CV accuracy": values["cv_accuracy_mean"],
            "CV 95% interval": "–".join(f"{bound:.1%}" for bound in values.get("cv_accuracy_95_ci", [])) or "not available",
            "CV fold SD": values["cv_accuracy_std"],
            "CV macro F1": values["cv_macro_f1_mean"],
            "Holdout accuracy": holdout.get("accuracy"),
            "Holdout 95% interval": "–".join(
                f"{bound:.1%}" for bound in holdout.get("accuracy_95_ci_date_block_bootstrap", [])
            ) or "not available",
            "Holdout macro F1": holdout.get("macro_f1"),
        })
    st.dataframe(
        pd.DataFrame(comparison_rows),
        column_config={
            "CV accuracy": st.column_config.NumberColumn(format="percent"),
            "CV fold SD": st.column_config.NumberColumn(format="percent"),
            "CV macro F1": st.column_config.NumberColumn(format="%.3f"),
            "Holdout accuracy": st.column_config.NumberColumn(format="percent"),
            "Holdout macro F1": st.column_config.NumberColumn(format="%.3f"),
        },
        hide_index=True,
    )
    st.write(
        f"Selected using chronological cross-validation: **{report.get('model', 'Random Forest')}**. "
        f"The current-category persistence reference averaged {persistence_cv['mean']:.1%} CV accuracy "
        f"and scored {persistence_test['accuracy']:.1%} on the later holdout."
    )
    st.caption(report.get("cv", {}).get("confidence_interval_note", "CV intervals are not available in this report."))
    st.caption(report.get("holdout_interval_note", "Holdout intervals are not available in this report."))
st.caption(
    f"Holdout dates: {report['test_start']} to {report['test_end']}. "
    "Only two stations and a short period are represented. The fold variation is large, "
    "so these scores do not establish forecast skill outside this pilot."
)

station_table = pd.DataFrame(
    [
        {"lat": stations[sid]["latitude"], "lon": stations[sid]["longitude"], "station": stations[sid]["name"]}
        for sid in available_ids
    ]
)
st.markdown("### Pilot locations")
st.map(station_table, latitude="lat", longitude="lon", size=80)
st.caption(
    "Rainfall source: CHIRPS v3 daily satellite estimates. River readings: Bihar WRD dashboard export supplied with the project."
)
