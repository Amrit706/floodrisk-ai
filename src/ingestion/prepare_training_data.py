#!/usr/bin/env python3
"""Join daily river levels and CHIRPS rain, engineer features, and label 24h risk."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "bihar.yaml"
RIVER_FILE = PROJECT_ROOT / "data" / "raw" / "river_level.csv"
RAINFALL_FILES = (
    PROJECT_ROOT / "data" / "raw" / "chirps" / "rainfall_2026.csv",
    PROJECT_ROOT / "data" / "raw" / "chirps" / "rainfall.csv",
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MAPPING_FILE = OUTPUT_DIR / "station_crosswalk.csv"
OUTPUT_FILE = OUTPUT_DIR / "flood_training_24h.csv"
RIVER_LABEL_FILE = OUTPUT_DIR / "river_level_labeled_24h.csv"
COVERAGE_FILE = OUTPUT_DIR / "data_coverage.csv"


def normalize_station(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def risk_label(level: float, warning: float | None, danger: float, hfl: float) -> str:
    if level >= hfl:
        return "Extreme"
    if level >= danger:
        return "Severe"
    if warning is not None and pd.notna(warning) and level >= warning:
        return "Above Normal"
    return "Normal"


def consecutive_next(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    previous_level = group["water_level_m"].shift(1)
    previous_date = group["date"].shift(1)
    next_level = group["water_level_m"].shift(-1)
    next_date = group["date"].shift(-1)

    group["water_level_change_24h_m"] = group["water_level_m"] - previous_level
    group.loc[group["date"] - previous_date != pd.Timedelta(days=1), "water_level_change_24h_m"] = pd.NA
    group["target_date"] = next_date
    group["target_water_level_m"] = next_level
    invalid_next = next_date - group["date"] != pd.Timedelta(days=1)
    group.loc[invalid_next, ["target_date", "target_water_level_m"]] = pd.NA
    return group


def main() -> None:
    if not CONFIG_FILE.exists() or not RIVER_FILE.exists():
        missing = [str(path) for path in (CONFIG_FILE, RIVER_FILE) if not path.exists()]
        raise FileNotFoundError("Required inputs are missing: " + ", ".join(missing))

    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    stations = config["stations"]
    levels = pd.read_csv(RIVER_FILE)

    levels["date"] = pd.to_datetime(levels["date"], errors="raise").dt.normalize()
    levels["water_level_m"] = pd.to_numeric(levels["water_level_m"], errors="coerce")
    levels["station_key"] = levels["station"].map(normalize_station)

    available_by_name: dict[str, list[pd.Series]] = {}
    for station_key, rows in levels.groupby("station_key", sort=False):
        available_by_name[station_key] = [rows.iloc[0]]

    mappings: list[dict] = []
    config_for_id: dict[str, dict] = {}
    river_key_to_config: dict[str, str] = {}
    for station in stations:
        station_key = normalize_station(station["name"])
        match = available_by_name.get(station_key, [])
        matched = bool(match)
        river_row = match[0] if matched else None
        mappings.append(
            {
                "config_station_id": station["station_id"],
                "config_station_name": station["name"],
                "river_station_id": river_row["station_id"] if matched else "",
                "river_station_name": river_row["station"] if matched else "",
                "match_status": "matched" if matched else "no river-level station match",
            }
        )
        if matched:
            config_for_id[station["station_id"]] = station
            river_key_to_config[station_key] = station["station_id"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mappings).to_csv(MAPPING_FILE, index=False)
    if not river_key_to_config:
        raise ValueError("No configured stations match river_level.csv station names.")
    levels = levels[levels["station_key"].isin(river_key_to_config)].copy()
    levels["station_id"] = levels["station_key"].map(river_key_to_config)
    levels["time_sort"] = levels["time"].astype(str).str.replace(":", "", regex=False)
    levels = levels.sort_values(["station_id", "date", "time_sort"])
    # Use each day's 14:00 observation where available; fall back to 06:00 on gaps.
    daily_levels = levels.drop_duplicates(["station_id", "date"], keep="last").copy()
    daily_levels["river_level_time"] = daily_levels["time"]

    metadata = pd.DataFrame(stations).set_index("station_id")
    for column in ("name", "river", "district", "latitude", "longitude", "warning_level_m", "danger_level_m", "highest_flood_level_m"):
        daily_levels[column] = daily_levels["station_id"].map(metadata[column])
    daily_levels = daily_levels.rename(columns={"name": "station_name", "highest_flood_level_m": "hfl_m"})
    daily_levels["risk_label"] = daily_levels.apply(
        lambda row: risk_label(row["water_level_m"], row["warning_level_m"], row["danger_level_m"], row["hfl_m"]),
        axis=1,
    )

    label_groups = []
    for _, group in daily_levels.groupby("station_id", sort=False):
        group = consecutive_next(group)
        group["risk_label_24h"] = group.apply(
            lambda row: risk_label(
                row["target_water_level_m"], row["warning_level_m"], row["danger_level_m"], row["hfl_m"]
            ) if pd.notna(row["target_water_level_m"]) else pd.NA,
            axis=1,
        )
        label_groups.append(group)
    river_labels = pd.concat(label_groups, ignore_index=True)
    # Keep only rows that have both the previous-day level-change feature and
    # an observed next-day target, so every row is usable for 24-hour labels.
    river_labels = river_labels.dropna(subset=["water_level_change_24h_m", "risk_label_24h"]).copy()
    river_labels["date"] = river_labels["date"].dt.strftime("%Y-%m-%d")
    river_labels["target_date"] = pd.to_datetime(river_labels["target_date"]).dt.strftime("%Y-%m-%d")
    river_label_columns = [
        "date", "target_date", "station_id", "station_name", "river", "district", "river_station_id",
        "river_level_time", "water_level_m", "water_level_change_24h_m", "warning_level_m",
        "danger_level_m", "hfl_m", "risk_label", "target_water_level_m", "risk_label_24h",
    ]
    river_labels["river_station_id"] = river_labels["station_key"].map(
        {key: rows[0]["station_id"] for key, rows in available_by_name.items() if key in river_key_to_config}
    )
    river_labels[river_label_columns].to_csv(RIVER_LABEL_FILE, index=False)

    rainfall_path = next((path for path in RAINFALL_FILES if path.exists()), None)
    rainfall = None
    if rainfall_path is not None:
        rainfall = pd.read_csv(rainfall_path)
        rainfall["date"] = pd.to_datetime(rainfall["date"], errors="raise").dt.normalize()
        rainfall["rainfall_mm"] = pd.to_numeric(rainfall["rainfall_mm"], errors="coerce")
        if "chirps_stage" not in rainfall:
            rainfall["chirps_stage"] = "not recorded"
        if "chirps_flavor" not in rainfall:
            rainfall["chirps_flavor"] = "not recorded"

    coverage_rows = []
    river_min, river_max = daily_levels["date"].min(), daily_levels["date"].max()
    for station_id, station in config_for_id.items():
        station_rain = rainfall[rainfall["station_id"] == station_id] if rainfall is not None else pd.DataFrame()
        overlap = station_rain[station_rain["date"].between(river_min, river_max)] if not station_rain.empty else station_rain
        coverage_rows.append({
            "station_id": station_id,
            "station_name": station["name"],
            "river_start": river_min.date().isoformat(),
            "river_end": river_max.date().isoformat(),
            "rainfall_file": rainfall_path.name if rainfall_path else "missing",
            "rainfall_start": station_rain["date"].min().date().isoformat() if not station_rain.empty else "",
            "rainfall_end": station_rain["date"].max().date().isoformat() if not station_rain.empty else "",
            "overlap_days": overlap["date"].nunique() if not overlap.empty else 0,
            "status": "matched dates" if not overlap.empty else "no rainfall/river date overlap",
        })
    pd.DataFrame(coverage_rows).to_csv(COVERAGE_FILE, index=False)
    if rainfall is None:
        print(f"River labels: {RIVER_LABEL_FILE}")
        print(f"Coverage: {COVERAGE_FILE}; no rainfall file was found, so merged training rows were not created.")
        return
    if not any(row["overlap_days"] for row in coverage_rows):
        print(f"River labels: {RIVER_LABEL_FILE}")
        print(f"Coverage: {COVERAGE_FILE}; {rainfall_path.name} does not overlap river dates {river_min.date()} to {river_max.date()}.")
        print("Merged training data was not created; obtain rainfall for the river observation period first.")
        return

    rainfall = rainfall[rainfall["station_id"].isin(config_for_id)].copy()
    rainfall = rainfall[["date", "station_id", "rainfall_mm", "chirps_stage", "chirps_flavor"]]
    daily = daily_levels.merge(rainfall, on=["date", "station_id"], how="inner", validate="one_to_one")

    for column in ("name", "river", "district", "latitude", "longitude", "warning_level_m", "danger_level_m", "highest_flood_level_m"):
        daily[column] = daily["station_id"].map(metadata[column])
    daily = daily.rename(columns={"name": "station_name", "highest_flood_level_m": "hfl_m"})
    daily["risk_label"] = daily.apply(
        lambda row: risk_label(
            row["water_level_m"],
            row["warning_level_m"],
            row["danger_level_m"],
            row["hfl_m"],
        ),
        axis=1,
    )

    daily = daily.sort_values(["station_id", "date"]).reset_index(drop=True)
    feature_groups = []
    for _, group in daily.groupby("station_id", sort=False):
        group = consecutive_next(group)
        for days in (1, 2, 3):
            group[f"rainfall_{days * 24}h_mm"] = group["rainfall_mm"].rolling(days, min_periods=days).sum()
        target_labels = group.apply(
            lambda row: risk_label(
                row["target_water_level_m"],
                row["warning_level_m"],
                row["danger_level_m"],
                row["hfl_m"],
            ) if pd.notna(row["target_water_level_m"]) else pd.NA,
            axis=1,
        )
        group["risk_label_24h"] = target_labels
        feature_groups.append(group)

    training = pd.concat(feature_groups, ignore_index=True)
    training = training.dropna(subset=[
        "rainfall_24h_mm",
        "rainfall_48h_mm",
        "rainfall_72h_mm",
        "water_level_change_24h_m",
        "risk_label_24h",
    ]).copy()
    training["date"] = training["date"].dt.strftime("%Y-%m-%d")
    training["target_date"] = pd.to_datetime(training["target_date"]).dt.strftime("%Y-%m-%d")

    columns = [
        "date", "target_date", "station_id", "station_name", "river", "district",
        "river_station_id", "river_level_time", "water_level_m", "water_level_change_24h_m",
        "rainfall_mm", "rainfall_24h_mm", "rainfall_48h_mm", "rainfall_72h_mm",
        "warning_level_m", "danger_level_m", "hfl_m", "risk_label", "target_water_level_m",
        "risk_label_24h", "chirps_stage", "chirps_flavor",
    ]
    training["river_station_id"] = training["station_key"].map(
        {key: rows[0]["station_id"] for key, rows in available_by_name.items() if key in river_key_to_config}
    )
    training[columns].to_csv(OUTPUT_FILE, index=False)

    unmatched = [row["config_station_name"] for row in mappings if row["match_status"] != "matched"]
    print(f"Matched configured stations: {len(config_for_id)} / {len(stations)}")
    print(f"Unmatched stations: {', '.join(unmatched) if unmatched else 'none'}")
    print(f"Training rows: {len(training):,}; dates: {training['date'].min()} through {training['date'].max()}")
    print("Target classes:")
    print(training["risk_label_24h"].value_counts().to_string())
    print(f"Mapping: {MAPPING_FILE}")
    print(f"Training data: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

