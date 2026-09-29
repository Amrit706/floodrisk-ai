#!/usr/bin/env python3
"""Download CHIRPS v3 daily rainfall overlapping FloodRiskAI river levels.

Uses one consistent satellite flavor: final CHIRPS through August 2026,
then the preliminary release through September 25, 2026. The existing 2024
rainfall.csv is left untouched.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile
import yaml
from dhis2eo.data.chc.chirps3.daily import url_for_day

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "bihar.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "chirps"
OUTPUT_FILE = OUTPUT_DIR / "rainfall_2026.csv"

START_DATE = "2026-06-01"
FINAL_END_DATE = "2026-08-31"
PRELIM_START_DATE = "2026-09-01"
PRELIM_END_DATE = "2026-09-25"
FLAVOR = "sat"


def load_stations() -> list[dict]:
    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config["stations"]


def dates_between(start: str, end: str):
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while current <= last:
        yield current
        current += timedelta(days=1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_stations()
    bbox = (
        min(item["longitude"] for item in stations) - 0.10,
        min(item["latitude"] for item in stations) - 0.10,
        max(item["longitude"] for item in stations) + 0.10,
        max(item["latitude"] for item in stations) + 0.10,
    )

    records: list[dict] = []
    for day in dates_between(START_DATE, PRELIM_END_DATE):
        day_text = day.isoformat()
        stage = "final" if day_text <= FINAL_END_DATE else "prelim"
        url = url_for_day(day, stage=stage, flavor=FLAVOR)
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        with MemoryFile(response.content) as memfile:
            with memfile.open() as raster:
                values = list(raster.sample([(s["longitude"], s["latitude"]) for s in stations], masked=True))
        for station, sample in zip(stations, values):
            amount = float(sample[0]) if sample.count() else float("nan")
            records.append({
                "date": day_text,
                "station_id": station["station_id"],
                "station_name": station["name"],
                "river": station["river"],
                "district": station["district"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "rainfall_mm": amount,
                "chirps_stage": stage,
                "chirps_flavor": FLAVOR,
            })
        print(f"Read {day_text} ({stage})")

    rainfall = pd.DataFrame(records).sort_values(["station_id", "date"])
    rainfall.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(rainfall):,} rows for {rainfall['station_id'].nunique()} configured stations.")
    print(f"Coverage: {rainfall['date'].min()} through {rainfall['date'].max()}.")
    print(f"Final/preliminary rows: {(rainfall['chirps_stage'] == 'final').sum():,} / {(rainfall['chirps_stage'] == 'prelim').sum():,}.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
