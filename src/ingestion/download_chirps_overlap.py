#!/usr/bin/env python3
"""Download CHIRPS v3 daily rainfall overlapping FloodRiskAI river levels.

Uses one consistent satellite flavor: final CHIRPS through August 2026,
then the preliminary release through September 25, 2026. The existing 2024
rainfall.csv is left untouched.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import os
from pathlib import Path

import pandas as pd
import rasterio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "bihar.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "chirps"
OUTPUT_FILE = OUTPUT_DIR / "rainfall_2026.csv"

START_DATE = "2026-06-01"
FINAL_END_DATE = "2026-08-31"
PRELIM_START_DATE = "2026-09-01"
PRELIM_END_DATE = "2026-09-25"
FLAVOR = "sat"
MAX_WORKERS = 6
CHIRPS_DAILY_BASE_URL = os.environ.get(
    "CHIRPS_DAILY_BASE_URL",
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily",
).rstrip("/")


def url_for_day(day: date, stage: str, flavor: str) -> str:
    """Build the public CHIRPS v3 daily GeoTIFF URL for a date."""
    product = "prelim" if stage == "prelim" else flavor
    filename = f"chirps-v3.0.{product}.{day:%Y.%m.%d}.tif"
    return (
        f"{CHIRPS_DAILY_BASE_URL}/{stage}/{flavor}/{day.year}/{filename}"
    )


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


def read_day(day: date, stations: list[dict]) -> tuple[str, str, list[dict]]:
    day_text = day.isoformat()
    stage = "final" if day_text <= FINAL_END_DATE else "prelim"
    url = url_for_day(day, stage=stage, flavor=FLAVOR)
    raster_url = url if url.startswith("/vsicurl/") else f"/vsicurl/{url}"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(raster_url) as raster:
            if raster.crs is None or raster.crs.to_epsg() != 4326:
                raise ValueError(f"Unexpected CRS for CHIRPS raster {url}: {raster.crs}")
            values = list(
                raster.sample(
                    [(s["longitude"], s["latitude"]) for s in stations],
                    masked=True,
                )
            )

    rows = []
    for station, sample in zip(stations, values):
        amount = float(sample[0]) if sample.count() else float("nan")
        rows.append({
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
    return day_text, stage, rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_stations()

    records: list[dict] = []
    days = list(dates_between(START_DATE, PRELIM_END_DATE))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_day, day, stations): day for day in days}
        for future in as_completed(futures):
            day_text, stage, day_rows = future.result()
            records.extend(day_rows)
            print(f"Read {day_text} ({stage}); {len(records) // len(stations)}/{len(days)} days complete")

    rainfall = pd.DataFrame(records).sort_values(["station_id", "date"])
    rainfall.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(rainfall):,} rows for {rainfall['station_id'].nunique()} configured stations.")
    print(f"Coverage: {rainfall['date'].min()} through {rainfall['date'].max()}.")
    print(f"Final/preliminary rows: {(rainfall['chirps_stage'] == 'final').sum():,} / {(rainfall['chirps_stage'] == 'prelim').sum():,}.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
