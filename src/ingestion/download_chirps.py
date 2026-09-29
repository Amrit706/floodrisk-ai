from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
import xarray as xr

from dhis2eo.data.chc import chirps3


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "configs" / "bihar.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "chirps"
OUTPUT_FILE = OUTPUT_DIR / "rainfall.csv"


# ============================================================
# DATE RANGE
# ============================================================

START_DATE = "2024-01"
END_DATE = "2024-12"


# ============================================================
# LOAD STATIONS
# ============================================================

def load_stations():

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config["stations"]


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stations = load_stations()

    # --------------------------------------------------------
    # Bounding box covering all 5 stations
    # --------------------------------------------------------

    lats = [
        station["latitude"]
        for station in stations
    ]

    lons = [
        station["longitude"]
        for station in stations
    ]

    min_lon = min(lons) - 0.10
    max_lon = max(lons) + 0.10
    min_lat = min(lats) - 0.10
    max_lat = max(lats) + 0.10

    bbox = [
        min_lon,
        min_lat,
        max_lon,
        max_lat
    ]

    print("=" * 60)
    print("FloodRiskAI - CHIRPS v3 Rainfall Downloader")
    print("=" * 60)

    print(f"\nPeriod : {START_DATE} → {END_DATE}")

    print("\nBounding box:")
    print(bbox)

    print("\nStations:")
    for station in stations:
        print(
            f"  {station['station_id']} "
            f"({station['latitude']}, "
            f"{station['longitude']})"
        )

    print("\nDownloading spatially clipped CHIRPS data...")
    print("This may take some time initially.\n")

    # --------------------------------------------------------
    # Download monthly NetCDF files
    # --------------------------------------------------------

    files = chirps3.daily.download(
        start=START_DATE,
        end=END_DATE,
        bbox=bbox,
        dirname=OUTPUT_DIR,
        prefix="chirps3_daily"
    )

    print("\nDownloaded files:")

    for file in files:
        print(f"  {file}")

    if not files:
        print("\nNo CHIRPS files were downloaded.")
        return

    # --------------------------------------------------------
    # Open all monthly files
    # --------------------------------------------------------

    print("\nOpening CHIRPS data...")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords"
    )

    print(ds)

    # --------------------------------------------------------
    # Extract rainfall for each station
    # --------------------------------------------------------

    records = []

    for station in stations:

        station_id = station["station_id"]

        lat = station["latitude"]
        lon = station["longitude"]

        print(
            f"\nExtracting rainfall for "
            f"{station_id}..."
        )

        # Find nearest CHIRPS grid cell
        station_data = ds["precip"].sel(
            x=lon,
            y=lat,
            method="nearest"
        )

        dates = pd.to_datetime(
            station_data["time"].values
        )

        rainfall_values = station_data.values

        for date, rainfall in zip(
            dates,
            rainfall_values
        ):

            records.append({

                "date": date.strftime(
                    "%Y-%m-%d"
                ),

                "station_id": station_id,

                "station_name": station["name"],

                "river": station["river"],

                "district": station["district"],

                "latitude": lat,

                "longitude": lon,

                "rainfall_mm": float(rainfall)

            })

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    rainfall = pd.DataFrame(records)

    rainfall["date"] = pd.to_datetime(
        rainfall["date"]
    )

    rainfall = rainfall.sort_values(
        [
            "station_id",
            "date"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    rainfall.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        f"Rows generated : "
        f"{len(rainfall):,}"
    )

    print(
        f"Stations       : "
        f"{rainfall['station_id'].nunique()}"
    )

    print(
        f"Dates          : "
        f"{rainfall['date'].min().date()} "
        f"→ "
        f"{rainfall['date'].max().date()}"
    )

    print(
        f"Missing values : "
        f"{rainfall['rainfall_mm'].isna().sum()}"
    )

    print("\nPreview:")
    print(rainfall.head(10))

    print("\nSaved to:")
    print(OUTPUT_FILE)

    # Close dataset
    ds.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()