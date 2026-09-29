#!/usr/bin/env python3
"""Download Bihar WRD's dashboard and export its embedded station readings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import requests

DASHBOARD_URL = "https://wrd.bihar.gov.in/rivers/2026/waterLevel.php"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
HTML_FILE = OUTPUT_DIR / "waterLevel_debug.html"
CSV_FILE = OUTPUT_DIR / "river_level.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://wrd.bihar.gov.in/",
}

CSV_COLUMNS = [
    "date",
    "time",
    "station_id",
    "river_code",
    "river",
    "station_code",
    "station",
    "water_level_m",
    "danger_level_m",
    "highest_flood_level_m",
    "station_location_km",
]


def extract_api_response(html: str) -> dict[str, Any]:
    """Read the JSON object assigned to apiResponse in the dashboard source."""
    match = re.search(r"\bconst\s+apiResponse\s*=\s*", html)
    if not match:
        raise ValueError(
            "Dashboard HTML does not contain the expected 'const apiResponse' payload. "
            "The WRD page may have changed its format."
        )

    try:
        payload, _ = json.JSONDecoder().raw_decode(html[match.end():])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not decode dashboard apiResponse JSON: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("Dashboard apiResponse is missing its data object.")
    return payload


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload["data"]
    rivers = {row["riverCode"]: row["riverName"] for row in data.get("riverTable", [])}
    stations = {
        (row["riverCode"], row["siteCode"]): row
        for row in data.get("siteTable", [])
    }
    levels = data.get("levelTable", [])
    if not rivers or not stations or not levels:
        raise ValueError("Dashboard response has no rivers, stations, or level readings.")

    result: list[dict[str, Any]] = []
    for level in levels:
        river_code = str(level["riverCode"])
        station_code = str(level["siteCode"])
        station = stations.get((river_code, station_code))
        river_name = rivers.get(river_code)
        if station is None or river_name is None:
            raise ValueError(
                f"Reading references unknown station {river_code}/{station_code}."
            )

        result.append(
            {
                "date": level["readingDate"],
                "time": f"{int(level['readingTime']):02d}:00",
                "station_id": f"{river_code}-{station_code}",
                "river_code": river_code,
                "river": river_name,
                "station_code": station_code,
                "station": station["siteName"],
                "water_level_m": level["levelReading"],
                "danger_level_m": station.get("dl", ""),
                "highest_flood_level_m": station.get("hfl", ""),
                "station_location_km": station.get("siteLocation", ""),
            }
        )

    result.sort(key=lambda row: (row["date"], row["time"], row["station_id"]))
    return result


def download_dashboard() -> str:
    with requests.Session() as session:
        session.headers.update(HEADERS)
        response = session.get(DASHBOARD_URL, timeout=45)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        type=Path,
        help="Parse a previously saved dashboard HTML file instead of downloading it.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.html:
        html_path = args.html.expanduser().resolve()
        html = html_path.read_text(encoding="utf-8")
    else:
        html = download_dashboard()
        HTML_FILE.write_text(html, encoding="utf-8")
        html_path = HTML_FILE

    payload = extract_api_response(html)
    rows = rows_from_payload(payload)
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    dates = [row["date"] for row in rows]
    print(f"Parsed dashboard: {html_path}")
    print(f"Saved {len(rows):,} station readings to {CSV_FILE}")
    print(f"Coverage: {min(dates)} through {max(dates)}")
    print(f"Stations: {len({row['station_id'] for row in rows})}")


if __name__ == "__main__":
    main()
