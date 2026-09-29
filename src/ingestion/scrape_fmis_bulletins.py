#!/usr/bin/env python3
"""Extract station context and Nepal rainfall from Bihar FMIS daily bulletins.

The official site exposes its current daily bulletin at a fixed PDF URL. Each
run saves a date-stamped copy and rebuilds the CSV from all saved copies, so
scheduled daily runs accumulate a local bulletin history.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber
import requests

BULLETIN_URL = "https://www.fmiscwrdbihar.gov.in/bulletin/Flood%20Bulletin.pdf"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BULLETIN_DIR = RAW_DIR / "fmis_bulletins"
CSV_FILE = RAW_DIR / "fmis_bulletin_extract.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/154.0.0.0 Safari/537.36"
)
COLUMNS = [
    "record_type",
    "report_date",
    "report_time",
    "basin",
    "district",
    "gauge_site",
    "observation_time",
    "high_flood_level_raw",
    "high_flood_level_m",
    "danger_level_m",
    "water_level_m",
    "trend",
    "status",
    "rain_gauge_site",
    "rainfall_raw",
    "rainfall_mm",
    "rainfall_source",
    "source_pdf",
]

# Basin labels are split across rows in the printed PDF. Station names provide
# a stable way to reconstruct the basin grouping without guessing from fragments.
BASIN_BY_STATION = {
    "chatia": "Gandak",
    "dumariaghat": "Gandak",
    "re waghat": "Gandak",
    "rewaghat": "Gandak",
    "hajipur": "Gandak",
    "lalbegiaghat": "Burhi Gandak",
    "ahirwalia": "Burhi Gandak",
    "sikandarpur": "Burhi Gandak",
    "samastipur": "Burhi Gandak",
    "rosera": "Burhi Gandak",
    "khagaria": "Burhi Gandak",
    "dhengbridge": "Bagmati / Adhwara",
    "sonakhan": "Bagmati / Adhwara",
    "dubbadhar": "Bagmati / Adhwara",
    "kansar": "Bagmati / Adhwara",
    "benibad": "Bagmati / Adhwara",
    "hayaghat": "Bagmati / Adhwara",
    "kamtaul": "Bagmati / Adhwara",
    "ekmighat": "Bagmati / Adhwara",
    "jainagar": "Kamla",
    "jhanjharpur": "Kamla",
    "basua": "Kosi",
    "baltara": "Kosi",
    "kursela": "Kosi",
    "dhengraghat": "Mahananda",
    "jhawa": "Mahananda",
}


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def parse_number(value: str) -> str:
    """Return the first numeric component, preserving missing/trace values blank."""
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return match.group(0) if match else ""


def bulletin_date(table: list[list[Any]]) -> str:
    candidates: list[str] = []
    for row in table[:5]:
        for cell in row:
            text = clean_cell(cell)
            if text:
                candidates.append(text)
    for text in candidates:
        match = re.search(r"\b(\d{1,2}-[A-Za-z]{3}-\d{2,4})\b", text)
        if match:
            raw = match.group(1)
            for fmt in ("%d-%b-%y", "%d-%b-%Y"):
                try:
                    return datetime.strptime(raw.title(), fmt).date().isoformat()
                except ValueError:
                    pass
    raise ValueError("Could not find the bulletin reporting date in the PDF table.")


def reporting_time(table: list[list[Any]]) -> str:
    for row in table[:4]:
        for cell in row:
            text = clean_cell(cell)
            match = re.search(r"REPORTING TIME AT\s+(\d{1,2}:\d{2})\s*([AP])\.?M", text, re.I)
            if match:
                return f"{match.group(1)} {match.group(2).upper()}M"
    return ""


def extract_rows(pdf_bytes: bytes, source_pdf: str) -> list[dict[str, str]]:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if not pdf.pages:
            raise ValueError("PDF has no pages.")
        tables = pdf.pages[0].extract_tables()
        if not tables:
            raise ValueError("No table found; this does not look like the FMIS daily bulletin.")
        table = max(tables, key=len)

    report_date = bulletin_date(table)
    report_time = reporting_time(table)
    rows: list[dict[str, str]] = []
    current_basin = ""

    # The first five rows are the bulletin title and multi-row column headers.
    for cells in table[5:]:
        cells = list(cells) + [None] * max(0, 10 - len(cells))
        district = clean_cell(cells[1])
        gauge_raw = clean_cell(cells[2])
        high_flood_raw = clean_cell(cells[3])
        danger_raw = clean_cell(cells[4])
        water_raw = clean_cell(cells[5])
        trend = clean_cell(cells[6])
        status = clean_cell(cells[7])
        rain_raw_site = clean_cell(cells[8])
        rainfall_raw = clean_cell(cells[9])

        gauge_site = gauge_raw.rstrip("^*").strip()
        if gauge_site:
            current_basin = BASIN_BY_STATION.get(gauge_site.casefold(), current_basin)
        rain_source_marked = "*" in rain_raw_site
        rain_gauge_site = clean_cell(rain_raw_site.replace("*", ""))

        if gauge_site:
            rows.append({
                "record_type": "station_status",
                "report_date": report_date,
                "report_time": report_time,
                "basin": current_basin,
                "district": district,
                "gauge_site": gauge_site,
                "observation_time": "14:00" if "^" in gauge_raw else ("06:00" if gauge_site else ""),
                "high_flood_level_raw": high_flood_raw,
                "high_flood_level_m": parse_number(high_flood_raw),
                "danger_level_m": parse_number(danger_raw),
                "water_level_m": parse_number(water_raw),
                "trend": trend,
                "status": status,
                "rain_gauge_site": "",
                "rainfall_raw": "",
                "rainfall_mm": "",
                "rainfall_source": "",
                "source_pdf": source_pdf,
            })
        if rain_gauge_site:
            rows.append({
                "record_type": "rainfall",
                "report_date": report_date,
                "report_time": report_time,
                "basin": current_basin,
                "district": "",
                "gauge_site": "",
                "observation_time": "",
                "high_flood_level_raw": "",
                "high_flood_level_m": "",
                "danger_level_m": "",
                "water_level_m": "",
                "trend": "",
                "status": "",
                "rain_gauge_site": rain_gauge_site,
                "rainfall_raw": rainfall_raw,
                "rainfall_mm": parse_number(rainfall_raw),
                "rainfall_source": "hydrology.gov.np" if rain_source_marked else "mfd.gov.np",
                "source_pdf": source_pdf,
            })
    if not rows:
        raise ValueError("The bulletin table contained no station or rainfall records.")
    return rows


def download_latest(session: requests.Session) -> Path:
    response = session.get(BULLETIN_URL, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("FMIS bulletin URL did not return a PDF document.")

    parsed = extract_rows(response.content, "")
    date_stamp = parsed[0]["report_date"].replace("-", "")
    pdf_path = BULLETIN_DIR / f"Flood_Bulletin_{date_stamp}.pdf"
    pdf_path.write_bytes(response.content)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip download and rebuild the CSV from PDFs already in data/raw/fmis_bulletins.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Also extract a local bulletin PDF; can be supplied more than once.",
    )
    args = parser.parse_args()

    BULLETIN_DIR.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        with requests.Session() as session:
            session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
            saved = download_latest(session)
        print(f"Downloaded current FMIS bulletin: {saved.name}")

    pdf_paths = sorted(BULLETIN_DIR.glob("Flood_Bulletin_*.pdf"))
    pdf_paths.extend(path.expanduser().resolve() for path in args.pdf)
    if not pdf_paths:
        raise SystemExit("No saved FMIS bulletins found. Run without --offline to download the latest.")

    extracted: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for pdf_path in pdf_paths:
        rows = extract_rows(pdf_path.read_bytes(), pdf_path.name)
        for row in rows:
            key = tuple(row[column] for column in COLUMNS[:-1])
            if key not in seen:
                seen.add(key)
                extracted.append(row)

    extracted.sort(
        key=lambda row: (
            row["report_date"], row["report_time"], row["basin"], row["record_type"],
            row["gauge_site"], row["rain_gauge_site"]
        )
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(extracted)

    print(f"Saved {len(extracted):,} FMIS bulletin rows to {CSV_FILE}")
    print(f"Bulletins covered: {len({row['report_date'] for row in extracted})}")
    print(f"Date range: {extracted[0]['report_date']} through {extracted[-1]['report_date']}")


if __name__ == "__main__":
    main()


