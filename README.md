# FloodRiskAI

FloodRiskAI is a citizen-facing research prototype that estimates the river-risk category about 24 hours after a selected observation. The pilot covers two stations with matching river records: Dumariaghat on the Gandak and Chatia on the Gandak. It is a historical replay, not a live warning service.

## What is included

- Bihar station metadata and published danger/HFL thresholds in `configs/bihar.yaml`.
- Bihar WRD dashboard river readings and CHIRPS v3 daily rainfall samples for 1 June–25 September 2026.
- An alignment and feature-preparation pipeline for 24/48/72-hour rainfall totals, daily river-level change, and next-day risk labels.
- A class-weighted Random Forest with a chronological holdout report.
- A Streamlit app for replaying an observation, viewing the model's next-day category, and comparing it with the historical target.

The pilot data contains 228 training rows (114 per matched station). Its target classes are Normal, Above Normal, and Severe. No Extreme examples were present, so the model has not learned that category. Bagaha, Chanpatia, and Ahirwalia do not have matching river-station names in the supplied river-level export and are not included in the model.

## Run locally

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/ingestion/prepare_training_data.py
python src/models/train_model.py
streamlit run app.py
```

The prepared training CSV and the trained model are included in this archive. Re-running preparation requires `data/raw/river_level.csv` and `data/raw/chirps/rainfall_2026.csv`. The CHIRPS downloader can retrieve the rainfall sample again; river data must be obtained from the Bihar WRD dashboard or supplied as the expected CSV. The downloaded source files in this project folder are provided for reproducibility and review.

## Evaluation and limits

The model evaluation uses a chronological 80/20 date split, with later dates held out. In the included run the holdout has 46 rows and macro F1 is about 0.90. That score is only a small pilot check: it comes from two stations and one short monsoon period, and the future label is derived from station danger/HFL thresholds. It does not establish forecast skill in other districts, seasons, or flood events. Random Forest class scores are not calibrated probabilities.

River observations currently end on 29 September 2026; CHIRPS input ends on 25 September. The app labels itself as a historical replay. Do not use it for evacuation or other emergency decisions; follow official CWC, Bihar WRD, and local authority alerts.

## Data sources

- Rainfall: [CHIRPS v3](https://chc.ucsb.edu/data/chirps3), daily satellite estimates sampled at configured station coordinates. Dates through 31 August use the final satellite product; 1–25 September use the preliminary satellite product.
- River levels: Bihar Water Resources Department [water-level dashboard](https://wrd.bihar.gov.in/rivers/2026/waterLevel.php), exported readings included in `data/raw/river_level.csv`.
- Station danger and highest flood levels: station metadata recorded in `configs/bihar.yaml`; Chatia has no warning level in the supplied config.

## AI assistance

AI tools assisted with code editing and debugging. The project submitter should review the data sources, understand the model and its limitations, and be able to explain and maintain the submitted work.
