# FloodRiskAI

FloodRiskAI is a citizen-facing research prototype that estimates the river-risk category about 24 hours after a selected observation. The pilot covers two stations with matching river records: Dumariaghat on the Gandak and Chatia on the Gandak. It is a historical replay, not a live warning service.

## What is included

- Bihar station metadata and published danger/HFL thresholds in `configs/bihar.yaml`.
- Bihar WRD dashboard river readings and CHIRPS v3 daily rainfall samples for 1 June–25 September 2026.
- An alignment and feature-preparation pipeline for daily river change, non-overlapping rainfall windows, and next-day risk labels.
- Numeric-feature correlation/VIF diagnostics, four-fold chronological cross-validation, Random Forest hyperparameter tuning, and a PCA comparison.
- Side-by-side tuning of Random Forest and XGBoost using the same expanding-window date-based cross-validation folds. XGBoost searches 32 configurations, including no, partial, or full class reweighting and tree regularization.
- A final model selected from cross-validation and evaluated on later dates held out from model tuning; PCA + Random Forest remains as a reference candidate.
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

The model evaluation uses a chronological 80/20 date split. The included later-date holdout has 46 rows; the tuned forest scored 95.7% accuracy (44/46) and 0.936 macro F1. This is not a 99% result and should not be presented as proof of real-world flood prediction.

The training portion uses four expanding-window folds, with all stations on a date kept together. The expanded XGBoost search averaged 91.1% accuracy across the folds (11.7-point fold standard deviation; approximate 95% t interval 69.6%–100%). The tuned Random Forest averaged 87.5% (16.1-point standard deviation; approximate 95% t interval 58.0%–100%). These intervals are very wide because only four temporal folds are available. They overlap substantially, so the one-standard-error rule keeps the simpler Random Forest; this does not establish that it is more accurate than XGBoost. The current-category persistence reference averaged 95.5% across those folds, higher than either learned candidate.

On the 46-row later-date holdout, both Random Forest and XGBoost scored 95.7% (44/46), with macro F1 0.936. The approximate 95% moving date-block bootstrap interval was 89.1%–100%. This date period has already been examined during development, so treat this as exploratory comparison rather than a fresh final validation. More years and additional verified stations are needed to narrow uncertainty. The report in `models/evaluation.json` includes fold scores, approximate CV intervals, date-block holdout intervals, and each candidate's best parameters.

A simple reference that predicts tomorrow's category equals today's observed category scored 95.5% average cross-validation accuracy and 93.5% on the original holdout. Compare this baseline with every tuned model in `models/evaluation.json`; there is not enough data to conclude any model is reliably better. The project removes the duplicate `rainfall_mm` input, expresses river level relative to danger, and uses non-overlapping rainfall blocks. The remaining numeric VIF values are about 1.0-1.1. The target is defined from station thresholds, and only two stations from a short monsoon period are available. More years, additional verified stations, and event-based validation are needed before any operational claim. Random Forest and XGBoost class scores are not calibrated probabilities.

River observations currently end on 29 September 2026; CHIRPS input ends on 25 September. The app labels itself as a historical replay. Do not use it for evacuation or other emergency decisions; follow official CWC, Bihar WRD, and local authority alerts.

## Data sources

- Rainfall: [CHIRPS v3](https://chc.ucsb.edu/data/chirps3), daily satellite estimates sampled at configured station coordinates. Dates through 31 August use the final satellite product; 1–25 September use the preliminary satellite product.
- River levels: Bihar Water Resources Department [water-level dashboard](https://wrd.bihar.gov.in/rivers/2026/waterLevel.php), exported readings included in `data/raw/river_level.csv`.
- Station danger and highest flood levels: station metadata recorded in `configs/bihar.yaml`; Chatia has no warning level in the supplied config.

## AI assistance

AI tools assisted with code editing and debugging. The project submitter should review the data sources, understand the model and its limitations, and be able to explain and maintain the submitted work.
