# 🌊 FloodRiskAI

### Citizen-facing, 24-hour river-risk research prototype for the Gandak pilot in Bihar

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/) [![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/) [![Random Forest](https://img.shields.io/badge/Selected%20model-Random%20Forest-228B22)](https://scikit-learn.org/stable/modules/ensemble.html#forest) [![XGBoost](https://img.shields.io/badge/Compared%20model-XGBoost-189A45)](https://xgboost.readthedocs.io/) [![Data](https://img.shields.io/badge/Rainfall-CHIRPS%20v3-2E8B57)](https://chc.ucsb.edu/data/chirps3) [![Data](https://img.shields.io/badge/River%20levels-Bihar%20WRD-1565C0)](https://wrd.bihar.gov.in/rivers/2026/waterLevel.php)

> FloodRiskAI estimates the river-risk category about 24 hours after a selected observation and explains the historical signals behind the result. It is a **historical replay prototype**, not a live warning service.

[![Repository](https://img.shields.io/badge/GitHub-Amrit706%2Ffloodrisk--ai-181717?logo=github)](https://github.com/Amrit706/floodrisk-ai)
[![Hackathon](https://img.shields.io/badge/Hackathon-HackNowa%202026-6c5ce7)](https://unstop.com/hackathons/hacknowa-global-hackathon-2026-innovation-hacks-1752198/amp)
[![Status](https://img.shields.io/badge/Status-Research%20prototype-orange)](#-limitations-and-responsible-use)

🚀 **Streamlit app:** [Open FloodRiskAI](https://floodrisk-ai.streamlit.app/) 

---

## 💡 The idea

Flood monitoring data is useful, but can be difficult for residents to interpret. FloodRiskAI explores a plain-language question:

> **“For this station and observation date, what risk category did the model estimate for the next day—and what historical factors help explain it?”**

The app lets a user select a supported station and historical observation, view the model’s next-day category and compare it with the threshold-derived historical target.

## ✨ What’s included

- 🗺️ Bihar station metadata and configured danger / highest flood level thresholds in `configs/bihar.yaml`.
- 🌧️ CHIRPS v3 daily rainfall samples and 🏞️ Bihar WRD dashboard river readings for **1 June–25 September 2026**.
- 🔗 A data alignment and feature-preparation pipeline for daily river change, non-overlapping rainfall windows, and next-day risk labels.
- 🧪 Numeric feature correlation and VIF diagnostics, four-fold chronological cross-validation, Random Forest tuning, and a PCA + Random Forest comparison.
- ⚡ A side-by-side XGBoost comparison using the same expanding-window, date-based folds; the search tests 32 configurations, including class reweighting and tree regularization.
- 🏆 Model selection based on cross-validation, with a later-date holdout evaluation. PCA + Random Forest is retained as a comparison candidate.
- 📱 A Streamlit app for historical replay, next-day category display, and comparison with the historical target.

## 📍 Data coverage

| Coverage item | Current pilot |
|---|---|
| Region | Bihar Gandak pilot: Chatia (Purba Champaran) and Dumariaghat (Gopalganj) |
| River | Gandak |
| Matched stations | Dumariaghat and Chatia |
| Prepared training data | 228 rows: 114 per station |
| River observations available through | 29 September 2026 |
| CHIRPS rainfall input available through | 25 September 2026 |
| Target categories present | Normal, Above Normal, Severe |
| Extreme examples | None in the supplied data |

**Stations not included:** Bagaha, Chanpatia, and Ahirwalia do not have matching river-station names in the supplied river-level export, so they are not part of this model. Kosi/Koshi stations are also outside this Gandak pilot. Do not interpret the app as providing predictions for those places.

## 🔬 How the pipeline works

1. **Collect and align:** Match Bihar WRD river-level observations with CHIRPS daily rainfall for each supported station/date.
2. **Prepare features:** Calculate daily river-level change and non-overlapping rainfall windows. Express river level relative to the configured danger level; remove the duplicate `rainfall_mm` input.
3. **Create the target:** Derive Normal, Above Normal, Severe, or Extreme labels from the station threshold configuration. Shift the target so features at time *t* correspond to the category at *t + 24 hours*.
4. **Validate chronologically:** Keep all stations on a date together, tune on four expanding-window folds, and evaluate on later dates.
5. **Compare candidates:** Evaluate tuned Random Forest, PCA + Random Forest, and class-weighted XGBoost against a persistence reference (tomorrow’s category equals today’s).
6. **Serve a replay:** Load the saved model in Streamlit and present the historical next-day estimate and context.

The target is threshold-derived, not an independently observed flood-impact label. Model scores therefore describe agreement with these configured categories, not verified flood outcomes.

## 📊 Model evaluation

The current run selected **tuned Random Forest** using the one-standard-error rule. XGBoost was included in the comparison, but the evidence does not establish a reliable winner.

| Candidate | 4-fold chronological CV accuracy | CV macro F1 | Later-date holdout accuracy | Holdout macro F1 |
|---|---:|---:|---:|---:|
| 🌲 Tuned Random Forest — selected | 87.5% ± 16.1 percentage points | 0.659 | 95.7% (44/46) | 0.936 |
| 📉 PCA + Random Forest | 88.4% ± 16.2 percentage points | 0.672 | 93.5% | 0.901 |
| ⚡ Tuned XGBoost | 91.1% ± 11.7 percentage points | 0.692 | 95.7% (44/46) | 0.936 |
| ↔️ Persistence reference | 95.5% | — | 93.5% | — |

Approximate cross-validation 95% intervals are wide: Random Forest **58.0–100%**, PCA + Random Forest **58.5–100%**, and XGBoost **69.6–100%**. The date-block bootstrap 95% interval on the later-date holdout was **89.1–100%** for Random Forest and XGBoost, and **87.0–97.8%** for PCA + Random Forest. See `models/evaluation.json` for fold scores, intervals, and selected parameters.

### 🧭 Reading these scores responsibly

- **95.7% holdout accuracy is 44 correct predictions out of 46 rows.** It is not evidence of 95.7% real-world warning accuracy.
- Random Forest and XGBoost tied on the holdout. XGBoost’s higher average CV score is uncertain with only four variable temporal folds; the intervals overlap substantially.
- The persistence reference scored higher than either learned candidate on average CV. Current evidence does not show that ML reliably beats the simple reference.
- There are no Extreme examples, so the model has neither learned nor been evaluated on that category.
- The later-date holdout has already been examined during development. Treat the scores as exploratory—not as a fresh, independent final validation.
- The remaining numeric VIF values are approximately 1.0–1.1. Low VIF suggests little linear multicollinearity in these numeric inputs; it does not establish predictive quality.
- Random Forest and XGBoost class scores are **not calibrated probabilities**.

## 🚀 Run locally

Use Python 3.11 or newer. From the repository root:

```bash
git clone https://github.com/Amrit706/floodrisk-ai.git
cd floodrisk-ai
python -m venv .venv
```

Activate the environment and install dependencies:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Prepare data, train/evaluate the model, and launch the app:

```bash
python src/ingestion/prepare_training_data.py
python src/models/train_model.py
streamlit run app.py
```

The prepared training CSV and trained model are included in the supplied project archive. Re-running preparation requires `data/raw/river_level.csv` and `data/raw/chirps/rainfall_2026.csv`. The CHIRPS downloader can retrieve the rainfall sample again; river data must be obtained from the Bihar WRD dashboard or supplied as the expected CSV. Downloaded source files are included in the project folder for reproducibility and review.

**Deployment dependency note:** the app imports `yaml`; ensure `PyYAML` is in `requirements.txt` (`PyYAML` provides `import yaml`) so Streamlit Community Cloud can install it.

## 🧰 Tools and data sources

- **App:** Streamlit
- **Language / data processing:** Python, pandas, NumPy
- **Machine learning:** scikit-learn Random Forest, PCA, XGBoost
- **Rainfall:** [CHIRPS v3](https://chc.ucsb.edu/data/chirps3), sampled at configured station coordinates. Dates through 31 August use the final satellite product; 1–25 September use the preliminary satellite product.
- **River level:** [Bihar Water Resources Department water-level dashboard](https://wrd.bihar.gov.in/rivers/2026/waterLevel.php); exported readings are stored in `data/raw/river_level.csv`.
- **Thresholds / station metadata:** `configs/bihar.yaml`; Chatia has no warning level in the supplied configuration.

## 🗂️ Project structure

```text
app.py                         Streamlit interface
configs/bihar.yaml             Bihar station metadata and thresholds
data/raw/                      Source rainfall and river-level files
src/ingestion/                 Data download and preparation scripts
src/models/train_model.py      Feature prep, tuning, comparison, evaluation
src/models/balanced_xgb.py     Class-weighted XGBoost estimator
models/                        Saved model and evaluation artifacts
requirements.txt               Python dependencies
```

Generated data and model files may be excluded from some Git versions. Check the repository contents and `.gitignore` if a required file is missing after cloning or deployment.

## ⚠️ Limitations and responsible use

- This is a **research prototype**, not an official forecast, emergency alert, or evacuation tool.
- It replays historical observations; it does not currently provide a verified live forecast.
- Coverage is limited to two matched Gandak stations in a short monsoon period.
- Bagaha, Chanpatia, Ahirwalia, and Kosi/Koshi stations are not represented in the supplied matched training data.
- Threshold-derived categories are not the same as verified flood impacts; only three categories occur in this dataset.
- Scores may change substantially for other years, stations, or flood conditions. More verified seasons, stations, and station-held-out evaluation are needed before operational claims.
- For safety decisions, follow official CWC, Bihar WRD, and local authority advisories.

## 🛣️ Next steps

- Add longer verified records and additional matched stations, including Bagaha or Kosi/Koshi only when usable historical data is available.
- Document missing values, source quality checks, and station matching.
- Evaluate on a season or station withheld from all model development.
- Add documented Extreme examples and check threshold labels with domain sources.
- Improve prediction calibration and live data handling before considering real-time use.

## 🤝 AI assistance and authorship

AI tools assisted with code editing and debugging. The project submitter should review the data sources, understand the pipeline and limitations, and be able to explain and maintain the submitted work.

---

<p align="center"><strong>🌧️ Clearer context from river and rainfall data—one careful prototype at a time.</strong></p>



