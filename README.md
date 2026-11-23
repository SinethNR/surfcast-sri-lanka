# 🏄 SurfCast Sri Lanka: Ensemble Learning Dashboard

Predicting **tomorrow's good surf days** at **Arugam Bay** (east coast) and **Hikkaduwa** (south-west coast),
Sri Lanka, with Bagging, Boosting, Stacking and hybrid models. Built for the Machine Learning hands-on mini project
(Ensemble Learning).

**Live dashboard:** https://surfcast-sri-lanka.streamlit.app

> 🤖 **AI-use declaration:** parts of the code, dashboard and documentation were generated with **Claude (Anthropic)**.
> The student ran all the code, checked the outputs and understands the methods. Student: **Dheeraka**.

## Why this problem
Surf tourism is a key income source for Sri Lankan coastal towns. The two coasts have **opposite surf seasons**
because the monsoon winds swap sides of the island: offshore winds at Arugam Bay in the south-west monsoon (≈ May–Sep),
and at Hikkaduwa in the north-east monsoon (≈ Nov–Mar). A one-day-ahead "good surf" forecast helps surf schools,
guesthouses and tourists plan.

## Data (Sri Lankan, free, no login)
| Source | Variables |
|---|---|
| [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api) | daily max wave / swell / wind-wave height, period, dominant direction |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | wind speed/gusts/direction, rain, humidity, pressure, cloud |

Two spots, daily. The script requests **2010-01-01 → 2025-12-31**, but Open-Meteo swell data only exists from **2021-10-02**, so about 4 years are used (train Oct 2021 to 2024, test 2025). `download_data.py` downloads and caches everything into `data/raw/`.

**Label: good surf day.** Swell 0.8–2.5 m, swell period ≥ 9 s, swell from the break's swell window, and wind ≤ 20 km/h
(or ≤ 35 km/h if offshore). **Target:** the label for *tomorrow*, predicted from today's information only.

## Methods
* **Time-based split:** train on the older years, test on the **most recent full year, 2025** (no random split, no leakage).
* **Features:** today's wave/weather values, sin/cos of month, day-of-year and all directions, offshore-wind
  component, swell alignment, wave-energy proxy, lags, rolling means, and today's label (persistence).
* **Models:** persistence baseline, Decision Tree, **Bagging** (Bagged Trees, Random Forest), **Boosting** (AdaBoost,
  Gradient Boosting, XGBoost, LightGBM), **Stacking** (RF + XGB + LGBM + LogReg → LogReg, blocked K-fold) and
  **Soft Voting**.
* **Analysis:** bias-variance (train/test gap, tree-depth curve), ROC and confusion matrices, OOB error vs trees,
  error vs `max_features`, RF Gini vs XGBoost gain importance, threshold trade-off, RandomizedSearchCV with
  `TimeSeriesSplit`, and stacking with blocked vs shuffled CV.
* **Hybrids:** **RF + SARIMA** (monthly swell; RF corrects SARIMA residuals) and **RF + MLP** (daily swell-height
  regression; blend weight chosen on a validation year).

## Results
Test set = 2025 (728 spot-days), threshold 0.5. All values come from `results/metrics.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Persistence (baseline) | 0.810 | 0.667 | 0.667 | 0.667 | 0.767 |
| Decision Tree | 0.769 | 0.587 | 0.638 | 0.611 | 0.730 |
| Random Forest (best bagging, best overall F1) | 0.845 | 0.767 | 0.652 | 0.705 | 0.883 |
| Gradient Boosting (best boosting) | 0.835 | 0.723 | 0.681 | 0.701 | 0.874 |
| Stacking | 0.838 | 0.746 | 0.652 | 0.696 | 0.885 |

Hybrid RF + MLP (daily swell height): RMSE 0.1675 m vs 0.1692 m for RF alone. RF + SARIMA did not beat plain SARIMA
with only about 4 years of monthly data.

## Project structure
```
download_data.py        reproducible data download (retries + caching)
features.py             shared feature engineering (notebook + dashboard)
SurfCast_SriLanka.ipynb full analysis: EDA → features → models → interpretation → save models
app.py                  Streamlit dashboard (7 tabs)
models/*.pkl            saved models (joblib, compress=3)
results/*.csv           metrics / predictions read by the dashboard
requirements.txt
GUIDE_SINHALA.md        simple Sinhala explanation + 3-minute video script
```

## How to run (Windows PowerShell)
```powershell
cd SurfCast_SL
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python download_data.py               # about 4-5 min (waits between calls for the free API limit)
$env:PYTHONIOENCODING="utf-8"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 SurfCast_SriLanka.ipynb
streamlit run app.py                  # opens http://localhost:8501
```
macOS/Linux: use `source .venv/bin/activate` and `export PYTHONIOENCODING=utf-8`.

## Dashboard tabs
**Overview** (KPIs, map, seasonal see-saw) · **Data explorer** · **Model comparison** (live threshold slider,
confusion matrix, ROC, bias-variance) · **Inside Random Forest** (OOB, *m*, importance, tuning) · **Predict** (pick a
real test day, what-if sliders, every model's probability, actual outcome) · **Hybrid models** · **About**.

## Hosting (bonus)
Push to GitHub (models are < 100 MB each), then deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud)
with `app.py` as the entry point.

## Limitations
Open-Meteo values are gridded model/reanalysis data, not buoy measurements at the exact break. The "good surf" label
uses simple rules, while real surf quality also depends on tide, sandbank and local wind effects.

## Credits
Weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0).
