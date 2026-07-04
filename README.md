# PMA MRTB 2173 — Logistics Revenue & Route Profitability
## Agile Data Science MVP — PyCharm / Local Setup

## Project Structure
logistics-revenue-pma/
  config.py                 <- Central settings (all paths defined here)
  requirements.txt          <- Install dependencies
  conftest.py               <- pytest config
  app.py                    <- Q4/Q5: Streamlit dashboard
  01_eda_quality.py         <- Q1: EDA and data quality checks
  04_clean_dataset.py       <- Q1: Cleaning pipeline
  02_modeling.py            <- Q2: Baseline + improved model
  03_drift_monitoring.py    <- Q5: Drift and performance analysis
  validate_data.py          <- Q3(a): Automated validation
  tests/test_validate.py    <- Q3(c): 11 pytest tests
  data/                     <- Place all 16 CSVs here
  charts/                   <- Auto-created, charts saved here
  outputs/                  <- Auto-created, models saved here
  .github/workflows/ci.yml  <- Q3(c): GitHub Actions CI

## Setup in PyCharm
1. File -> Settings -> Project -> Python Interpreter -> Add -> Virtualenv -> New -> OK
2. In Terminal: pip install -r requirements.txt
3. Place all 16 CSVs into data/ folder

## Run Order
  python 01_eda_quality.py
  python 04_clean_dataset.py
  python 02_modeling.py
  python 03_drift_monitoring.py
  python validate_data.py
  pytest tests/test_validate.py -v
  streamlit run app.py

## Key Results
  Sprint 2  Linear Regression  MAE $339.85  R2 0.9449
  Sprint 3  Random Forest      MAE $231.32  R2 0.9725
  Improvement: MAE -31.9%  R2 +0.0276
