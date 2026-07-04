"""
config.py — Central configuration for all scripts
All paths, constants and settings are defined here.
Import this at the top of every script.
"""
import os

# ── Project root ──────────────────────────────────────────────
# This file sits in the project root.
# All other paths are relative to it.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Folder paths ──────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, 'data')
CHARTS_DIR  = os.path.join(BASE_DIR, 'charts')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# ── File paths ────────────────────────────────────────────────
# Source data
LOADS_CSV         = os.path.join(DATA_DIR, 'loads.csv')
TRIPS_CSV         = os.path.join(DATA_DIR, 'trips.csv')
FUEL_CSV          = os.path.join(DATA_DIR, 'fuel_purchases.csv')
ROUTES_CSV        = os.path.join(DATA_DIR, 'routes.csv')
CUSTOMERS_CSV     = os.path.join(DATA_DIR, 'customers.csv')
TRUCKS_CSV        = os.path.join(DATA_DIR, 'trucks.csv')
TRAILERS_CSV      = os.path.join(DATA_DIR, 'trailers.csv')
DRIVERS_CSV       = os.path.join(DATA_DIR, 'drivers.csv')

# Derived datasets
MASTER_CSV        = os.path.join(DATA_DIR, 'master_dataset.csv')
MASTER_CLEAN_CSV  = os.path.join(DATA_DIR, 'master_dataset_clean.csv')
EXCEPTIONS_CSV    = os.path.join(DATA_DIR, 'fuel_cost_exceptions.csv')

# Model outputs
BASELINE_MODEL    = os.path.join(OUTPUTS_DIR, 'baseline_model.pkl')
IMPROVED_MODEL    = os.path.join(OUTPUTS_DIR, 'improved_model.pkl')
MODEL_RESULTS     = os.path.join(OUTPUTS_DIR, 'model_results.json')
TEST_PREDICTIONS  = os.path.join(OUTPUTS_DIR, 'test_predictions.csv')
DRIFT_SUMMARY     = os.path.join(OUTPUTS_DIR, 'drift_and_degradation_summary.json')
DRIFT_RESULTS     = os.path.join(OUTPUTS_DIR, 'drift_results.csv')

# ── Modelling constants ───────────────────────────────────────
TARGET = 'revenue'
TEST_SIZE = 0.25
RANDOM_STATE = 42

NUMERIC_BASE = [
    'weight_lbs', 'pieces', 'typical_distance_miles',
    'base_rate_per_mile', 'fuel_surcharge_rate',
    'typical_transit_days', 'credit_terms_days',
    'annual_revenue_potential'
]
CATEGORICAL_BASE = [
    'load_type', 'booking_type', 'customer_type',
    'primary_freight_type', 'account_status'
]
NUMERIC_ENG = [
    'weight_per_piece', 'expected_rate_revenue',
    'customer_tenure_years'
]
CATEGORICAL_ENG = ['revenue_potential_bucket']

NUMERIC_FULL     = NUMERIC_BASE + NUMERIC_ENG
CATEGORICAL_FULL = CATEGORICAL_BASE + CATEGORICAL_ENG

# ── Data quality constants ────────────────────────────────────
GALLONS_RATIO_THRESHOLD = 3.0
MAX_MISSING_PCT         = 0.32
PROFIT_MARGIN_THRESHOLD = -1.0  # -100%

# ── Create folders if they don't exist ───────────────────────
for folder in [DATA_DIR, CHARTS_DIR, OUTPUTS_DIR]:
    os.makedirs(folder, exist_ok=True)
