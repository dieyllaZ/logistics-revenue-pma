"""
Q1(c)(d) — Dataset Cleaning Pipeline
Produces master_dataset_clean.csv

Run from project root:
    python code/04_clean_dataset.py
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT  = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
df    = pd.read_csv(os.path.join(DATA, 'master_dataset.csv'))
trips = pd.read_csv(os.path.join(DATA, 'trips.csv'))
fuel  = pd.read_csv(os.path.join(DATA, 'fuel_purchases.csv'))
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Gallons ratio check ───────────────────────────────────────
print("\nRunning gallons ratio check...")
avg_mpg = trips['average_mpg'].mean()

fuel_per_trip = fuel.groupby('trip_id').agg(
    gallons_purchased=('gallons',     'sum'),
    fuel_cost_total  =('total_cost',  'sum'),
    fuel_stops       =('gallons',     'count')
).reset_index()

trips_check = trips[['trip_id', 'load_id', 'actual_distance_miles']].merge(
    fuel_per_trip, on='trip_id', how='left'
)
trips_check['gallons_needed'] = trips_check['actual_distance_miles'] / avg_mpg
trips_check['gallons_ratio']  = (
    trips_check['gallons_purchased'] / trips_check['gallons_needed']
)

RATIO_THRESHOLD = 3.0
valid_ids       = set(trips_check.loc[trips_check['gallons_ratio'] <= RATIO_THRESHOLD, 'load_id'])
misassigned_ids = set(trips_check.loc[trips_check['gallons_ratio'] >  RATIO_THRESHOLD, 'load_id'])

# ── Fuel cost status ──────────────────────────────────────────
df['fuel_cost_status'] = 'Missing'
df.loc[df['load_id'].isin(valid_ids),       'fuel_cost_status'] = 'Valid'
df.loc[df['load_id'].isin(misassigned_ids), 'fuel_cost_status'] = 'Misassigned'

print(f"\nFuel cost status:")
print(df['fuel_cost_status'].value_counts())

# ── Clean fuel cost ───────────────────────────────────────────
df['fuel_cost_clean'] = np.where(
    df['fuel_cost_status'] == 'Valid',
    df['fuel_cost'],
    np.nan
)

# ── Recalculate profit metrics ────────────────────────────────
df['total_revenue_with_surcharge'] = df['revenue'] + df['fuel_surcharge']

df['estimated_profit_clean'] = np.where(
    df['fuel_cost_clean'].notna(),
    df['total_revenue_with_surcharge'] - df['fuel_cost_clean'],
    np.nan
)
df['profit_margin_clean'] = np.where(
    df['fuel_cost_clean'].notna(),
    df['estimated_profit_clean'] / df['total_revenue_with_surcharge'],
    np.nan
)

impl_before = (df['profit_margin_pct'] < -1.0).sum()
impl_after  = (df['profit_margin_clean'] < -1.0).sum()
print(f"\nImplausible rows before: {impl_before:,}")
print(f"Implausible rows after : {impl_after:,}")

# ── Profitability confidence flag ─────────────────────────────
def assign_confidence(row):
    if row['fuel_cost_status'] == 'Misassigned':
        return 'Unreliable — misassigned fuel records'
    elif row['fuel_cost_status'] == 'Missing':
        return 'Unknown — no fuel data'
    elif row['profit_margin_clean'] < -0.20:
        return 'Low margin — monitor'
    else:
        return 'Reliable'

df['profitability_confidence'] = df.apply(assign_confidence, axis=1)
print(f"\nProfitability confidence:")
print(df['profitability_confidence'].value_counts())

# ── Engineered features ───────────────────────────────────────
df['weight_per_piece'] = (
    df['weight_lbs'] / df['pieces'].replace(0, np.nan)
)
df['expected_rate_revenue'] = (
    df['typical_distance_miles'] * df['base_rate_per_mile']
)
df['customer_tenure_years'] = (
    pd.to_datetime('2025-01-01')
    - pd.to_datetime(df['contract_start_date'], errors='coerce')
).dt.days / 365.25

df['revenue_potential_bucket'] = pd.cut(
    df['annual_revenue_potential'],
    bins=[0, 200000, 500000, 1000000, np.inf],
    labels=['<200K', '200K-500K', '500K-1M', '1M+']
).astype(str)

# ── Save ──────────────────────────────────────────────────────
clean_path      = os.path.join(DATA, 'master_dataset_clean.csv')
exception_path  = os.path.join(DATA, 'fuel_cost_exceptions.csv')

df.to_csv(clean_path, index=False)
print(f"\nSaved: {clean_path}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

exception_log = trips_check[trips_check['gallons_ratio'] > RATIO_THRESHOLD][[
    'trip_id', 'load_id', 'actual_distance_miles',
    'gallons_needed', 'gallons_purchased', 'gallons_ratio', 'fuel_cost_total'
]].sort_values('gallons_ratio', ascending=False)
exception_log.to_csv(exception_path, index=False)
print(f"Saved: {exception_path} ({len(exception_log):,} rows)")

print("\nCleaning complete.")
