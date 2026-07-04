"""
Q1 — Cleaning Pipeline
Run: python 04_clean_dataset.py
Produces: data/master_dataset_clean.csv
"""
import pandas as pd
import numpy as np
from config import (DATA_DIR, MASTER_CSV, MASTER_CLEAN_CSV,
                    EXCEPTIONS_CSV, TRIPS_CSV, FUEL_CSV,
                    GALLONS_RATIO_THRESHOLD)

print("="*65)
print("CLEANING PIPELINE — master_dataset_clean.csv")
print("="*65)

# ── Load ──────────────────────────────────────────────────────
df    = pd.read_csv(MASTER_CSV)
trips = pd.read_csv(TRIPS_CSV)
fuel  = pd.read_csv(FUEL_CSV)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Step 1: Gallons ratio check ───────────────────────────────
print("\nStep 1 — Gallons ratio check (fuel integrity)...")
avg_mpg = trips['average_mpg'].mean()

fuel_per_trip = fuel.groupby('trip_id').agg(
    gallons_purchased=('gallons', 'sum'),
    fuel_cost_total=('total_cost', 'sum'),
    fuel_stops=('gallons', 'count')
).reset_index()

trips_check = trips[['trip_id', 'load_id', 'actual_distance_miles']].merge(
    fuel_per_trip, on='trip_id', how='left'
)
trips_check['gallons_needed'] = trips_check['actual_distance_miles'] / avg_mpg
trips_check['gallons_ratio']  = trips_check['gallons_purchased'] / trips_check['gallons_needed']

valid_ids      = set(trips_check.loc[trips_check['gallons_ratio'] <= GALLONS_RATIO_THRESHOLD, 'load_id'])
misassigned_ids = set(trips_check.loc[trips_check['gallons_ratio'] >  GALLONS_RATIO_THRESHOLD, 'load_id'])

# ── Step 2: Fuel cost status ──────────────────────────────────
print("Step 2 — Assigning fuel_cost_status...")
df['fuel_cost_status'] = 'Missing'
df.loc[df['load_id'].isin(valid_ids),        'fuel_cost_status'] = 'Valid'
df.loc[df['load_id'].isin(misassigned_ids),  'fuel_cost_status'] = 'Misassigned'

print(df['fuel_cost_status'].value_counts().to_string())

# ── Step 3: Clean fuel cost ───────────────────────────────────
print("\nStep 3 — Creating fuel_cost_clean...")
df['fuel_cost_clean'] = np.where(
    df['fuel_cost_status'] == 'Valid',
    df['fuel_cost'],
    np.nan
)

# ── Step 4: Clean profit metrics ─────────────────────────────
print("Step 4 — Recalculating profit metrics...")
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

implausible_before = (df['profit_margin_pct'] < -1.0).sum()
implausible_after  = (df['profit_margin_clean'].dropna() < -1.0).sum()
print(f"  Implausible before: {implausible_before:,}")
print(f"  Implausible after : {implausible_after:,}")

# ── Step 5: Profitability confidence flag ─────────────────────
print("Step 5 — Adding profitability_confidence flag...")
def confidence(row):
    if row['fuel_cost_status'] == 'Misassigned':
        return 'Unreliable — misassigned fuel records'
    elif row['fuel_cost_status'] == 'Missing':
        return 'Unknown — no fuel data'
    elif pd.notna(row['profit_margin_clean']) and row['profit_margin_clean'] < -0.20:
        return 'Low margin — monitor'
    else:
        return 'Reliable'

df['profitability_confidence'] = df.apply(confidence, axis=1)
print(df['profitability_confidence'].value_counts().to_string())

# ── Step 6: Engineered features ───────────────────────────────
print("\nStep 6 — Adding engineered features...")
df['weight_per_piece'] = df['weight_lbs'] / df['pieces'].replace(0, np.nan)
df['expected_rate_revenue'] = df['typical_distance_miles'] * df['base_rate_per_mile']
df['customer_tenure_years'] = (
    pd.to_datetime('2025-01-01')
    - pd.to_datetime(df['contract_start_date'], errors='coerce')
).dt.days / 365.25
df['revenue_potential_bucket'] = pd.cut(
    df['annual_revenue_potential'],
    bins=[0, 200000, 500000, 1000000, np.inf],
    labels=['<200K', '200K-500K', '500K-1M', '1M+']
).astype(str)

# ── Step 7: Save outputs ──────────────────────────────────────
df.to_csv(MASTER_CLEAN_CSV, index=False)
print(f"\nSaved: {MASTER_CLEAN_CSV}")
print(f"       {df.shape[0]:,} rows × {df.shape[1]} columns")

exception_log = trips_check[trips_check['gallons_ratio'] > GALLONS_RATIO_THRESHOLD][[
    'trip_id', 'load_id', 'actual_distance_miles',
    'gallons_needed', 'gallons_purchased', 'gallons_ratio', 'fuel_cost_total'
]].sort_values('gallons_ratio', ascending=False)
exception_log.to_csv(EXCEPTIONS_CSV, index=False)
print(f"Saved: {EXCEPTIONS_CSV} ({len(exception_log):,} rows)")

print("\nCleaning pipeline COMPLETE.")
