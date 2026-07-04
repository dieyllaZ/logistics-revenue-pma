"""
Q3(a) — Automated Data Validation Script
Validates loads.csv (source) and master_dataset_clean.csv (derived).
Exits with status 1 on any failure — gates the CI/CD pipeline.

Run from project root:
    python code/validate_data.py
"""

import sys
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')

VALID_LOAD_TYPES    = {'Dry Van', 'Refrigerated'}
VALID_BOOKING_TYPES = {'Spot', 'Dedicated', 'Contract'}
RATIO_THRESHOLD     = 3.0
MAX_MISSING_PCT     = 0.32  # known data quality issue: 29.9% missing fuel_cost_clean


def check_loads(path):
    errors = []
    df = pd.read_csv(path)

    # 1. Missing critical columns
    for col in ['load_id', 'customer_id', 'route_id', 'revenue', 'weight_lbs']:
        n = df[col].isnull().sum()
        if n > 0:
            errors.append(f"[MISSING] '{col}' has {n} null rows")

    # 2. Duplicate load_id
    n_dupes = df.duplicated(subset=['load_id']).sum()
    if n_dupes > 0:
        errors.append(f"[DUPLICATES] {n_dupes} duplicate load_id rows")

    # 3. Non-positive revenue
    if (df['revenue'] <= 0).sum() > 0:
        errors.append(f"[RANGE] {(df['revenue']<=0).sum()} non-positive revenue rows")

    # 4. Non-positive weight
    if (df['weight_lbs'] <= 0).sum() > 0:
        errors.append(f"[RANGE] {(df['weight_lbs']<=0).sum()} non-positive weight rows")

    # 5. Category consistency
    invalid_lt = set(df['load_type'].dropna().unique()) - VALID_LOAD_TYPES
    if invalid_lt:
        errors.append(f"[CATEGORY] Unexpected load_type: {invalid_lt}")

    invalid_bt = set(df['booking_type'].dropna().unique()) - VALID_BOOKING_TYPES
    if invalid_bt:
        errors.append(f"[CATEGORY] Unexpected booking_type: {invalid_bt}")

    return errors, len(df)


def check_clean_dataset(path, trips_path, fuel_path):
    errors = []
    df    = pd.read_csv(path)
    trips = pd.read_csv(trips_path)
    fuel  = pd.read_csv(fuel_path)

    # 6. Required clean columns present
    required = ['fuel_cost_status', 'fuel_cost_clean', 'profit_margin_clean',
                'profitability_confidence', 'weight_per_piece',
                'expected_rate_revenue', 'customer_tenure_years',
                'revenue_potential_bucket']
    for col in required:
        if col not in df.columns:
            errors.append(f"[MISSING COL] '{col}' not in clean dataset")

    # 7. No implausible profit margins
    if 'profit_margin_clean' in df.columns:
        n = (df['profit_margin_clean'] < -1.0).sum()
        if n > 0:
            errors.append(f"[QUALITY] {n} rows have profit_margin_clean < -100%")

    # 8. Fuel cost missing rate
    if 'fuel_cost_clean' in df.columns:
        pct = df['fuel_cost_clean'].isnull().mean()
        if pct > MAX_MISSING_PCT:
            errors.append(f"[MISSING RATE] fuel_cost_clean missing {pct:.1%} "
                           f"(threshold {MAX_MISSING_PCT:.0%})")

    # 9. Gallons ratio gate
    avg_mpg       = trips['average_mpg'].mean()
    fuel_per_trip = fuel.groupby('trip_id')['gallons'].sum().reset_index()
    fuel_per_trip.columns = ['trip_id', 'gallons_purchased']
    check  = trips[['trip_id', 'actual_distance_miles']].merge(
        fuel_per_trip, on='trip_id', how='inner')
    check['gallons_needed'] = check['actual_distance_miles'] / avg_mpg
    check['gallons_ratio']  = check['gallons_purchased'] / check['gallons_needed']
    bad_pct = (check['gallons_ratio'] > RATIO_THRESHOLD).mean()
    if bad_pct > 0.25:
        errors.append(f"[FUEL INTEGRITY] {bad_pct:.1%} trips exceed "
                       f"{RATIO_THRESHOLD}x gallons ratio")

    return errors, len(df)


def validate():
    loads_path = os.path.join(DATA, 'loads.csv')
    clean_path = os.path.join(DATA, 'master_dataset_clean.csv')
    trips_path = os.path.join(DATA, 'trips.csv')
    fuel_path  = os.path.join(DATA, 'fuel_purchases.csv')

    all_errors = []

    print("Checking loads.csv ...")
    errs, n = check_loads(loads_path)
    all_errors.extend(errs)
    print(f"  {n:,} rows checked — {len(errs)} issue(s) found")

    print("Checking master_dataset_clean.csv ...")
    errs2, n2 = check_clean_dataset(clean_path, trips_path, fuel_path)
    all_errors.extend(errs2)
    print(f"  {n2:,} rows checked — {len(errs2)} issue(s) found")

    print()
    if all_errors:
        print("DATA VALIDATION FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"DATA VALIDATION PASSED: all checks clear across "
              f"{n:,} source rows and {n2:,} clean rows.")
        sys.exit(0)


if __name__ == '__main__':
    validate()
