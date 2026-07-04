"""
Q3(a) — Automated Data Validation Script
Run: python validate_data.py
Exits with code 1 on any failure — gates CI/CD pipeline.
"""
import sys
import pandas as pd
import numpy as np
from config import (LOADS_CSV, MASTER_CLEAN_CSV, TRIPS_CSV, FUEL_CSV,
                    GALLONS_RATIO_THRESHOLD, MAX_MISSING_PCT)

VALID_LOAD_TYPES    = {'Dry Van', 'Refrigerated'}
VALID_BOOKING_TYPES = {'Spot', 'Dedicated', 'Contract'}


def check_loads(path):
    errors = []
    df     = pd.read_csv(path)
    for col in ['load_id', 'customer_id', 'route_id', 'revenue', 'weight_lbs']:
        n = df[col].isnull().sum()
        if n > 0:
            errors.append(f"[MISSING] '{col}' has {n} null rows")
    if df.duplicated(subset=['load_id']).sum() > 0:
        errors.append(f"[DUPLICATES] {df.duplicated(subset=['load_id']).sum()} duplicate load_id")
    if (df['revenue'] <= 0).sum() > 0:
        errors.append(f"[RANGE] {(df['revenue']<=0).sum()} non-positive revenue rows")
    if (df['weight_lbs'] <= 0).sum() > 0:
        errors.append(f"[RANGE] {(df['weight_lbs']<=0).sum()} non-positive weight rows")
    invalid_lt = set(df['load_type'].dropna().unique()) - VALID_LOAD_TYPES
    if invalid_lt:
        errors.append(f"[CATEGORY] Unexpected load_type: {invalid_lt}")
    invalid_bt = set(df['booking_type'].dropna().unique()) - VALID_BOOKING_TYPES
    if invalid_bt:
        errors.append(f"[CATEGORY] Unexpected booking_type: {invalid_bt}")
    return errors, len(df)


def check_clean(path, trips_path, fuel_path):
    errors = []
    df     = pd.read_csv(path)
    trips  = pd.read_csv(trips_path)
    fuel   = pd.read_csv(fuel_path)

    required = ['fuel_cost_status', 'fuel_cost_clean', 'profit_margin_clean',
                'profitability_confidence', 'weight_per_piece',
                'expected_rate_revenue', 'customer_tenure_years',
                'revenue_potential_bucket']
    for col in required:
        if col not in df.columns:
            errors.append(f"[MISSING COL] '{col}' not found in clean dataset")

    if 'profit_margin_clean' in df.columns:
        n_impl = (df['profit_margin_clean'] < -1.0).sum()
        if n_impl > 0:
            errors.append(f"[QUALITY] {n_impl} rows have profit_margin_clean < -100%")

    if 'fuel_cost_clean' in df.columns:
        miss_pct = df['fuel_cost_clean'].isnull().mean()
        if miss_pct > MAX_MISSING_PCT:
            errors.append(f"[MISSING RATE] fuel_cost_clean missing {miss_pct:.1%} "
                           f"(threshold {MAX_MISSING_PCT:.0%})")

    avg_mpg = trips['average_mpg'].mean()
    fp      = fuel.groupby('trip_id')['gallons'].sum().reset_index()
    fp.columns = ['trip_id', 'gallons_purchased']
    chk = trips[['trip_id', 'actual_distance_miles']].merge(fp, on='trip_id', how='inner')
    chk['gallons_needed'] = chk['actual_distance_miles'] / avg_mpg
    chk['gallons_ratio']  = chk['gallons_purchased'] / chk['gallons_needed']
    bad_pct = (chk['gallons_ratio'] > GALLONS_RATIO_THRESHOLD).mean()
    if bad_pct > 0.25:
        errors.append(f"[FUEL INTEGRITY] {bad_pct:.1%} trips exceed "
                       f"{GALLONS_RATIO_THRESHOLD}x gallons ratio")

    return errors, len(df)


def validate():
    all_errors = []

    print("Checking loads.csv ...")
    errs, n = check_loads(LOADS_CSV)
    all_errors.extend(errs)
    print(f"  {n:,} rows — {len(errs)} issue(s)")

    print("Checking master_dataset_clean.csv ...")
    errs2, n2 = check_clean(MASTER_CLEAN_CSV, TRIPS_CSV, FUEL_CSV)
    all_errors.extend(errs2)
    print(f"  {n2:,} rows — {len(errs2)} issue(s)")

    print()
    if all_errors:
        print("DATA VALIDATION FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"DATA VALIDATION PASSED: all checks clear "
              f"({n:,} source rows, {n2:,} clean rows).")
        sys.exit(0)


if __name__ == '__main__':
    validate()
