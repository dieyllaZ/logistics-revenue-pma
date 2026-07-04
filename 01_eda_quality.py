"""
Q1 — EDA and Data Quality Checks
Run: python 01_eda_quality.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import (DATA_DIR, CHARTS_DIR, OUTPUTS_DIR,
                    LOADS_CSV, ROUTES_CSV, CUSTOMERS_CSV,
                    TRIPS_CSV, FUEL_CSV, MASTER_CSV)

sns.set_style('whitegrid')

# ── Load source tables ────────────────────────────────────────
print("Loading source tables...")
loads     = pd.read_csv(LOADS_CSV)
routes    = pd.read_csv(ROUTES_CSV)
customers = pd.read_csv(CUSTOMERS_CSV)
trips     = pd.read_csv(TRIPS_CSV)
fuel      = pd.read_csv(FUEL_CSV)

# ── Q1(a) Dataset preview ─────────────────────────────────────
print("\n" + "="*65)
print("Q1(a) DATASET PREVIEW — loads.head()")
print("="*65)
print(loads.head())
print()
loads.info()

# ── Build master dataset ──────────────────────────────────────
print("\nBuilding master dataset...")
df = loads.merge(routes,    on='route_id',    how='left', suffixes=('', '_route'))
df = df.merge(customers,    on='customer_id', how='left', suffixes=('', '_cust'))

trip_fuel = fuel.groupby('trip_id').agg(
    fuel_cost=('total_cost', 'sum')
).reset_index()
trip_link = trips[['trip_id', 'load_id', 'actual_distance_miles']]
load_cost = trip_link.merge(trip_fuel, on='trip_id', how='left')
df = df.merge(load_cost[['load_id', 'fuel_cost', 'actual_distance_miles']],
              on='load_id', how='left')

df['total_revenue_with_surcharge'] = df['revenue'] + df['fuel_surcharge']
df['estimated_profit'] = df['total_revenue_with_surcharge'] - df['fuel_cost'].fillna(0)
df['profit_margin_pct'] = df['estimated_profit'] / df['total_revenue_with_surcharge']

df.to_csv(MASTER_CSV, index=False)
print(f"Master dataset saved: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Q1(b) EDA Charts ──────────────────────────────────────────
print("\nGenerating EDA charts...")

# Chart 1: Revenue distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['revenue'].dropna(), bins=40, kde=True, color='#3b6fa0')
plt.title('Distribution of Shipment Revenue ($)')
plt.xlabel('Revenue ($)')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/dist_revenue.png', dpi=130)
plt.close()

# Chart 2: Revenue vs distance scatter
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df.sample(5000, random_state=42),
                x='typical_distance_miles', y='revenue',
                hue='load_type', alpha=0.4, s=15)
plt.title('Revenue vs Route Distance by Load Type')
plt.xlabel('Typical Distance (miles)')
plt.ylabel('Revenue ($)')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/revenue_vs_distance.png', dpi=130)
plt.close()

# Chart 3: Correlation heatmap
num_cols = ['revenue', 'weight_lbs', 'pieces', 'typical_distance_miles',
            'base_rate_per_mile', 'fuel_surcharge_rate', 'typical_transit_days',
            'credit_terms_days', 'annual_revenue_potential']
corr = df[num_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Key Numeric Features')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/correlation_heatmap.png', dpi=130)
plt.close()

# Chart 4: Revenue by booking type
cat_summary = df.groupby('booking_type')['revenue'].mean().sort_values()
plt.figure(figsize=(7, 4))
cat_summary.plot(kind='barh', color='#3b6fa0')
plt.title('Average Revenue by Booking Type')
plt.xlabel('Average Revenue ($)')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/revenue_by_booking_type.png', dpi=130)
plt.close()

print(f"Charts saved to: {CHARTS_DIR}/")

# ── Q1(c) Data Quality Checks ─────────────────────────────────
print("\n" + "="*65)
print("Q1(c) DATA QUALITY CHECKS")
print("="*65)

# Check 1: Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nCheck 1 — Missing Values:")
print(missing)

# Check 2: Duplicate records
dup_load = df.duplicated(subset=['load_id']).sum()
dup_full = df.duplicated().sum()
print(f"\nCheck 2 — Duplicate Records:")
print(f"  Duplicate load_id : {dup_load:,}")
print(f"  Fully duplicate   : {dup_full:,}")

# Issue 2: Implausible profit margins
implausible = (df['profit_margin_pct'] < -1.0).sum()
worst = df.nsmallest(1, 'profit_margin_pct')
print(f"\nIssue 2 — Implausible Profit Margins (< -100%):")
print(f"  Rows flagged  : {implausible:,} ({implausible/len(df)*100:.2f}%)")
print(f"  Worst case    : {worst['load_id'].values[0]}")
print(f"  Worst margin  : {worst['profit_margin_pct'].values[0]*100:.1f}%")
print(f"  Revenue       : ${worst['revenue'].values[0]:,.2f}")
print(f"  Fuel cost     : ${worst['fuel_cost'].values[0]:,.2f}")

print("\nQ1 COMPLETE.")
