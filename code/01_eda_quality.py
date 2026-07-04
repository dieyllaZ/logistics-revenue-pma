"""
Q1(a)(b)(c) — EDA and Data Quality Checks
Logistics Revenue & Route Profitability
PMA MRTB 2173

Run from project root:
    python code/01_eda_quality.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ─────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'data')
CHARTS = os.path.join(BASE, 'charts')
OUT    = os.path.join(BASE, 'outputs')

for d in [CHARTS, OUT]:
    os.makedirs(d, exist_ok=True)

sns.set_style('whitegrid')

# ── Load raw tables ───────────────────────────────────────────
print("Loading data...")
loads     = pd.read_csv(os.path.join(DATA, 'loads.csv'))
routes    = pd.read_csv(os.path.join(DATA, 'routes.csv'))
customers = pd.read_csv(os.path.join(DATA, 'customers.csv'))
trips     = pd.read_csv(os.path.join(DATA, 'trips.csv'))
fuel      = pd.read_csv(os.path.join(DATA, 'fuel_purchases.csv'))

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

# Link fuel cost for profitability analysis
fuel_per_trip = fuel.groupby('trip_id').agg(
    fuel_cost=('total_cost', 'sum')
).reset_index()
trip_link = trips[['trip_id', 'load_id', 'actual_distance_miles']]
load_cost  = trip_link.merge(fuel_per_trip, on='trip_id', how='left')
df = df.merge(load_cost[['load_id', 'fuel_cost', 'actual_distance_miles']],
              on='load_id', how='left')

# Derive profitability metric (raw — before cleaning)
df['total_revenue_with_surcharge'] = df['revenue'] + df['fuel_surcharge']
df['estimated_profit'] = (
    df['total_revenue_with_surcharge'] - df['fuel_cost'].fillna(0)
)
df['profit_margin_pct'] = (
    df['estimated_profit'] / df['total_revenue_with_surcharge']
)

df.to_csv(os.path.join(DATA, 'master_dataset.csv'), index=False)
print(f"Master dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Q1(b) EDA ─────────────────────────────────────────────────
print("\n" + "="*65)
print("Q1(b) EXPLORATORY DATA ANALYSIS")
print("="*65)

# 1. Distribution: revenue
plt.figure(figsize=(8, 5))
sns.histplot(df['revenue'].dropna(), bins=40, kde=True, color='#3b6fa0')
plt.title('Distribution of Shipment Revenue ($)')
plt.xlabel('Revenue ($)')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'dist_revenue.png'), dpi=130)
plt.close()
print("Chart saved: dist_revenue.png")

# 2. Relationship: revenue vs distance scatter
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df.sample(5000, random_state=42),
    x='typical_distance_miles', y='revenue',
    hue='load_type', alpha=0.4, s=15
)
plt.title('Revenue vs. Route Distance, by Load Type')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'revenue_vs_distance.png'), dpi=130)
plt.close()
print("Chart saved: revenue_vs_distance.png")

# 3. Relationship: correlation heatmap
num_cols = ['revenue', 'weight_lbs', 'typical_distance_miles',
            'base_rate_per_mile', 'fuel_surcharge_rate',
            'typical_transit_days', 'credit_terms_days',
            'annual_revenue_potential']
corr = df[num_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Key Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'correlation_heatmap.png'), dpi=130)
plt.close()
print("Chart saved: correlation_heatmap.png")
print("\nCorrelations with revenue:")
print(corr['revenue'].sort_values(ascending=False).to_string())

# 4. Categorical: revenue by booking type
cat_summary = df.groupby('booking_type')['revenue'].mean().sort_values()
plt.figure(figsize=(7, 4))
cat_summary.plot(kind='barh', color='#3b6fa0')
plt.title('Average Revenue by Booking Type')
plt.xlabel('Average Revenue ($)')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'revenue_by_booking_type.png'), dpi=130)
plt.close()
print("Chart saved: revenue_by_booking_type.png")
print("\nAverage revenue by booking_type:")
print(cat_summary)

# ── Q1(c) Data Quality Checks ─────────────────────────────────
print("\n" + "="*65)
print("Q1(c) DATA QUALITY CHECKS")
print("="*65)

# Check 1 — Missing values
print("\n── Check 1: Missing Values ──")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"Columns with missing values: {len(missing)}")
print(missing)

# Check 2 — Duplicate records
print("\n── Check 2: Duplicate Records ──")
dup_load_id  = df.duplicated(subset=['load_id']).sum()
dup_full_row = df.duplicated().sum()
print(f"Duplicate load_id rows : {dup_load_id:,}")
print(f"Fully duplicate rows   : {dup_full_row:,}")

# Check 3 — Implausible profit margins (Issue 2)
print("\n── Check 3: Implausible Profit Margins ──")
implausible_mask = df['profit_margin_pct'] < -1.0
n_impl = implausible_mask.sum()
print(f"Rows with profit_margin_pct < -100% : {n_impl:,} ({n_impl/len(df)*100:.2f}%)")
print(f"Most extreme value                  : {df['profit_margin_pct'].min()*100:.1f}%")
print("\nSample of implausible rows:")
print(df.loc[implausible_mask,
             ['load_id', 'revenue', 'fuel_surcharge',
              'fuel_cost', 'profit_margin_pct']]
      .sort_values('profit_margin_pct').head(5).to_string(index=False))

print("\nQ1 complete. Charts saved to charts/")
