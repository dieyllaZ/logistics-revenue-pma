"""
Q5(b) — Data Drift and Model Performance Degradation Analysis

Run from project root:
    python code/03_drift_monitoring.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import joblib, json, os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'data')
CHARTS = os.path.join(BASE, 'charts')
OUT    = os.path.join(BASE, 'outputs')

for d in [CHARTS, OUT]:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'master_dataset_clean.csv'))
df['load_date'] = pd.to_datetime(df['load_date'], errors='coerce')
df = df.dropna(subset=['load_date']).sort_values('load_date')

# ── Chronological split ───────────────────────────────────────
cutoff  = df['load_date'].quantile(0.70)
train_w = df[df['load_date'] <= cutoff]
recent  = df[df['load_date'] >  cutoff]

print("="*60)
print("Q5(b) DRIFT & PERFORMANCE DEGRADATION ANALYSIS")
print("="*60)
print(f"Train  : {train_w['load_date'].min().date()} → "
      f"{train_w['load_date'].max().date()} ({len(train_w):,} rows)")
print(f"Recent : {recent['load_date'].min().date()} → "
      f"{recent['load_date'].max().date()} ({len(recent):,} rows)")

# ── Feature drift: KS test ────────────────────────────────────
print("\n── Feature Drift (KS test) ──")
drift_features = ['typical_distance_miles', 'base_rate_per_mile',
                  'weight_lbs', 'revenue', 'expected_rate_revenue']
drift_results = []
for feat in drift_features:
    a, b   = train_w[feat].dropna(), recent[feat].dropna()
    ks, pv = stats.ks_2samp(a, b)
    drift_results.append({
        'feature'       : feat,
        'train_mean'    : round(a.mean(), 2),
        'recent_mean'   : round(b.mean(), 2),
        'ks_statistic'  : round(ks, 4),
        'p_value'       : round(pv, 4),
        'drift_detected': pv < 0.05
    })

drift_df = pd.DataFrame(drift_results)
print(drift_df.to_string(index=False))
drift_df.to_csv(os.path.join(OUT, 'drift_results.csv'), index=False)

# Feature drift chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat in zip(axes, ['base_rate_per_mile', 'typical_distance_miles']):
    ax.hist(train_w[feat], bins=40, alpha=0.5,
            label='Train', density=True, color='#3b6fa0')
    ax.hist(recent[feat],  bins=40, alpha=0.5,
            label='Recent', density=True, color='#d32f2f')
    ax.set_title(f'Drift: {feat}')
    ax.legend()
plt.suptitle('Q5(b) Feature Drift: Train vs Recent Window', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'drift_feature_distributions.png'), dpi=130)
plt.close()
print("Chart saved: drift_feature_distributions.png")

# ── Label drift: monthly average revenue ─────────────────────
print("\n── Label Drift (monthly avg revenue) ──")
df['year_month']  = df['load_date'].dt.to_period('M').astype(str)
monthly_rev       = df.groupby('year_month')['revenue'].mean()
overall_mean      = monthly_rev.mean()
monthly_std       = monthly_rev.std()
print(f"Overall monthly mean : ${overall_mean:,.2f}")
print(f"CV (std/mean)        : {monthly_std/overall_mean*100:.1f}%")

plt.figure(figsize=(11, 5))
monthly_rev.plot(marker='o', color='#3b6fa0', linewidth=1.5, markersize=5)
plt.axhline(overall_mean, color='gray', linestyle='--', label='Overall mean')
plt.axhline(overall_mean + 2*monthly_std, color='orange',
            linestyle=':', label='±2σ')
plt.axhline(overall_mean - 2*monthly_std, color='orange', linestyle=':')
plt.title('Q5(b) Label Drift: Monthly Average Revenue')
plt.ylabel('Avg Revenue ($)')
plt.xticks(rotation=45)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'drift_label_monthly_revenue.png'), dpi=130)
plt.close()
print("Chart saved: drift_label_monthly_revenue.png")

# ── Model performance on recent window ────────────────────────
print("\n── Model Performance: Recent Window ──")
model       = joblib.load(os.path.join(OUT, 'improved_model.pkl'))
feature_cols = list(model.named_steps['prep'].feature_names_in_)

recent_eval = recent.copy()
for c in [col for col in feature_cols if col not in recent_eval.columns]:
    recent_eval[c] = np.nan

y_true  = recent_eval['revenue']
y_pred  = model.predict(recent_eval[feature_cols])
rec_m   = {
    'MAE' : mean_absolute_error(y_true, y_pred),
    'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
    'R2'  : r2_score(y_true, y_pred)
}

with open(os.path.join(OUT, 'model_results.json')) as f:
    orig = json.load(f)

print(f"\n{'Metric':<8}  {'Original':>14}  {'Recent':>14}  {'Change':>12}")
print("-"*52)
for m in ['MAE', 'RMSE', 'R2']:
    o, r   = orig['improved'][m], rec_m[m]
    change = r - o
    sign   = '+' if change >= 0 else ''
    if m in ('MAE', 'RMSE'):
        print(f"{m:<8}  ${o:>13,.2f}  ${r:>13,.2f}  {sign}${change:>9,.2f}")
    else:
        print(f"{m:<8}  {o:>14.4f}  {r:>14.4f}  {sign}{change:>11.4f}")

# Monthly performance chart
monthly_perf = []
for ym, grp in df.groupby('year_month'):
    mc = [c for c in feature_cols if c not in grp.columns]
    for c in mc:
        grp = grp.copy(); grp[c] = np.nan
    pr   = model.predict(grp[feature_cols])
    monthly_perf.append({
        'month': ym,
        'MAE'  : mean_absolute_error(grp['revenue'], pr),
        'R2'   : r2_score(grp['revenue'], pr)
    })

perf_df = pd.DataFrame(monthly_perf)
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
ax1.plot(perf_df['month'], perf_df['MAE'], color='#d32f2f',
         marker='o', markersize=4, label='MAE ($)', linewidth=1.5)
ax2.plot(perf_df['month'], perf_df['R2'],  color='#3b6fa0',
         marker='s', markersize=4, label='R²', linewidth=1.5, linestyle='--')
ax1.set_ylabel('MAE ($)', color='#d32f2f')
ax2.set_ylabel('R²',       color='#3b6fa0')
ax1.tick_params(axis='x', rotation=45)
plt.title('Q5(b) Model Performance Over Time — Monthly MAE and R²')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'model_performance_over_time.png'), dpi=130)
plt.close()
print("Chart saved: model_performance_over_time.png")

summary = {
    'train_window'   : {'rows': len(train_w)},
    'recent_window'  : {'rows': len(recent)},
    'drift_results'  : drift_results,
    'original_metrics': orig['improved'],
    'recent_metrics' : rec_m,
}
with open(os.path.join(OUT, 'drift_and_degradation_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nQ5(b) complete.")
