"""
Q5 — Drift Monitoring and Model Performance Degradation
Run: python 03_drift_monitoring.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import joblib, json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import (MASTER_CLEAN_CSV, CHARTS_DIR, OUTPUTS_DIR,
                    IMPROVED_MODEL, MODEL_RESULTS, DRIFT_SUMMARY, DRIFT_RESULTS)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(MASTER_CLEAN_CSV)
df['load_date'] = pd.to_datetime(df['load_date'], errors='coerce')
df = df.dropna(subset=['load_date']).sort_values('load_date')

print("="*65)
print("Q5 — DRIFT & PERFORMANCE DEGRADATION")
print("="*65)

# ── Chronological split ───────────────────────────────────────
cutoff  = df['load_date'].quantile(0.70)
train_w = df[df['load_date'] <= cutoff]
recent_w = df[df['load_date'] >  cutoff]

print(f"Train window  : {train_w['load_date'].min().date()} to "
      f"{train_w['load_date'].max().date()} ({len(train_w):,} rows)")
print(f"Recent window : {recent_w['load_date'].min().date()} to "
      f"{recent_w['load_date'].max().date()} ({len(recent_w):,} rows)")

# ── Feature drift: KS test ────────────────────────────────────
print("\n── Feature Drift (KS test) ──")
drift_features = ['typical_distance_miles', 'base_rate_per_mile',
                   'weight_lbs', 'revenue', 'expected_rate_revenue']
drift_results  = []
for feat in drift_features:
    a      = train_w[feat].dropna()
    b      = recent_w[feat].dropna()
    ks, pv = stats.ks_2samp(a, b)
    drift_results.append({
        'feature'       : feat,
        'train_mean'    : round(a.mean(), 2),
        'recent_mean'   : round(b.mean(), 2),
        'mean_shift'    : round(b.mean() - a.mean(), 2),
        'ks_statistic'  : round(ks, 4),
        'p_value'       : round(pv, 4),
        'drift_detected': pv < 0.05
    })

drift_df = pd.DataFrame(drift_results)
print(drift_df.to_string(index=False))
drift_df.to_csv(DRIFT_RESULTS, index=False)

# ── Feature drift chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(train_w['base_rate_per_mile'], bins=40, alpha=0.5,
             label='Train', density=True, color='#3b6fa0')
axes[0].hist(recent_w['base_rate_per_mile'], bins=40, alpha=0.5,
             label='Recent', density=True, color='#d32f2f')
axes[0].legend(); axes[0].set_title('Drift: Base Rate per Mile')

axes[1].hist(train_w['typical_distance_miles'], bins=40, alpha=0.5,
             label='Train', density=True, color='#3b6fa0')
axes[1].hist(recent_w['typical_distance_miles'], bins=40, alpha=0.5,
             label='Recent', density=True, color='#d32f2f')
axes[1].legend(); axes[1].set_title('Drift: Typical Distance (miles)')

plt.suptitle('Q5 — Feature Drift: Train vs Recent Window',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/drift_feature_distributions.png', dpi=130)
plt.close()

# ── Label drift: monthly revenue ─────────────────────────────
df['year_month'] = df['load_date'].dt.to_period('M').astype(str)
monthly_rev = df.groupby('year_month')['revenue'].mean()

plt.figure(figsize=(11, 5))
monthly_rev.plot(marker='o', color='#3b6fa0', linewidth=1.5, markersize=4)
plt.axhline(monthly_rev.mean(), color='gray', linestyle='--', label='Overall mean')
plt.title('Monthly Average Revenue — Label Drift Check')
plt.ylabel('Average Revenue ($)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/drift_label_monthly_revenue.png', dpi=130)
plt.close()

# ── Performance degradation ───────────────────────────────────
print("\n── Model Performance on Recent Window ──")
model        = joblib.load(IMPROVED_MODEL)
feature_cols = list(model.named_steps['prep'].feature_names_in_)

recent_eval = recent_w.copy()
for c in feature_cols:
    if c not in recent_eval.columns:
        recent_eval[c] = np.nan

y_true  = recent_eval['revenue']
y_pred  = model.predict(recent_eval[feature_cols])

recent_metrics = {
    'MAE' : mean_absolute_error(y_true, y_pred),
    'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
    'R2'  : r2_score(y_true, y_pred)
}

with open(MODEL_RESULTS) as f:
    orig = json.load(f)

print(f"{'Metric':<8} {'Original Test':>15} {'Recent Window':>15} {'Change':>10}")
print("-"*52)
for m in ['MAE', 'RMSE', 'R2']:
    ov  = orig['improved'][m]
    rv  = recent_metrics[m]
    chg = rv - ov
    sgn = '+' if chg >= 0 else ''
    if m in ('MAE','RMSE'):
        print(f"{m:<8} ${ov:>14,.2f} ${rv:>14,.2f} {sgn}${chg:>8,.2f}")
    else:
        print(f"{m:<8}  {ov:>14.4f}  {rv:>14.4f} {sgn}{chg:>9.4f}")

# Monthly performance chart
monthly_perf = []
for ym, grp in df.groupby('year_month'):
    g = grp.copy()
    for c in feature_cols:
        if c not in g.columns:
            g[c] = np.nan
    p    = model.predict(g[feature_cols])
    mae  = mean_absolute_error(g['revenue'], p)
    r2   = r2_score(g['revenue'], p)
    monthly_perf.append({'month': ym, 'MAE': mae, 'R2': r2})

perf_df = pd.DataFrame(monthly_perf)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
ax1.plot(perf_df['month'], perf_df['MAE'], color='#d32f2f',
         marker='o', markersize=4, linewidth=1.5, label='MAE ($)')
ax2.plot(perf_df['month'], perf_df['R2'], color='#3b6fa0',
         marker='s', markersize=4, linewidth=1.5, linestyle='--', label='R²')
ax1.set_ylabel('MAE ($)', color='#d32f2f')
ax2.set_ylabel('R²',      color='#3b6fa0')
ax1.set_xlabel('Month')
ax1.tick_params(axis='x', rotation=45)
plt.title('Model Performance Over Time — Monthly MAE and R²')
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper left')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/model_performance_over_time.png', dpi=130)
plt.close()

# Save summary
summary = {
    'train_window'    : {'start': str(train_w['load_date'].min().date()),
                          'end'  : str(train_w['load_date'].max().date()),
                          'rows' : len(train_w)},
    'recent_window'   : {'start': str(recent_w['load_date'].min().date()),
                          'end'  : str(recent_w['load_date'].max().date()),
                          'rows' : len(recent_w)},
    'drift_results'   : drift_results,
    'original_metrics': orig['improved'],
    'recent_metrics'  : recent_metrics,
}
with open(DRIFT_SUMMARY, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nSaved: {DRIFT_RESULTS}")
print(f"Saved: {DRIFT_SUMMARY}")
print("\nQ5 COMPLETE.")
