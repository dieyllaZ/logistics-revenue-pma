"""
Q2 — Feature Engineering, Baseline Model (Sprint 2), Improved Model (Sprint 3)
Target  : revenue ($) — predict expected shipment revenue at booking time

Run from project root:
    python code/02_modeling.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os

from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LinearRegression
from sklearn.ensemble           import RandomForestRegressor
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'data')
CHARTS = os.path.join(BASE, 'charts')
OUT    = os.path.join(BASE, 'outputs')

for d in [CHARTS, OUT]:
    os.makedirs(d, exist_ok=True)

# ── Load clean dataset ────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, 'master_dataset_clean.csv'))
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

TARGET = 'revenue'

# ── Feature selection ─────────────────────────────────────────
# Baseline (Sprint 2): raw booking-time fields only
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

# Improved (Sprint 3): raw + 4 engineered features
NUMERIC_FULL     = NUMERIC_BASE + [
    'weight_per_piece', 'expected_rate_revenue', 'customer_tenure_years'
]
CATEGORICAL_FULL = CATEGORICAL_BASE + ['revenue_potential_bucket']

X_base = df[NUMERIC_BASE     + CATEGORICAL_BASE]
X_full = df[NUMERIC_FULL     + CATEGORICAL_FULL]
y      = df[TARGET]

# ── Train / test split ────────────────────────────────────────
Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.25, random_state=42
)
Xf_train, Xf_test, _,      _      = train_test_split(
    X_full, y, test_size=0.25, random_state=42
)
print(f"\nTrain rows: {len(Xb_train):,}  |  Test rows: {len(Xb_test):,}")

# ── Preprocessors ─────────────────────────────────────────────
prep_base = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                       NUMERIC_BASE),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_BASE)
])
prep_full = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                       NUMERIC_FULL),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FULL)
])

# ════════════════════════════════════════════════════════════════
# SPRINT 2 — Baseline: Linear Regression
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SPRINT 2 — Baseline Model: Linear Regression")
print("="*60)

baseline = Pipeline([('prep', prep_base), ('reg', LinearRegression())])
baseline.fit(Xb_train, y_train)
y_pred_base = baseline.predict(Xb_test)

base_m = {
    'MAE' : mean_absolute_error(y_test, y_pred_base),
    'RMSE': mean_squared_error(y_test, y_pred_base) ** 0.5,
    'R2'  : r2_score(y_test, y_pred_base)
}
print(f"MAE  : ${base_m['MAE']:,.2f}")
print(f"RMSE : ${base_m['RMSE']:,.2f}")
print(f"R²   : {base_m['R2']:.4f}")

lims = [y_test.min(), y_test.max()]
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_base, alpha=0.15, s=10, color='#3b6fa0')
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Revenue ($)')
plt.ylabel('Predicted Revenue ($)')
plt.title('Sprint 2 — Baseline (Linear Regression)\nPredicted vs Actual Revenue')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'sprint2_baseline_pred_vs_actual.png'), dpi=130)
plt.close()
print("Chart saved: sprint2_baseline_pred_vs_actual.png")

# ════════════════════════════════════════════════════════════════
# SPRINT 3 — Improved: Random Forest + Tuning
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SPRINT 3 — Improved Model: Random Forest (tuned)")
print("="*60)

rf_pipe = Pipeline([('prep', prep_full),
                    ('reg',  RandomForestRegressor(random_state=42))])
param_grid = {
    'reg__n_estimators': [100, 200],
    'reg__max_depth'   : [10, 16],
}
print("Running GridSearchCV (2-fold, neg_MAE)...")
grid = GridSearchCV(rf_pipe, param_grid, cv=2,
                    scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(Xf_train, y_train)
improved   = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

y_pred_imp = improved.predict(Xf_test)
imp_m = {
    'MAE' : mean_absolute_error(y_test, y_pred_imp),
    'RMSE': mean_squared_error(y_test, y_pred_imp) ** 0.5,
    'R2'  : r2_score(y_test, y_pred_imp)
}
print(f"MAE  : ${imp_m['MAE']:,.2f}")
print(f"RMSE : ${imp_m['RMSE']:,.2f}")
print(f"R²   : {imp_m['R2']:.4f}")
print(f"MAE improvement: {(base_m['MAE']-imp_m['MAE'])/base_m['MAE']*100:.1f}%")

# Predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_imp, alpha=0.15, s=10, color='#2e7d32')
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Revenue ($)')
plt.ylabel('Predicted Revenue ($)')
plt.title('Sprint 3 — Improved (Random Forest tuned)\nPredicted vs Actual Revenue')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'sprint3_improved_pred_vs_actual.png'), dpi=130)
plt.close()

# Residuals comparison
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test - y_pred_base, label='Baseline residuals',
            fill=True, alpha=0.3, color='#3b6fa0')
sns.kdeplot(y_test - y_pred_imp,  label='Improved residuals',
            fill=True, alpha=0.3, color='#2e7d32')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residual Distribution: Baseline vs Improved Model')
plt.xlabel('Actual − Predicted Revenue ($)')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'residuals_comparison.png'), dpi=130)
plt.close()

# Feature importance
ohe       = improved.named_steps['prep'].named_transformers_['cat']
cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FULL))
all_names = NUMERIC_FULL + cat_names
imp_s     = (pd.Series(improved.named_steps['reg'].feature_importances_,
                        index=all_names)
             .sort_values(ascending=False).head(15))
plt.figure(figsize=(8, 6))
imp_s.sort_values().plot(kind='barh', color='#2e7d32')
plt.title('Top 15 Feature Importances — Improved Model')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'feature_importance.png'), dpi=130)
plt.close()
print("Charts saved: improved, residuals, feature_importance")

# ── Save models and results ───────────────────────────────────
joblib.dump(baseline, os.path.join(OUT, 'baseline_model.pkl'))
joblib.dump(improved, os.path.join(OUT, 'improved_model.pkl'))

results = {
    'baseline': {**base_m, 'model': 'LinearRegression'},
    'improved': {**imp_m,  'model': str(grid.best_params_)},
    'mae_improvement_pct': (base_m['MAE']-imp_m['MAE'])/base_m['MAE']*100
}
with open(os.path.join(OUT, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save test predictions for Q5 monitoring
test_out = Xf_test.copy()
test_out['actual_revenue'] = y_test.values
test_out['pred_baseline']  = y_pred_base
test_out['pred_improved']  = y_pred_imp
test_out['residual_base']  = y_test.values - y_pred_base
test_out['residual_imp']   = y_test.values - y_pred_imp
test_out.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

print("\nSaved: baseline_model.pkl, improved_model.pkl,")
print("       model_results.json, test_predictions.csv")
print("\nQ2 complete.")
