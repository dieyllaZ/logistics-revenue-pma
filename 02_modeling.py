"""
Q2 — Feature Engineering, Baseline and Improved Model
Run: python 02_modeling.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (MASTER_CLEAN_CSV, CHARTS_DIR, OUTPUTS_DIR,
                    BASELINE_MODEL, IMPROVED_MODEL, MODEL_RESULTS,
                    TEST_PREDICTIONS, TARGET, TEST_SIZE, RANDOM_STATE,
                    NUMERIC_BASE, CATEGORICAL_BASE,
                    NUMERIC_ENG, CATEGORICAL_ENG,
                    NUMERIC_FULL, CATEGORICAL_FULL)

# ── Load clean dataset ────────────────────────────────────────
df = pd.read_csv(MASTER_CLEAN_CSV)
print("="*65)
print("Q2 — MODELLING (master_dataset_clean.csv)")
print("="*65)
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Feature matrices ──────────────────────────────────────────
X_base = df[NUMERIC_BASE + CATEGORICAL_BASE]
X_full = df[NUMERIC_FULL + CATEGORICAL_FULL]
y      = df[TARGET]

Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_base, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
Xf_train, Xf_test, _,      _      = train_test_split(
    X_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"Train: {len(Xb_train):,}  |  Test: {len(Xb_test):,}")

# ── Preprocessors ─────────────────────────────────────────────
prep_base = ColumnTransformer(transformers=[
    ('num', StandardScaler(),              NUMERIC_BASE),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_BASE)
])
prep_full = ColumnTransformer(transformers=[
    ('num', StandardScaler(),              NUMERIC_FULL),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FULL)
])

# ════════════════════════════════════════════════════════════════
# SPRINT 2 — Baseline: Linear Regression
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SPRINT 2 — Baseline: Linear Regression")
print("="*65)

baseline_pipe = Pipeline([('prep', prep_base), ('reg', LinearRegression())])
baseline_pipe.fit(Xb_train, y_train)
y_pred_base = baseline_pipe.predict(Xb_test)

base_metrics = {
    'MAE' : mean_absolute_error(y_test, y_pred_base),
    'RMSE': mean_squared_error(y_test, y_pred_base) ** 0.5,
    'R2'  : r2_score(y_test, y_pred_base)
}
print(f"MAE       : ${base_metrics['MAE']:,.2f}")
print(f"RMSE      : ${base_metrics['RMSE']:,.2f}")
print(f"R-squared :  {base_metrics['R2']:.4f}")

lims = [y_test.min(), y_test.max()]
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_base, alpha=0.15, s=10, color='#3b6fa0')
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Revenue ($)')
plt.ylabel('Predicted Revenue ($)')
plt.title('Sprint 2 — Baseline (Linear Regression)\nPredicted vs Actual Revenue')
plt.legend()
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/sprint2_baseline_pred_vs_actual.png', dpi=130)
plt.close()

# ════════════════════════════════════════════════════════════════
# SPRINT 3 — Improved: Random Forest + tuning
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SPRINT 3 — Improved: Random Forest (GridSearchCV)")
print("="*65)

rf_pipe = Pipeline([('prep', prep_full),
                    ('reg', RandomForestRegressor(random_state=RANDOM_STATE))])
param_grid = {
    'reg__n_estimators': [100, 200],
    'reg__max_depth'   : [10, 16],
}
print("Running GridSearchCV (2-fold, neg_MAE) — this takes a few minutes...")
grid = GridSearchCV(rf_pipe, param_grid, cv=2,
                    scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(Xf_train, y_train)
improved_pipe = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

y_pred_imp = improved_pipe.predict(Xf_test)
imp_metrics = {
    'MAE' : mean_absolute_error(y_test, y_pred_imp),
    'RMSE': mean_squared_error(y_test, y_pred_imp) ** 0.5,
    'R2'  : r2_score(y_test, y_pred_imp)
}
print(f"MAE       : ${imp_metrics['MAE']:,.2f}")
print(f"RMSE      : ${imp_metrics['RMSE']:,.2f}")
print(f"R-squared :  {imp_metrics['R2']:.4f}")

mae_impr = (base_metrics['MAE'] - imp_metrics['MAE']) / base_metrics['MAE'] * 100
print(f"\nMAE improvement: {mae_impr:.1f}% reduction from baseline")

# Charts
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_imp, alpha=0.15, s=10, color='#2e7d32')
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Revenue ($)')
plt.ylabel('Predicted Revenue ($)')
plt.title('Sprint 3 — Improved (Random Forest)\nPredicted vs Actual Revenue')
plt.legend()
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/sprint3_improved_pred_vs_actual.png', dpi=130)
plt.close()

plt.figure(figsize=(8, 5))
sns.kdeplot(y_test - y_pred_base, label='Baseline residuals',
            fill=True, alpha=0.3, color='#3b6fa0')
sns.kdeplot(y_test - y_pred_imp, label='Improved residuals',
            fill=True, alpha=0.3, color='#2e7d32')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residual Distribution: Baseline vs Improved Model')
plt.xlabel('Actual − Predicted Revenue ($)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/residuals_comparison.png', dpi=130)
plt.close()

ohe       = improved_pipe.named_steps['prep'].named_transformers_['cat']
cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FULL))
all_names = NUMERIC_FULL + cat_names
imp_series = (pd.Series(improved_pipe.named_steps['reg'].feature_importances_,
                         index=all_names)
              .sort_values(ascending=False).head(15))
plt.figure(figsize=(8, 6))
imp_series.sort_values().plot(kind='barh', color='#2e7d32')
plt.title('Top 15 Feature Importances — Improved Model (Random Forest)')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/feature_importance.png', dpi=130)
plt.close()

# ── Save models and results ───────────────────────────────────
joblib.dump(baseline_pipe, BASELINE_MODEL)
joblib.dump(improved_pipe, IMPROVED_MODEL)

results = {
    'baseline': {**base_metrics, 'model': 'LinearRegression'},
    'improved': {**imp_metrics,  'model': str(grid.best_params_)},
    'mae_improvement_pct': mae_impr,
    'features_numeric':    NUMERIC_FULL,
    'features_categorical': CATEGORICAL_FULL,
}
with open(MODEL_RESULTS, 'w') as f:
    json.dump(results, f, indent=2, default=str)

test_out = Xf_test.copy()
test_out['actual_revenue'] = y_test.values
test_out['pred_baseline']  = y_pred_base
test_out['pred_improved']  = y_pred_imp
test_out['residual_base']  = y_test.values - y_pred_base
test_out['residual_imp']   = y_test.values - y_pred_imp
test_out.to_csv(TEST_PREDICTIONS, index=False)

print(f"\nSaved: {BASELINE_MODEL}")
print(f"Saved: {IMPROVED_MODEL}")
print(f"Saved: {MODEL_RESULTS}")
print(f"Saved: {TEST_PREDICTIONS}")
print("\nQ2 COMPLETE.")
