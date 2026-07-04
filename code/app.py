"""
Q4/Q5 — Streamlit Dashboard
Shipment Revenue & Route Profitability Analytics

Run from project root:
    streamlit run code/app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ── Resolve paths relative to project root ────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT  = os.path.join(BASE, 'outputs')

st.set_page_config(
    page_title="Revenue & Route Profitability Dashboard",
    layout="wide",
    page_icon="📦"
)


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA, 'master_dataset_clean.csv'))


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(OUT, 'improved_model.pkl'))


@st.cache_data
def load_predictions():
    return pd.read_csv(os.path.join(OUT, 'test_predictions.csv'))


df    = load_data()
model = load_model()
preds = load_predictions()

# ── Header ────────────────────────────────────────────────────
st.title("📦 Shipment Revenue & Route Profitability Dashboard")
st.caption("PMA MRTB 2173 — Agile Data Science MVP | Logistics Operations")

# ════════════════════════════════════════════════════════════════
# SIDEBAR — Interactive filters (Q4 a-ii)
# ════════════════════════════════════════════════════════════════
st.sidebar.header("🔽 Filters")

# Interactive feature 1: Multiselect
load_type_filter = st.sidebar.multiselect(
    "Load Type",
    options=sorted(df['load_type'].dropna().unique()),
    default=sorted(df['load_type'].dropna().unique())
)

# Interactive feature 2: Dropdown
booking_type_filter = st.sidebar.selectbox(
    "Booking Type",
    options=['All'] + sorted(df['booking_type'].dropna().unique().tolist())
)

# Interactive feature 3: Revenue range slider
rev_min = int(df['revenue'].min())
rev_max = int(df['revenue'].max())
rev_range = st.sidebar.slider(
    "Revenue Range ($)",
    min_value=rev_min, max_value=rev_max,
    value=(rev_min, rev_max)
)

# Interactive feature 4: Checkbox
reliable_only = st.sidebar.checkbox(
    "Show only reliable profit margins",
    value=False,
    help="Excludes loads with misassigned or missing fuel cost records"
)

# Apply filters
filtered = df[df['load_type'].isin(load_type_filter)].copy()
if booking_type_filter != 'All':
    filtered = filtered[filtered['booking_type'] == booking_type_filter]
filtered = filtered[
    (filtered['revenue'] >= rev_range[0]) &
    (filtered['revenue'] <= rev_range[1])
]
if reliable_only:
    filtered = filtered[filtered['profitability_confidence'] == 'Reliable']

st.markdown(f"**{len(filtered):,} loads match current filters** "
            f"(of {len(df):,} total)")

# ── KPI row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",         f"${filtered['revenue'].sum():,.0f}")
k2.metric("Avg Revenue / Load",    f"${filtered['revenue'].mean():,.0f}")
k3.metric("Avg Rate ($/mile)",     f"${filtered['base_rate_per_mile'].mean():.2f}")
reliable = filtered[filtered['profitability_confidence'] == 'Reliable']
k4.metric("Avg Profit Margin",
          f"{reliable['profit_margin_clean'].mean()*100:.1f}%"
          if len(reliable) > 0 else "N/A")
k5.metric("Reliable Margin Coverage",
          f"{(filtered['profitability_confidence']=='Reliable').mean()*100:.1f}%")

st.divider()

# ════════════════════════════════════════════════════════════════
# VISUALIZATION 1 — Revenue distribution
# ════════════════════════════════════════════════════════════════
st.subheader("📊 1. Shipment Revenue Distribution")
fig1 = px.histogram(
    filtered, x='revenue', nbins=40, color='load_type',
    labels={'revenue': 'Revenue ($)', 'load_type': 'Load Type'},
    title=f"Revenue Distribution — {len(filtered):,} loads"
)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Revenue is right-skewed, concentrated between $1,000–$5,000, "
           "reflecting a mixed regional and long-haul book of business.")

# ════════════════════════════════════════════════════════════════
# VISUALIZATION 2 — Revenue by customer type
# ════════════════════════════════════════════════════════════════
st.subheader("👥 2. Average Revenue by Customer Type & Booking Type")
v2  = (filtered.groupby(['customer_type', 'booking_type'])['revenue']
       .mean().reset_index())
fig2 = px.bar(
    v2, x='customer_type', y='revenue', color='booking_type',
    barmode='group',
    labels={'revenue': 'Avg Revenue ($)', 'customer_type': 'Customer Type'},
    title="Average Revenue by Customer Type and Booking Type"
)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Revenue is broadly similar across booking types, suggesting "
           "pricing is driven primarily by route distance rather than contract type.")

# ════════════════════════════════════════════════════════════════
# VISUALIZATION 3 — Route profitability
# ════════════════════════════════════════════════════════════════
st.subheader("🛣️ 3. Route Profitability — Top & Bottom Lanes")
st.caption("⚠️ Only **Reliable** margin data shown — "
           "misassigned/missing fuel records excluded.")

reliable_df = filtered[filtered['profitability_confidence'] == 'Reliable'].copy()
if len(reliable_df) > 0 and 'origin_city' in reliable_df.columns:
    reliable_df['lane'] = (reliable_df['origin_city'] + ', ' +
                            reliable_df['origin_state'] + ' → ' +
                            reliable_df['destination_city'] + ', ' +
                            reliable_df['destination_state'])
    lane_m = (reliable_df.groupby('lane')['profit_margin_clean']
              .agg(['mean', 'count']).reset_index())
    lane_m.columns = ['lane', 'avg_margin', 'load_count']
    lane_m = lane_m[lane_m['load_count'] >= 5]
    chart3 = pd.concat([
        lane_m.nsmallest(8, 'avg_margin'),
        lane_m.nlargest(8,  'avg_margin')
    ]).sort_values('avg_margin')
    fig3 = px.bar(
        chart3, x='avg_margin', y='lane', orientation='h',
        color='avg_margin', color_continuous_scale='RdYlGn',
        labels={'avg_margin': 'Avg Profit Margin'},
        title="Top 8 & Bottom 8 Lanes by Avg Profit Margin (reliable only)"
    )
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No reliable margin data with current filters.")

st.divider()

# ════════════════════════════════════════════════════════════════
# PREDICTIVE OUTPUT (Q4 a-iii)
# ════════════════════════════════════════════════════════════════
st.subheader("🔮 Predict Expected Revenue for a New Load")
st.markdown("Enter load details and click **Predict** to get an estimated "
            "revenue from the Sprint 3 Random Forest model (R² = 0.9725).")

colA, colB, colC = st.columns(3)
with colA:
    p_distance = st.number_input("Typical Distance (miles)", 50, 3000, 500)
    p_weight   = st.number_input("Weight (lbs)", 1000, 45000, 20000)
with colB:
    p_rate     = st.number_input("Base Rate ($/mile)", 0.50, 5.00, 2.20, step=0.05)
    p_pieces   = st.number_input("Pieces", 1, 30, 14)
with colC:
    p_lt = st.selectbox("Load Type",    sorted(df['load_type'].dropna().unique()))
    p_ct = st.selectbox("Customer Type", sorted(df['customer_type'].dropna().unique()))

if st.button("⚡ Predict Expected Revenue", type="primary"):
    sample = df.iloc[[0]].copy()
    sample['typical_distance_miles'] = p_distance
    sample['weight_lbs']             = p_weight
    sample['base_rate_per_mile']     = p_rate
    sample['pieces']                 = p_pieces
    sample['load_type']              = p_lt
    sample['customer_type']          = p_ct
    sample['weight_per_piece']       = p_weight / p_pieces
    sample['expected_rate_revenue']  = p_distance * p_rate
    feature_cols = list(model.named_steps['prep'].feature_names_in_)
    pred = model.predict(sample[feature_cols])[0]
    simple = p_distance * p_rate
    col1, col2 = st.columns(2)
    col1.success(f"**Predicted Revenue: ${pred:,.0f}**")
    col2.info(f"Simple rate × distance: ${simple:,.0f}  "
              f"| Model adjustment: {((pred/simple)-1)*100:+.1f}%")

st.divider()

# ════════════════════════════════════════════════════════════════
# MONITORING SECTION (Q5 a)
# ════════════════════════════════════════════════════════════════
st.header("📈 Model & Data Monitoring")

actual   = preds['actual_revenue']
pred_imp = preds['pred_improved']
live_mae = np.mean(np.abs(actual - pred_imp))
live_r2  = 1 - np.sum((actual-pred_imp)**2) / np.sum((actual-actual.mean())**2)
drift_d  = abs(filtered['typical_distance_miles'].mean() -
               df['typical_distance_miles'].mean())
drift_r  = abs(filtered['revenue'].mean() - df['revenue'].mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model MAE (test set)",       f"${live_mae:,.0f}")
m2.metric("Model R² (test set)",        f"{live_r2:.4f}")
m3.metric("Distance Feature Drift (mi)", f"{drift_d:.1f}")
m4.metric("Revenue Feature Drift ($)",  f"${drift_r:,.0f}")

# Profitability confidence monitor
st.subheader("Data Quality Monitor — Profitability Confidence")
conf_counts = df['profitability_confidence'].value_counts().reset_index()
conf_counts.columns = ['Status', 'Count']
fig_m = px.bar(
    conf_counts, x='Status', y='Count', color='Status',
    color_discrete_map={
        'Reliable'                              : '#2e7d32',
        'Low margin — monitor'                  : '#f9a825',
        'Unknown — no fuel data'                : '#78909c',
        'Unreliable — misassigned fuel records' : '#c62828'
    },
    title="Profitability Confidence Status Across All Loads"
)
st.plotly_chart(fig_m, use_container_width=True)

# Residual distribution
st.subheader("Residual Distribution Monitor")
fig_r = px.histogram(
    preds, x='residual_imp', nbins=60,
    labels={'residual_imp': 'Residual: Actual − Predicted ($)'},
    title="Model Residual Distribution (test set)"
)
fig_r.add_vline(x=0, line_dash="dash", line_color="red")
st.plotly_chart(fig_r, use_container_width=True)
