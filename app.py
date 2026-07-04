"""
Q4/Q5 — Streamlit Dashboard
Run: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from config import (MASTER_CLEAN_CSV, IMPROVED_MODEL, TEST_PREDICTIONS)

st.set_page_config(
    page_title="Revenue & Route Profitability Dashboard",
    layout="wide",
    page_icon="📦"
)

@st.cache_data
def load_data():
    return pd.read_csv(MASTER_CLEAN_CSV)

@st.cache_resource
def load_model():
    return joblib.load(IMPROVED_MODEL)

@st.cache_data
def load_predictions():
    return pd.read_csv(TEST_PREDICTIONS)

df    = load_data()
model = load_model()
preds = load_predictions()

st.title("📦 Shipment Revenue & Route Profitability Dashboard")
st.caption("Agile Data Science MVP — MRTB 2173 | Logistics Operations")

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.header("🔽 Filters")

load_type_filter = st.sidebar.multiselect(
    "Load Type",
    options=sorted(df['load_type'].dropna().unique()),
    default=sorted(df['load_type'].dropna().unique())
)
booking_type_filter = st.sidebar.selectbox(
    "Booking Type",
    options=['All'] + sorted(df['booking_type'].dropna().unique().tolist())
)
rev_min = int(df['revenue'].min())
rev_max = int(df['revenue'].max())
rev_range = st.sidebar.slider("Revenue Range ($)", rev_min, rev_max, (rev_min, rev_max))
reliable_only = st.sidebar.checkbox("Show only reliable profit margins", value=False)

filtered = df[df['load_type'].isin(load_type_filter)].copy()
if booking_type_filter != 'All':
    filtered = filtered[filtered['booking_type'] == booking_type_filter]
filtered = filtered[(filtered['revenue'] >= rev_range[0]) &
                     (filtered['revenue'] <= rev_range[1])]
if reliable_only:
    filtered = filtered[filtered['profitability_confidence'] == 'Reliable']

st.markdown(f"**{len(filtered):,} loads match filters** (of {len(df):,} total)")

# ── KPIs ──────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"${filtered['revenue'].sum():,.0f}")
k2.metric("Avg Revenue / Load", f"${filtered['revenue'].mean():,.0f}")
k3.metric("Avg Rate ($/mile)", f"${filtered['base_rate_per_mile'].mean():.2f}")
reliable = filtered[filtered['profitability_confidence'] == 'Reliable']
k4.metric("Avg Profit Margin (reliable)",
           f"{reliable['profit_margin_clean'].mean()*100:.1f}%" if len(reliable) > 0 else "N/A")
k5.metric("Reliable Coverage",
           f"{(filtered['profitability_confidence']=='Reliable').mean()*100:.1f}%")

st.divider()

# ── Visualization 1: Revenue distribution ────────────────────
st.subheader("📊 1. Shipment Revenue Distribution")
fig1 = px.histogram(filtered, x='revenue', nbins=40, color='load_type',
                     labels={'revenue': 'Revenue ($)'})
st.plotly_chart(fig1, use_container_width=True)

# ── Visualization 2: Revenue by customer type ────────────────
st.subheader("👥 2. Average Revenue by Customer Type & Booking Type")
v2 = filtered.groupby(['customer_type', 'booking_type'])['revenue'].mean().reset_index()
fig2 = px.bar(v2, x='customer_type', y='revenue', color='booking_type',
               barmode='group', labels={'revenue': 'Avg Revenue ($)'})
st.plotly_chart(fig2, use_container_width=True)

# ── Visualization 3: Route profitability ─────────────────────
st.subheader("🛣️ 3. Route Profitability — Top & Bottom Lanes")
st.caption("Only Reliable margin loads shown.")
rel_df = filtered[filtered['profitability_confidence'] == 'Reliable'].copy()
if len(rel_df) > 0 and 'origin_city' in rel_df.columns:
    rel_df['lane'] = (rel_df['origin_city'] + ', ' + rel_df['origin_state'] +
                       ' → ' + rel_df['destination_city'] + ', ' + rel_df['destination_state'])
    lane_m = (rel_df.groupby('lane')['profit_margin_clean']
              .agg(['mean','count']).reset_index())
    lane_m.columns = ['lane','avg_margin','load_count']
    lane_m = lane_m[lane_m['load_count'] >= 5]
    chart3 = pd.concat([lane_m.nsmallest(8,'avg_margin'),
                         lane_m.nlargest(8,'avg_margin')]).sort_values('avg_margin')
    fig3 = px.bar(chart3, x='avg_margin', y='lane', orientation='h',
                   color='avg_margin', color_continuous_scale='RdYlGn',
                   labels={'avg_margin': 'Avg Profit Margin'})
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Predictive output ─────────────────────────────────────────
st.subheader("🔮 Predict Expected Revenue for a New Load")
cA, cB, cC = st.columns(3)
with cA:
    p_dist   = st.number_input("Distance (miles)", 50, 3000, 500)
    p_weight = st.number_input("Weight (lbs)", 1000, 45000, 20000)
with cB:
    p_rate   = st.number_input("Base Rate ($/mile)", 0.50, 5.00, 2.20, step=0.05)
    p_pieces = st.number_input("Pieces", 1, 30, 14)
with cC:
    p_lt = st.selectbox("Load Type", sorted(df['load_type'].dropna().unique()))
    p_ct = st.selectbox("Customer Type", sorted(df['customer_type'].dropna().unique()))

if st.button("⚡ Predict Expected Revenue", type="primary"):
    sample = df.iloc[[0]].copy()
    sample['typical_distance_miles'] = p_dist
    sample['weight_lbs']             = p_weight
    sample['base_rate_per_mile']     = p_rate
    sample['pieces']                 = p_pieces
    sample['load_type']              = p_lt
    sample['customer_type']          = p_ct
    sample['weight_per_piece']       = p_weight / p_pieces
    sample['expected_rate_revenue']  = p_dist * p_rate
    feat_cols = list(model.named_steps['prep'].feature_names_in_)
    pred = model.predict(sample[feat_cols])[0]
    c1, c2 = st.columns(2)
    c1.success(f"**Predicted Revenue: ${pred:,.0f}**")
    c2.info(f"Rate × Distance estimate: ${p_dist * p_rate:,.0f}")

st.divider()

# ── Monitoring ────────────────────────────────────────────────
st.header("📈 Model & Data Monitoring")
m1, m2, m3, m4 = st.columns(4)

actual   = preds['actual_revenue']
pred_imp = preds['pred_improved']
live_mae = np.mean(np.abs(actual - pred_imp))
live_r2  = 1 - np.sum((actual-pred_imp)**2) / np.sum((actual-actual.mean())**2)

m1.metric("Model MAE (test set)",  f"${live_mae:,.0f}")
m2.metric("Model R² (test set)",   f"{live_r2:.4f}")
m3.metric("Distance Drift (mean shift, mi)",
           f"{abs(filtered['typical_distance_miles'].mean()-df['typical_distance_miles'].mean()):.1f}")
m4.metric("Revenue Drift (mean shift, $)",
           f"${abs(filtered['revenue'].mean()-df['revenue'].mean()):,.0f}")

st.subheader("Data Quality — Profitability Confidence")
conf = df['profitability_confidence'].value_counts().reset_index()
conf.columns = ['Status','Count']
color_map = {
    'Reliable'                               : '#2e7d32',
    'Low margin — monitor'                   : '#f9a825',
    'Unknown — no fuel data'                 : '#78909c',
    'Unreliable — misassigned fuel records'  : '#c62828'
}
fig_m = px.bar(conf, x='Status', y='Count', color='Status',
                color_discrete_map=color_map,
                title="Profitability Confidence Status Across All Loads")
st.plotly_chart(fig_m, use_container_width=True)
