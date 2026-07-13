import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set modern, wide page layout
st.set_page_config(page_title="SiliconPulse AI", layout="wide", initial_sidebar_state="collapsed")

# Inject minimal, dark-themed CSS styling matching the volt.fm inspiration
st.markdown("""
    <style>
        body { color: #ffffff; }
        .metric-card {
            background-color: #111111;
            border: 1px solid #222222;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .status-title {
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1.5px;
            color: #888888;
            font-weight: bold;
        }
        .price-display {
            font-size: 3rem;
            font-weight: 800;
            color: #00ff66;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Load machine learning pipeline components safely from the root directory
@st.cache_resource
def load_assets():
    model = joblib.load('macro_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    prod_features = joblib.load('product_features.pkl')
    defaults = joblib.load('macro_defaults.pkl')
    return model, tfidf, prod_features, defaults

try:
    macro_model, tfidf, product_features, macro_defaults = load_assets()
except Exception as e:
    st.error("Model assets not found in the root directory. Please check your GitHub repository layout.")
    st.stop()

# --- HEADER INTERFACE ---
st.title("⚡ SiliconPulse AI")
st.markdown("### Geopolitical Supply-Chain Forecasting Engine")
st.markdown("---")

# --- MAIN LAYOUT DIVISION ---
col_input, col_display = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("<p class='status-title'>Simulation Parameters</p>", unsafe_allow_html=True)
    
    # Category Selection drop-down
    selected_prod = st.selectbox("Target Hardware Architecture", macro_defaults['unique_products'])
    
    # Financial Inputs via clean split rows
    c1, c2 = st.columns(2)
    with c1:
        input_price = st.number_input("Current Spot Market Price ($)", min_value=0.0, value=500.0, step=10.0)
    with c2:
        severity = st.slider("Geopolitical Disruption Severity Index", min_value=0, max_value=10, value=5)
        
    # Unstructured Trade Policy Text Scenario Simulation Box
    custom_scenario = st.text_area(
        "Simulated Policy, Sanction, or Export Regulation Text",
        value="US Department of Commerce imposes sweeping lithography restrictions targeting advanced foundry equipment nodes.",
        height=120
    )

# --- ENGINE CALCULATION MATRIX ---
# Text Vectorization transformation processing
text_vector = tfidf.transform([custom_scenario]).toarray()

# Construct One-Hot encoded categoric array mapping matches perfectly 
product_vector = np.zeros((1, len(product_features)))
if selected_prod in product_features:
    prod_idx = product_features.index(selected_prod)
    product_vector[0, prod_idx] = 1

# Reconstruct relative mock time-series pricing dynamics features
lag2_mock = input_price * 0.98
rolling_mean_mock = (input_price + lag2_mock) / 2

# Compile final row vectors seamlessly with loaded defaults structural arrays
numeric_vector = np.array([[
    severity, input_price, lag2_mock, rolling_mean_mock,
    macro_defaults['global_wafer_capacity'],
    macro_defaults['global_ai_shipments'],
    macro_defaults['global_ai_revenue_m'],
    macro_defaults['total_industry_capex'],
    macro_defaults['avg_industry_margin']
]])

final_features_matrix = np.hstack([text_vector, product_vector, numeric_vector])
predicted_output = macro_model.predict(final_features_matrix)[0]

# --- LIVE CARD METRICS DISPLAY PANEL ---
with col_display:
    st.markdown("<p class='status-title'>Simulated Forecasting Results</p>", unsafe_allow_html=True)
    
    # Main Output Display Card Container block inspired by volt.fm
    st.markdown(f"""
        <div class="metric-card">
            <span class="status-title">Forecasted Next-Month Price</span>
            <div class="price-display">${predicted_output:,.2f}</div>
            <p style="color:#888888; font-size:0.9rem; margin:0;">
                Target Component Portfolio: <b>{selected_prod}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Explanatory Analytical context metrics panel grid blocks
    st.markdown("<p class='status-title'>Active Macroeconomic Baseline Signals</p>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Global Fabrication Cap", f"{macro_defaults['global_wafer_capacity']/1e6:.1f}M Wafers")
    with m2:
        st.metric("Annual Industry Capex", f"${macro_defaults['total_industry_capex']:.1f}B")
    with m3:
        st.metric("AI Market Run-Rate", f"${macro_defaults['global_ai_revenue_m']/1e3:.1f}B")
  
