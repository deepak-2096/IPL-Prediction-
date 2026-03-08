"""
IPL Powerplay-to-Final-Score Prediction App
Uses Multiple Linear Regression to predict final score from powerplay runs and wickets.
"""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0d0d0d;
        color: #f0ede6;
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        letter-spacing: 0.05em;
        line-height: 1;
        color: #f0ede6;
        margin: 0;
    }

    .hero-accent {
        color: #f97316;
    }

    .hero-sub {
        font-size: 0.95rem;
        color: #888;
        margin-top: 0.4rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .divider {
        border: none;
        border-top: 1px solid #222;
        margin: 2rem 0;
    }

    .result-box {
        background: linear-gradient(135deg, #1a1a1a, #111);
        border: 1px solid #f97316;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #888;
        margin-bottom: 0.5rem;
    }

    .result-score {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        color: #f97316;
        line-height: 1;
    }

    .result-range {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.5rem;
    }

    .stat-pill {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        color: #aaa;
        margin: 0.2rem;
    }

    /* Streamlit widget overrides */
    .stSlider > div > div > div {
        background: #f97316 !important;
    }

    label {
        color: #aaa !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .stButton > button {
        background: #f97316;
        color: #0d0d0d;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.3rem;
        letter-spacing: 0.08em;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        background: #ea6a0a;
        color: #0d0d0d;
        border: none;
    }

    .warning-box {
        background: #1a1209;
        border: 1px solid #92400e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #fbbf24;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & MODEL TRAINING (cached)
# =============================================================================

@st.cache_resource
def load_and_train():
    df = pd.read_csv("ipl_data.csv")
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df["Powerplay_Runs"] <= 150]
    df = df[df["Powerplay_Wickets"] <= 10]

    if df.empty:
        return None, None, None

    X = df[["Powerplay_Runs", "Powerplay_Wickets"]]
    y = df["Final_Score"]

    model = LinearRegression()
    model.fit(X, y)

    return model, df, len(df)

model, df, n_samples = load_and_train()

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<p class="hero-title">IPL <span class="hero-accent">Score</span><br>Predictor</p>
<p class="hero-sub">Powerplay → Final Score · Multiple Linear Regression</p>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =============================================================================
# GUARD: model failed to train
# =============================================================================

if model is None:
    st.markdown('<div class="warning-box">⚠️ No valid data remaining after cleaning. Cannot train model.</div>', unsafe_allow_html=True)
    st.stop()

# =============================================================================
# INPUTS
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Powerplay Runs**")
    powerplay_runs = st.slider(
        "Powerplay Runs",
        min_value=0,
        max_value=150,
        value=60,
        step=1,
        label_visibility="collapsed"
    )
    st.markdown(f'<span class="stat-pill">🏏 {powerplay_runs} runs</span>', unsafe_allow_html=True)

with col2:
    st.markdown("**Wickets Lost**")
    powerplay_wickets = st.slider(
        "Powerplay Wickets",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        label_visibility="collapsed"
    )
    st.markdown(f'<span class="stat-pill">🎯 {powerplay_wickets} wickets</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PREDICT
# =============================================================================

if st.button("Predict Final Score"):
    input_df = pd.DataFrame({
        "Powerplay_Runs": [powerplay_runs],
        "Powerplay_Wickets": [powerplay_wickets]
    })
    prediction = model.predict(input_df)[0]
    low, high = int(prediction * 0.93), int(prediction * 1.07)

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Predicted Final Score</div>
        <div class="result-score">{int(prediction)}</div>
        <div class="result-range">Estimated range: {low} – {high}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f'<p style="color:#444; font-size:0.75rem; text-align:center;">Model trained on {n_samples:,} cleaned innings · Linear Regression</p>', unsafe_allow_html=True)
