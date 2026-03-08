"""
IPL Powerplay-to-Final-Score Prediction App
Uses Multiple Linear Regression to predict final score from powerplay runs and wickets.
"""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# =============================================================================
# 1. DATA CLEANING
# =============================================================================

df = pd.read_csv("ipl_data.csv")

# Drop rows with missing or blank values
df = df.dropna()

# Drop exact duplicate rows
df = df.drop_duplicates()

# Filter impossible outliers using domain knowledge
# Powerplay is 6 overs - 180 runs would require 30 runs/over (impossible)
df = df[df["Powerplay_Runs"] <= 150]

# A team can only lose 10 wickets in cricket
df = df[df["Powerplay_Wickets"] <= 10]

# Edge case: if cleaning removed all rows, we cannot train
if df.empty:
    st.error("No valid data remaining after cleaning. Cannot train the model.")
    st.stop()

# =============================================================================
# 2. MODEL TRAINING
# =============================================================================

X = df[["Powerplay_Runs", "Powerplay_Wickets"]]
y = df["Final_Score"]

model = LinearRegression()
model.fit(X, y)

# =============================================================================
# 3. STREAMLIT WEB INTERFACE
# =============================================================================

st.set_page_config(page_title="IPL Final Score Predictor", layout="centered")
st.title("IPL Final Score Predictor")
st.markdown("Predict the final innings score from powerplay performance.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    powerplay_runs = st.slider("Powerplay Runs", min_value=0, max_value=100, value=50)
with col2:
    powerplay_wickets = st.slider("Powerplay Wickets", min_value=0, max_value=10, value=1)

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame({
        "Powerplay_Runs": [powerplay_runs],
        "Powerplay_Wickets": [powerplay_wickets]
    })
    prediction = model.predict(input_data)[0]
    st.metric(label="Predicted Final Score", value=f"{int(round(prediction))}")

st.divider()
st.caption(f"Model trained on {len(df)} cleaned samples.")


# =============================================================================
# DEPLOYMENT GUIDE (3 Steps for Streamlit Community Cloud)
# =============================================================================
#
# Step 1: Put Your Code on GitHub
#   - Create a free GitHub account if you don't have one
#   - Create a new repository and upload app.py, ipl_data.csv, and requirements.txt
#   - Make sure ipl_data.csv is in the same folder as app.py
#
# Step 2: Go to Streamlit Community Cloud
#   - Visit share.streamlit.io
#   - Sign in with your GitHub account
#   - Click "New app"
#
# Step 3: Deploy Your App
#   - Select your repository and branch (usually "main")
#   - Set the main file to app.py
#   - Click "Deploy" and wait a few minutes
#   - Your app will be live at a URL like https://your-app-name.streamlit.app
#
