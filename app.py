import pandas as pd
import numpy as np
import requests  # <-- NEW: This is the tool that talks to the internet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

"""Data Loading, Model Training and Streamlit UI

This app predicts a continuous outcome_score (higher is better)
from powerplay features using linear regression.
"""

# =============================================================================
# 1. LIVE DATA FETCHING
# =============================================================================

# This is a placeholder URL. You will replace this with a real Cricket API link later!
API_URL = "https://api.example-sports-data.com/latest_ipl_powerplays"

try:
    # 1. Ask the internet for the data
    response = requests.get(API_URL)
    
    # 2. Read the response as a dictionary (JSON format)
    live_data = response.json()
    
    # 3. Turn it into the exact same Pandas table you had before!
    df = pd.DataFrame(live_data)

except:
    # If the internet is down or the URL is fake, we show a friendly message and stop.
    st.error("Could not connect to the live data source. Please check your API URL.")
    st.stop()


# =============================================================================
# 2. MODEL TRAINING (Your exact operations)
# =============================================================================

X = df[['powerplay_score', 'powerplay_wickets']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


# =============================================================================
# 3. STREAMLIT UI (Your exact code)
# =============================================================================
st.title("🏏 IPL Powerplay Outcome Score Predictor (Linear Regression)")
st.write("Predict a continuous outcome score from powerplay performance")
st.markdown("---")

# SLIDERS
powerplay_score = st.slider(
    "Select Powerplay Score",
    min_value=0,
    max_value=120,
    value=50
)

powerplay_wickets = st.slider(
    "Select Powerplay Wickets",
    min_value=0,
    max_value=6,
    value=1
)

if st.button("Predict Score"):
    X_input = pd.DataFrame([[powerplay_score, powerplay_wickets]],
                           columns=['powerplay_score', 'powerplay_wickets'])
    pred = model.predict(X_input)[0]

    st.metric(label="Predicted Outcome Score", value=f"{pred:.2f}")
