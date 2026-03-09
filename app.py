import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

"""Data Loading, Model Training and Streamlit UI

This app predicts a continuous `outcome_score` (higher is better)
from powerplay features using linear regression and displays
R² and RMSE on a held-out test set.
"""

df = pd.read_csv('ipl_powerplay_finalscore_dataset_1000.csv')

X = df[['powerplay_score', 'powerplay_wickets']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


# ---------------- STREAMLIT UI ---------------- #
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
