import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

"""###Data Loading"""

# Generate sample IPL powerplay dataset
np.random.seed(42)
df = pd.DataFrame({
    'powerplay_score': np.random.randint(20, 120, 1000),
    'powerplay_wickets': np.random.randint(0, 6, 1000),
    'result': np.random.randint(0, 2, 1000)
})

X = df[['powerplay_score','powerplay_wickets']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- STREAMLIT UI ---------------- #

st.title("🏏 IPL Powerplay Win Predictor")

st.write("Predict match result based on powerplay performance")

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

if st.button("Predict Result"):

    X_test = pd.DataFrame([[powerplay_score, powerplay_wickets]], columns=['powerplay_score', 'powerplay_wickets'])

    y_pred = model.predict(X_test)

    if y_pred[0] == 1:
        st.success("🏆 Prediction: Team will WIN")
    else:
        st.error("❌ Prediction: Team will LOSE")
