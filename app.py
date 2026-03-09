import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

"""Data Loading, Model Training and Streamlit UI

This app predicts a continuous `outcome_score` (higher is better)
from powerplay features using linear regression and displays
R² and RMSE on a held-out test set.
"""

np.random.seed(42)
df = pd.DataFrame({
    'powerplay_score': np.random.randint(20, 120, 1000),
    'powerplay_wickets': np.random.randint(0, 6, 1000),
})

# Create a continuous outcome correlated with the features
df['outcome_score'] = (0.6 * df['powerplay_score'] - 5 * df['powerplay_wickets']
                       + np.random.normal(0, 10, len(df)))
df['outcome_score'] = df['outcome_score'].clip(lower=0)

X = df[['powerplay_score', 'powerplay_wickets']]
y = df['outcome_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
y_test_pred = model.predict(X_test)
r2 = r2_score(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)

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
    st.write(f"**Model R²:** {r2:.3f}  —  **RMSE:** {rmse:.3f}")
    st.write("**Model intercept:** {:.3f}".format(float(model.intercept_)))
    st.write("**Coefficients:** powerplay_score = {:.3f}, powerplay_wickets = {:.3f}".format(
        float(model.coef_[0]), float(model.coef_[1])
    ))
