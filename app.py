import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.jlb")
scaler = joblib.load("scaler.jlb")

st.title("Insurance Prediction")

# Input
age_input = st.number_input(
    "Enter your age:",
    min_value=0,
    max_value=120,
    step=1
)

# Prediction
if st.button("Predict"):
    age_array = np.array([[age_input]])   # shape (1, 1)
    age_scaled = scaler.transform(age_array)
    prediction = model.predict(age_scaled)

    if prediction[0] == 0:
        st.success("❌ No Insurance")
    else:
        st.success("✅ Has Insurance")
