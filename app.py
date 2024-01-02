# app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained machine learning model
with open('gbrt_model.joblib', 'rb') as file:
    model = joblib.load(file)

# Streamlit app code
st.title('Premium Mobility Battery Prediction')

# Create input components (e.g., sliders, text input, etc.)
feature1 = st.slider('Feature 1', 0.0, 10.0, 5.0)
feature2 = st.slider('Feature 2', 0.0, 10.0, 5.0)

# Make predictions based on user input
input_data = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0]])
prediction = model.predict(input_data)

# Display the prediction
st.write('Prediction:', prediction[0])
