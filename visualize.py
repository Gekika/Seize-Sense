# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load('best_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Title of the Streamlit app
st.title("Welcome to TempLapse")

# Function to preprocess input data
def preprocess_input(data):
    # Add preprocessing steps if any were applied during training
    return data

# Function to make predictions
def make_prediction(model, input_data):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction

# Sidebar for user input
st.sidebar.header('User Input Features')

# Dynamically create input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

# Convert user input to DataFrame
input_df = pd.DataFrame([input_data])

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Make predictions
if st.button('Predict'):
    prediction = make_prediction(model, input_df)
    st.subheader('Prediction')
    st.write(prediction)
