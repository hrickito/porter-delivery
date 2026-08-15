import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and the scaler
@st.cache_resource
def load_assets():
    model = load_model('delivery_model.keras') 
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("Porter Delivery Time Predictor")
st.write("Enter the order details below to predict the estimated delivery time in minutes.")

# Create input fields for the features required by our model
col1, col2 = st.columns(2)

with col1:
    total_items = st.number_input("Total Items", min_value=1, step=1)
    subtotal = st.number_input("Subtotal (Price)", min_value=0.0)
    num_distinct_items = st.number_input("Number of Distinct Items", min_value=1, step=1)
    min_item_price = st.number_input("Minimum Item Price", min_value=0.0)
    max_item_price = st.number_input("Maximum Item Price", min_value=0.0)

with col2:
    total_onshift_partners = st.number_input("Total On-Shift Partners", min_value=0, step=1)
    total_busy_partners = st.number_input("Total Busy Partners", min_value=0, step=1)
    total_outstanding_orders = st.number_input("Total Outstanding Orders", min_value=0, step=1)

# Prediction button
if st.button("Predict Delivery Time"):
    # Group inputs into a dataframe matching the exact order of your training features
    input_data = pd.DataFrame([[
        total_items,
        subtotal, 
        num_distinct_items, 
        min_item_price, 
        max_item_price, 
        total_onshift_partners, 
        total_busy_partners, 
        total_outstanding_orders
    ]], columns=[
        'total_items',
        'subtotal', 
        'num_distinct_items', 
        'min_item_price', 
        'max_item_price', 
        'total_onshift_partners', 
        'total_busy_partners', 
        'total_outstanding_orders'
    ])
    
    # Scale the input data using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Make the prediction
    prediction = model.predict(input_scaled)
    
    # Display the result
    st.success(f"Estimated Delivery Time: {prediction[0][0]:.2f} minutes")
