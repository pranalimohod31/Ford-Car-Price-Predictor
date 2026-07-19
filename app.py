# Import libraries
import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing files
model = joblib.load("LR_model (1).pkl")
scaler = joblib.load("scaler (1).pkl")
encoded_columns = joblib.load("columns (1).pkl")

# Page configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

st.title(" Ford Car Price Predictor")

# Numerical Inputs
year = st.number_input(
    "Manufacturing Year",
    min_value=1996,
    max_value=2024,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=50000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=600,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.0,
    max_value=5.0,
    value=1.5
)

# Categorical Inputs
transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

car_model = st.text_input("Enter Car Model")

# Prediction Button
if st.button("Predict Price"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    # One-Hot Encoding
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(columns=encoded_columns, fill_value=0)

    # Debug (optional)
    print(encoded_columns)
    print(input_data.columns)

    # Scale numerical columns
    numerical_columns = ["year", "mileage", "tax", "mpg", "engineSize"]
    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    # Prediction
    predicted_price = model.predict(input_data)

    # Show result
    st.success(f"Predicted Car Price: £{predicted_price[0]:,.2f}")