python
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL FILES ----------------
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_files():
    model = joblib.load(BASE_DIR / "LR_model (1).pkl")
    scaler = joblib.load(BASE_DIR / "scaler (1).pkl")
    columns = joblib.load(BASE_DIR / "columns (1).pkl")
    return model, scaler, columns

try:
    model, scaler, encoded_columns = load_files()
except Exception as e:
    st.error("Unable to load the model files. Check that the model, scaler, and columns files exist in your repository.")
    st.code(str(e))
    st.stop()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #102d4a, #071426);
    color: #f4f8ff;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    color: #f4f8ff !important;
}
.hero {
    text-align: center;
    padding: 24px 10px 30px 10px;
}
.hero-icon {
    font-size: 54px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}
.subtitle {
    color: #b9cce0 !important;
    font-size: 17px;
}
.form-card {
    background: #ffffff;
    color: #10243a;
    padding: 25px 30px;
    border-radius: 20px;
    margin: 12px 0 25px 0;
}
.form-card h3, .form-card p {
    color: #10243a !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label {
    color: #17314b !important;
    font-weight: 600;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    color: #10243a !important;
    background: #f5f8fc !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #f5f8fc !important;
    color: #10243a !important;
}
div.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: 15px;
    color: white;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #087cf0, #00b9d6);
}
div.stButton > button:hover {
    color: white;
    border: 1px solid #ffffff;
}
.result-card {
    background: linear-gradient(110deg, #dff7ee, #e5f4ff);
    color: #10243a;
    padding: 24px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}
.result-card h2, .result-card p {
    color: #10243a !important;
}
.feature-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 16px;
    padding: 20px 12px;
    text-align: center;
    min-height: 125px;
}
.feature-card h3 {
    font-size: 18px;
}
.feature-card p {
    color: #c4d4e7 !important;
    font-size: 14px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🚘</div>
    <h1>Ford Car Price Predictor</h1>
    <p class="subtitle">
        Estimate your Ford's market value with Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
st.markdown("""
<div class="form-card">
    <h3>🚗 Enter Your Car Details</h3>
    <p>Fill in the details below to estimate your car's price.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    car_model = st.selectbox(
        "Car Model",
        [
            "Fiesta", "Focus", "Puma", "Kuga", "EcoSport",
            "C-MAX", "Mondeo", "Ka+", "Ka", "Galaxy",
            "B-MAX", "Grand C-MAX", "S-MAX", "Edge",
            "Tourneo Connect", "Tourneo Custom", "Mustang",
            "Ranger", "Streetka", "Fusion", "Transit Tourneo"
        ]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic", "Semi-Auto"]
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1996,
        max_value=2026,
        value=2018,
        step=1
    )

with col2:
    mileage = st.number_input(
        "Mileage (miles)",
        min_value=0,
        max_value=300000,
        value=50000,
        step=1000
    )

    tax = st.number_input(
        "Road Tax (£)",
        min_value=0,
        max_value=600,
        value=150,
        step=10
    )

    mpg = st.number_input(
        "Fuel Economy (MPG)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.0,
        max_value=7.0,
        value=1.5,
        step=0.1
    )

st.write("")

# ---------------- PREDICTION ----------------
if st.button("✨ Predict Car Price", use_container_width=True):

    try:
        # Create input data using the same feature names as the dataset.
        input_data = pd.DataFrame({
            "model": [car_model],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuel_type],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engine_size]
        })

        # One-hot encode categorical inputs.
        input_data = pd.get_dummies(input_data)

        # Match the model's expected feature columns and order.
        input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Scale numerical features.
        numerical_columns = [
            "year", "mileage", "tax", "mpg", "engineSize"
        ]

        available_numerical = [
            col for col in numerical_columns
            if col in input_data.columns
        ]

        input_data[available_numerical] = scaler.transform(
            input_data[available_numerical]
        )

        # Make prediction.
        predicted_price = float(model.predict(input_data)[0])

        if predicted_price < 0:
            st.warning(
                "The model returned a negative estimate. "
                "Try different car details."
            )
        else:
            st.markdown(f"""
            <div class="result-card">
                <p>💷 Estimated Car Price</p>
                <h2>£{predicted_price:,.2f}</h2>
                <p>Estimated using your saved Linear Regression model.</p>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "This is a machine-learning estimate, not a guaranteed "
                "selling price. Actual prices may vary."
            )

    except Exception as e:
        st.error("Prediction failed. Please check the model's expected features.")
        st.exception(e)

# ---------------- FEATURES ----------------
st.write("")
st.markdown("### 💡 Why Use Our Predictor?")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI Powered</h3>
        <p>Uses a trained machine-learning model.</p>
    </div>
    """, unsafe_allow_html=True)

with feature2:
    st.markdown("""
    <div class="feature-card">
        <h3>⚡ Instant Estimate</h3>
        <p>Get a price estimate after entering your details.</p>
    </div>
    """, unsafe_allow_html=True)

with feature3:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Smart Analysis</h3>
        <p>Considers multiple car characteristics.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.caption("Ford Car Price Predictor | Machine Learning Project")
