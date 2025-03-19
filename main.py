import streamlit as st
from PIL import Image
import os
import pandas as pd
from joblib import load
from prediction_helper import predict
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", UserWarning)  # Suppress XGBoost warnings
warnings.simplefilter("ignore", InconsistentVersionWarning)  # Suppress scikit-learn warnings
warnings.simplefilter("ignore")  # Suppress all warnings

# Load Models and Scalers
model_rest = load("artifacts/model_rest.joblib")
scaler_rest = load("artifacts/scaler_rest.joblib")

model_young = load("artifacts/model_young.joblib")
scaler_young = load("artifacts/scaler_young.joblib")

# Set page configuration
st.set_page_config(page_title="Health Insurance Predictor", layout="wide")

# Center align everything
st.markdown("<style>div.block-container {text-align: center;}</style>", unsafe_allow_html=True)

# Load Image (Ensure the path is correct)
image_path = "artifacts/health_image.jpg"  # Corrected path
if os.path.exists(image_path):
    image = Image.open(image_path)
    new_width = 1400  # Adjust width as needed
    image = image.resize((new_width, image.height))  # Keep height unchanged
    st.image(image, use_container_width=False)  # Updated parameter
else:
    st.warning("⚠️ Image not found. Please check the file path.")

# Title of the app (Below the image)
st.markdown("<h1 style='text-align: center;'>Health Insurance Predictor App 💡</h1>", unsafe_allow_html=True)

# Remove unnecessary white space
st.markdown("---")  # Adds a separator instead of a blank space

# Layout: Three-column format
col1, col2, col3 = st.columns(3)

# Personal Details Section
with col1:
    st.header("Personal Details")
    age = st.number_input("😀 Age", min_value=18, max_value=100, value=18)
    insurance_plan = st.selectbox("🏥 Insurance Plan", ["Silver", "Gold", "Platinum"])
    gender = st.selectbox("⚧️ Gender", ["Male", "Female"])

# Lifestyle & Work Section
with col2:
    st.header("Lifestyle & Work")
    dependents = st.number_input("👨‍👩‍👦 Number of Dependents", min_value=0, value=0)
    employment_status = st.selectbox("💼 Employment Status", ["Employed", "Self-Employed", "Unemployed"])
    marital_status = st.selectbox("💍 Marital Status", ["Married", "Unmarried"])

# Health & Region Section
with col3:
    st.header("Health & Region")
    genetical_risk = st.slider("🧬 Genetical Risk", 0, 5, 0)
    income = st.number_input("💰 Income (Lakhs)", min_value=0.1, value=1.0)
    region = st.selectbox("📍 Region", ["Urban", "Rural", "Semi-Urban"])

# NEW: BMI Category Selection
bmi_category = st.selectbox("⚖️ BMI Category", ["Normal","Overweight", "Underweight", "Obesity"])

# NEW: Medical History Selection
df = pd.DataFrame({'medical_history': ['High blood pressure', 'No Disease',
       'Diabetes & High blood pressure', 'Diabetes & Heart disease',
       'Diabetes', 'Diabetes & Thyroid', 'Heart disease', 'Thyroid',
       'High blood pressure & Heart disease']})
medical_history = st.selectbox("🩺 Medical History", df['medical_history'].unique())

# Final separator
st.markdown("---")

# Collect all inputs into a dictionary
input_data = {
    "Age": age,
    "Insurance Plan": insurance_plan,
    "Gender": gender,
    "Number of Dependents": dependents,
    "Employment Status": employment_status,
    "Marital Status": marital_status,
    "Genetical Risk": genetical_risk,
    "Income (Lakhs)": income,
    "Region": region,
    "BMI Category": bmi_category,  # Added BMI category
    "Medical History": medical_history  # Added Medical History
}


if st.button("🔍 Predict Insurance Premium"):
    predicted_premium = predict(input_data)
    st.success(f"💰 Estimated Insurance Premium: ₹{predicted_premium}")

