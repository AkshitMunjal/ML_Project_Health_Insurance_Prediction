import pandas as pd
from joblib import load

model_rest = load("artifacts/model_rest.joblib")
scaler_rest = load("artifacts/scaler_rest.joblib")

model_young = load("artifacts/model_young.joblib")
scaler_young = load("artifacts/scaler_young.joblib")

risk_scores = {
    'diabetes': 6,
    'heart disease': 8,
    'high blood pressure': 6,
    'thyroid': 5,
    'no disease': 0,
    'none': 0
}


def calculate_risk_score(medical_history):
    if pd.isna(medical_history):
        return 0  # Default if no history provided

    diseases = medical_history.lower().split(" & ")

    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)

    max_score = max(risk_scores.values())
    min_score = min(risk_scores.values())

    normalized_score = (total_risk_score - min_score) / (max_score - min_score) if max_score > min_score else 0
    return normalized_score


def handle_scaling(age, df):
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object.get('cols_to_scale', [])
    scaler = scaler_object.get('scaler')

    if scaler and cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df  # Return modified df

import pandas as pd

def preprocess_input(input_dict):
    expected_cols = ['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
                     'genetical_risk', 'normalized_risk_score', 'gender_Male',
                     'region_Northwest', 'region_Southeast', 'region_Southwest',
                     'marital_status_Unmarried', 'bmi_category_Obesity',
                     'bmi_category_Overweight', 'bmi_category_Underweight',
                     'smoking_status_Occasional', 'smoking_status_Regular',
                     'employment_status_Salaried', 'employment_status_Self-Employed']

    insurance_plan_encoding = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}

    df = pd.DataFrame(0, columns=expected_cols, index=[0])

    for key, val in input_dict.items():
        if key == 'Age':
            df['age'] = val
        elif key == 'Number of Dependents':
            df['number_of_dependants'] = val
        elif key == 'Income (Lakhs)':
            df['income_lakhs'] = val
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(val, 0)
        elif key == 'Genetical Risk':
            df['genetical_risk'] = val
        elif key == 'Gender' and val == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region' and f'region_{val}' in df.columns:
            df[f'region_{val}'] = 1
        elif key == 'Marital Status' and val == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category' and f'bmi_category_{val}' in df.columns:
            df[f'bmi_category_{val}'] = 1
        elif key == 'Employment Status':
            employment_key = f'employment_status_{val.replace("-", "_")}'
            if employment_key in df.columns:
                df[employment_key] = 1
        elif key == 'Medical History':
            df['normalized_risk_score'] = calculate_risk_score(val)

    # **Step 1: Add 'income_level' before scaling**
    df['income_level'] = df['income_lakhs']  # Using 'income_lakhs' as a placeholder

    # **Step 2: Perform scaling**
    df = handle_scaling(input_dict['Age'], df)

    # **Step 3: Drop 'income_level' after scaling**
    df.drop(columns=['income_level'], inplace=True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)

    # Choose the appropriate model based on age
    model = model_young if input_dict["Age"] <= 25 else model_rest

    # Make a prediction
    predicted_premium = model.predict(input_df)[0]  # Assuming model.predict() outputs an array

    return round(predicted_premium, 2)  # Return rounded premium
