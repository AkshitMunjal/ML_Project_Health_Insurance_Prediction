
## Overview

This is a Streamlit-based web application that predicts health insurance premiums based on user inputs such as age, income, medical history, and lifestyle choices. The application leverages machine learning models trained on historical insurance data.
## Features

📌 User-friendly UI with three-section form layout

📊 Two trained models:

Model Young (XGBoost): 98% accuracy

Model Rest (Linear Regression): 97% accuracy

🏥 Supports various health & demographic factors such as BMI, medical history, and employment status

📉 Scales input data using pre-trained scalers

🧠 Predicts insurance premiums in real-time


## How It Works

1. The user provides personal, lifestyle, and health details

2. The system preprocesses the input, encodes categorical values, and scales numerical data

3. The appropriate machine learning model (XGBoost for young users, Linear Regression for others) is selected based on age

4. The insurance premium is predicted and displayed
## Installation


pip install -r requirements.txt

streamlit run app.py
## Usage

1. Run the app using streamlit run app.py

2. Fill in the required details

3. Click Predict Insurance Premium

4. View the estimated premium 🎯
## Screenshots

![Health Insurance Prediction App](https://github.com/AkshitMunjal/ML_Project_Health_Insurance_Prediction/blob/7c39d9a057b4965a708173fc10a378f8068212d9/streamlit_app_health_prediction.png?raw=true)

## Technologies

This project is created with below technologies/tools/resources:

✅Python: 3.7

✅Machine Learning

✅Jupyter Notebook

✅Git

✅Streamlit

✅Pycharm
## Let’s Connect! 🤝

I’m always excited to collaborate on data-driven projects or join innovative teams 🚀. Let’s build something extraordinary together! 🌟

📧Email: akshitmunjal479@gmail.com

🔗LinkedIn: www.linkedin.com/in/akshit-munjal-81851b188

🌐Portfolio:https://padlet.com/akshitmunjal479/projects-portfolio-ozbg80dvuk6fl9gg
## Web Application Link

https://ml-project-health-insurance-prediction.streamlit.app/