import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data
@st.cache_data
def load_data():
    train_data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
    print(train_data)
    return train_data

# Preprocess data for model
@st.cache_data
def preprocess_data(data):
    # Fill missing values in numeric columns with their median
    numeric_columns = data.select_dtypes(include=["number"]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Fill missing values in non-numeric columns with their mode
    non_numeric_columns = data.select_dtypes(exclude=["number"]).columns
    for col in non_numeric_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Map categorical values to numeric equivalents
    to_numeric = {
        'Male': 0, 'Female': 1,
        'Yes': 0, 'No': 1,
        'Graduate': 0, 'Not Graduate': 1,
        'Urban': 2, 'Semiurban': 1, 'Rural': 0,
        'Y': 1, 'N': 0,
        '3+': 3
    }
    data = data.applymap(lambda label: to_numeric.get(label) if label in to_numeric else label)

    # Convert 'Dependents' column to numeric if present
    if 'Dependents' in data.columns:
        data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce')
    
    # Replace remaining missing values with 0 (fallback)
    data.fillna(0, inplace=True)

    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['Loan_Status', 'Loan_ID'])
    y = data['Loan_Status']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Visualization page
def visualization_page(data):
    st.title("Data Visualization")

    st.write("### Dataset Overview")
    st.write(data.head())

    # Gender Distribution
    st.write("#### Gender Distribution")
    gender_count = data['Gender'].value_counts()
    fig, ax = plt.subplots()
    gender_count.plot(kind='bar', color=['blue', 'pink'], ax=ax)
    st.pyplot(fig)

    # Loan Status Distribution
    st.write("#### Loan Status Distribution")
    loan_status_count = data['Loan_Status'].value_counts()
    fig, ax = plt.subplots()
    loan_status_count.plot(kind='bar', color=['green', 'red'], ax=ax)
    st.pyplot(fig)

    # Income vs Loan Amount
    st.write("#### Income vs Loan Amount")
    fig, ax = plt.subplots()
    sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=data, ax=ax)
    st.pyplot(fig)

# Prediction page
def prediction_page(model):
    st.title("Loan Prediction")

    st.write("### Enter Details to Predict Loan Status")

    # Input fields
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    married = st.selectbox("Married", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=3, value=0)
    education = st.selectbox("Education", [1, 2], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")
    self_employed = st.selectbox("Self Employed", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good" if x == 1 else "Bad")
    property_area = st.selectbox("Property Area", [1, 2, 3], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x-1])

    # Prediction
    input_data = np.array([[gender, married, dependents, education, self_employed,
                             applicant_income, coapplicant_income, loan_amount,
                             loan_amount_term, credit_history, property_area]])
    prediction = model.predict(input_data)

    st.write("### Prediction Result")
    st.write("Loan Approved" if prediction[0] == 1 else "Loan Not Approved")

# Main app
def main():
    st.sidebar.title("Loan Prediction Dashboard")
    pages = ["Data Visualization", "Prediction"]
    choice = st.sidebar.radio("Navigation", pages)

    data = load_data()
    preprocessed_data = preprocess_data(data)
    model = train_model(preprocessed_data)

    if choice == "Data Visualization":
        visualization_page(preprocessed_data)
    elif choice == "Prediction":
        prediction_page(model)

if __name__ == "__main__":
    main()