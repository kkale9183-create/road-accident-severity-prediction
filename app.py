# Final app.py using Dropdown (No Number Input)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Road Accident Severity Prediction System")
st.write("Predict accident severity using Machine Learning")

# Load Dataset
df = pd.read_csv("indian_road_accident_severity_10000.csv")

# Remove missing values
df = df.dropna()

# Convert object columns to category codes
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# Target column
target_column = "Accident_Severity"

# Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
X = X.astype(float)

y = pd.to_numeric(y, errors="coerce").fillna(0)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("Enter Accident Details")

input_values = []

# Dropdown input instead of number input
for col in X.columns:
    unique_values = sorted(list(df[col].dropna().unique()))
    
    selected_value = st.selectbox(
        f"Select {col}",
        unique_values
    )
    
    input_values.append(float(selected_value))

# Create input dataframe
input_data = pd.DataFrame(
    [input_values],
    columns=X.columns
)

# Prediction
if st.button("Predict Severity"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
