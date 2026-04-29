# FINAL WORKING app.py (Fix prediction error)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Road Accident Severity Prediction System")
st.write("Predict accident severity using Machine Learning")

# Load Dataset
df = pd.read_csv("indian_road_accident_severity_10000.csv")

# Remove missing values
df = df.dropna()

# Convert all object columns into numeric codes
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# Target Column
target_column = "Accident_Severity"

# Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert ALL columns to numeric properly
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)
X = X.astype(float)

y = pd.to_numeric(y, errors="coerce")
y = y.fillna(0)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("Enter Accident Details")

# Create input fields for all columns
input_values = []

for col in X.columns:
    value = st.number_input(
        f"Enter {col}",
        min_value=0.0,
        value=0.0
    )
    input_values.append(value)

# Create input dataframe
input_data = pd.DataFrame(
    [input_values],
    columns=X.columns
)

# Convert input to float
input_data = input_data.astype(float)

# Prediction Button
if st.button("Predict Severity"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
