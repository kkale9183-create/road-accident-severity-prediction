# Correct app.py (Fix ValueError)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Road Accident Severity Prediction System")
st.write("Machine Learning Project using Streamlit")

# Load Dataset
df = pd.read_csv("indian_road_accident_severity_10000.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# Remove missing values
df = df.dropna()

# Convert all object columns into numeric
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# Target Column
target_column = "Accident_Severity"

# Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Extra safety: convert all columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("Enter Accident Details")

# Use first 4 columns for input example
feature_columns = X.columns[:4]

input_values = []

for col in feature_columns:
    value = st.number_input(f"Enter {col}", value=0)
    input_values.append(value)

input_data = pd.DataFrame(
    [input_values],
    columns=feature_columns
)

# Prediction
if st.button("Predict Severity"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
