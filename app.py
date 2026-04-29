# Final Correct app.py (Fix Feature Names Error)

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

# Convert object columns into numeric
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# Target column
target_column = "Accident_Severity"

# Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("Enter Accident Details")

# Create input for ALL feature columns
input_values = []

for col in X.columns:
    value = st.number_input(f"Enter {col}", value=0)
    input_values.append(value)

# Input dataframe with SAME columns as training data
input_data = pd.DataFrame(
    [input_values],
    columns=X.columns
)

# Prediction
if st.button("Predict Severity"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
