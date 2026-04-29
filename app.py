# Final Full Streamlit app.py Code

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Road Accident Severity Prediction System")
st.write("Predict accident severity using Machine Learning")

# Load Dataset
df = pd.read_csv("indian_road_accident_severity_10000.csv")

# Remove missing values
df = df.dropna()

# Save original target labels if available
target_column = "Accident_Severity"

# Convert object columns safely
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

y = pd.to_numeric(y, errors="coerce")
y = y.fillna(0)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("Enter Accident Details")

input_values = []

# Dropdown input fields
for col in X.columns:
    options = list(df[col].dropna().unique())

    selected_value = st.selectbox(
        f"Select {col}",
        options
    )

    input_values.append(selected_value)

# Create input dataframe
input_data = pd.DataFrame(
    [input_values],
    columns=X.columns
)

# Convert safely before prediction
input_data = input_data.apply(pd.to_numeric, errors="coerce")
input_data = input_data.fillna(0)

# Prediction
if st.button("Predict Severity"):
    prediction = model.predict(input_data)

    # Severity label mapping
    severity_map = {
        0: "Slight",
        1: "Serious",
        2: "Fatal"
    }

    predicted_label = severity_map.get(
        int(prediction[0]),
        "Unknown"
    )

    st.success(f"Predicted Accident Severity: {predicted_label}")
