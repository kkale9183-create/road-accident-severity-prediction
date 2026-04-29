import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Title
# -------------------------------
st.title("Road Accident Severity Prediction System")
st.write("Machine Learning Project using Streamlit")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("indian_road_accident_severity_10000.csv")

# Remove missing values
df = df.dropna()

# Target column
target_column = "Accident_Severity"

# -------------------------------
# Encode Categorical Columns
# -------------------------------
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# -------------------------------
# Features and Target
# -------------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert to numeric safely
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
y = pd.to_numeric(y, errors="coerce").fillna(0)

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("Enter Accident Details")

input_values = []

for col in X.columns:
    options = list(df[col].dropna().unique())

    selected_value = st.selectbox(
        f"Select {col}",
        options
    )

    input_values.append(selected_value)

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame(
    [input_values],
    columns=X.columns
)

input_data = input_data.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Severity"):
    prediction = model.predict(input_data)

    # Severity Mapping
    severity_map = {
        0: "Slight",
        1: "Serious",
        2: "Fatal"
    }

    predicted_label = severity_map.get(
        int(prediction[0]),
        "Unknown"
    )

    st.success(
        f"Predicted Accident Severity: {predicted_label}"
    )
