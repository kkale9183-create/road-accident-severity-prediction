# Step: Create new app.py file and download it

code = '''
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Road Accident Severity Prediction System")
st.write("Machine Learning Project using Streamlit")

# Load Dataset
df = pd.read_csv("indian_road_accident_severity_10000.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# Encode categorical columns
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].astype('category').cat.codes

# Features and Target
X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("Enter Accident Details")

# Input Fields
age = st.number_input("Driver Age", min_value=18, max_value=80, value=30)
speed = st.number_input("Vehicle Speed", min_value=0, max_value=200, value=60)
weather = st.selectbox("Weather Condition", [0, 1, 2])
road_type = st.selectbox("Road Type", [0, 1, 2])

# Input Data
input_data = pd.DataFrame([[
    age,
    speed,
    weather,
    road_type
]], columns=[
    "Driver_Age",
    "Vehicle_Speed",
    "Weather_Condition",
    "Road_Type"
])

# Prediction
if st.button("Predict Severity"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
'''

with open("app.py", "w") as f:
    f.write(code)

print("New app.py created successfully")

from google.colab import files
files.download("app.py")
