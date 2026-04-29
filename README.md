# 🚦 Road Accident Severity Prediction

An interactive **Streamlit web application** that uses Machine Learning to predict road accident severity and provides rich data analytics for 10,000 road accident records across 10 major Indian states (2017–2023).

---

## Features

### 🔮 Predict Severity
- Input accident scenario details (state, weather, road type, vehicle, cause, time)
- Get an AI-powered severity prediction: **Minor Injury**, **Grievous Injury**, or **Fatal**
- Color-coded result card (green / orange / red)
- Confidence probability bars for all three severity levels
- Contextual safety recommendations based on prediction
- Feature importance chart showing which factors drive severity the most

### 📊 Analytics Dashboard
- **KPI row** — total accidents, % fatal, % grievous, % minor, top cause
- **Accidents by State** — stacked bar chart with severity breakdown
- **Year-on-Year Trend** — line chart showing 2017–2023 trajectory
- **Accidents by Hour** — identifies peak risk hours across the day
- **Weather vs Severity** — grouped bar showing impact of weather on outcomes
- **Vehicle Type Distribution** — donut chart of involved vehicle types
- **Top Causes of Accidents** — horizontal stacked bar
- **Monthly Fatal Heatmap** — state × month density map for fatal accidents
- **Road Type vs Severity** — percentage-normalised stacked bar

### 🔍 Risk Explorer
- Filter by state, weather, road type, vehicle type, cause, and hour range
- See real historical accident counts and severity breakdown for that combination
- Hourly distribution chart for the filtered scenario
- Auto-generated dominant risk insight with leading cause

### 📋 Dataset Explorer
- Multi-select filters across all dimensions
- Live summary tables (severity, causes, vehicle types)
- Full filterable data table
- Download filtered data as CSV

---

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Web UI framework |
| scikit-learn | RandomForestClassifier model |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Plotly | Interactive charts |

---

## Getting Started

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset

`indian_road_accident_severity_10000.csv` — 10,000 synthetic road accident records with the following columns:

| Column | Values |
|---|---|
| State | 10 major Indian states |
| District | 10 major cities |
| Year | 2017–2023 |
| Month | 1–12 |
| Hour | 0–23 |
| Weather_Condition | Clear, Cloudy, Foggy, Rainy |
| Road_Type | National Highway, State Highway, Urban Road, Rural Road |
| Vehicle_Type | Two-Wheeler, Car, Bus, Truck, Auto Rickshaw |
| Cause_of_Accident | Overspeeding, Wrong Side Driving, Drunk Driving, Signal Jumping, Poor Road Condition, Mechanical Failure |
| Accident_Severity | Minor Injury, Grievous Injury, Fatal |

---

## Model

- **Algorithm:** Random Forest Classifier (150 estimators)
- **Encoding:** Label Encoding for all categorical features
- **Target:** `Accident_Severity` (3 classes)
- **Output:** Predicted class + probability scores for all 3 severity levels
