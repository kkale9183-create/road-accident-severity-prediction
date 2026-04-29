import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .severity-fatal {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .severity-grievous {
        background: linear-gradient(135deg, #ff8c00, #e65c00);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .severity-minor {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; border-left: 4px solid #4A90E2;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

SEVERITY_COLORS = {
    "Fatal": "#d62728",
    "Grievous Injury": "#ff7f0e",
    "Minor Injury": "#2ca02c",
}

SAFETY_TIPS = {
    "Fatal": [
        "This combination is statistically very high-risk.",
        "Never drive under influence of alcohol or drugs.",
        "Avoid driving during night hours in foggy/rainy weather.",
        "Always wear a helmet or seatbelt — it can save your life.",
        "Alert emergency contacts when driving in high-risk conditions.",
    ],
    "Grievous Injury": [
        "This combination carries a serious risk of severe injury.",
        "Reduce speed significantly on rural roads and highways.",
        "Maintain a safe following distance at all times.",
        "Ensure your vehicle is mechanically sound before long trips.",
        "Avoid overtaking on curves or low-visibility stretches.",
    ],
    "Minor Injury": [
        "Risk is relatively lower, but stay alert.",
        "Follow all traffic signals and road markings.",
        "Be cautious at intersections and pedestrian crossings.",
        "Keep distractions like mobile phones away while driving.",
        "Stay aware of cyclists and pedestrians sharing the road.",
    ],
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("indian_road_accident_severity_10000.csv")
    df = df.dropna()
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    df_enc = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    X = df_enc.drop("Accident_Severity", axis=1)
    y = df_enc["Accident_Severity"]

    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, encoders, list(X.columns)


df = load_data()
model, encoders, feature_cols = train_model(df)
severity_classes: list[str] = list(encoders["Accident_Severity"].classes_)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/traffic-jam.png",
        width=72,
    )
    st.title("Road Accident\nSeverity Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🔮 Predict Severity", "📊 Dashboard", "🔍 Risk Explorer", "📋 Dataset Explorer"],
    )
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} accident records | 2017–2023")
    st.caption("States: 10 major Indian states")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT SEVERITY
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Severity":
    st.title("🔮 Predict Accident Severity")
    st.markdown("Fill in the accident scenario details below to get an AI-powered severity prediction.")

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.subheader("Accident Scenario")

        c1, c2 = st.columns(2)
        with c1:
            state = st.selectbox("State", sorted(df["State"].unique()))
            year = st.selectbox("Year", sorted(df["Year"].unique(), reverse=True))
            month = st.selectbox("Month", list(MONTH_NAMES.values()),
                                 index=0,
                                 format_func=lambda x: x)
            hour = st.slider("Hour of Day", 0, 23, 8,
                             help="0 = midnight, 12 = noon, 23 = 11 PM")

        with c2:
            district = st.selectbox("District", sorted(df["District"].unique()))
            weather = st.selectbox("Weather Condition", sorted(df["Weather_Condition"].unique()))
            road_type = st.selectbox("Road Type", sorted(df["Road_Type"].unique()))
            vehicle = st.selectbox("Vehicle Type", sorted(df["Vehicle_Type"].unique()))

        cause = st.selectbox("Cause of Accident", sorted(df["Cause_of_Accident"].unique()))

        predict_btn = st.button("🚦 Predict Severity", use_container_width=True, type="primary")

    with col_right:
        st.subheader("Prediction Result")

        if predict_btn:
            month_num = {v: k for k, v in MONTH_NAMES.items()}[month]

            raw_input = {
                "State": state, "District": district, "Year": year,
                "Month": month_num, "Hour": hour,
                "Weather_Condition": weather, "Road_Type": road_type,
                "Vehicle_Type": vehicle, "Cause_of_Accident": cause,
            }

            encoded_input: dict[str, int | float] = {}
            for col in feature_cols:
                val = raw_input[col]
                if col in encoders:
                    encoded_input[col] = int(encoders[col].transform([val])[0])
                else:
                    encoded_input[col] = float(val)

            input_df = pd.DataFrame([encoded_input])[feature_cols]
            probs = model.predict_proba(input_df)[0]
            predicted_idx = int(np.argmax(probs))
            predicted_label: str = severity_classes[predicted_idx]

            # Result card
            css_class = {
                "Fatal": "severity-fatal",
                "Grievous Injury": "severity-grievous",
                "Minor Injury": "severity-minor",
            }[predicted_label]
            icon = {"Fatal": "💀", "Grievous Injury": "🚑", "Minor Injury": "🩹"}[predicted_label]

            st.markdown(
                f'<div class="{css_class}">{icon} {predicted_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Probability bars
            st.markdown("**Confidence per severity level:**")
            prob_df = pd.DataFrame({
                "Severity": severity_classes,
                "Probability": [round(p * 100, 1) for p in probs],
            }).sort_values("Probability", ascending=False)

            fig_prob = px.bar(
                prob_df, x="Probability", y="Severity",
                orientation="h",
                color="Severity",
                color_discrete_map=SEVERITY_COLORS,
                text="Probability",
                range_x=[0, 100],
            )
            fig_prob.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_prob.update_layout(
                showlegend=False, margin=dict(l=0, r=20, t=10, b=0),
                height=180, xaxis_title="Probability (%)", yaxis_title="",
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Safety tips
            st.markdown("**Safety Recommendations:**")
            for tip in SAFETY_TIPS[predicted_label]:
                st.markdown(f"- {tip}")

        else:
            st.info("Configure the accident scenario on the left and click **Predict Severity**.")

            # Feature importance preview
            st.markdown("**Top Factors Influencing Severity:**")
            fi_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(8)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0),
                height=280,
                xaxis_title="Importance Score",
                yaxis_title="",
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Accident Analytics Dashboard")
    st.markdown("Explore patterns across 10,000 road accident records from 10 major Indian states (2017–2023).")

    # KPI row
    total = len(df)
    fatal_pct = round(len(df[df["Accident_Severity"] == "Fatal"]) / total * 100, 1)
    grievous_pct = round(len(df[df["Accident_Severity"] == "Grievous Injury"]) / total * 100, 1)
    minor_pct = round(len(df[df["Accident_Severity"] == "Minor Injury"]) / total * 100, 1)
    top_cause = df["Cause_of_Accident"].value_counts().idxmax()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Accidents", f"{total:,}")
    k2.metric("Fatal", f"{fatal_pct}%", delta=None)
    k3.metric("Grievous Injury", f"{grievous_pct}%")
    k4.metric("Minor Injury", f"{minor_pct}%")
    k5.metric("Top Cause", top_cause)

    st.markdown("---")

    # Row 1: Accidents by State | Year Trend
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        st.subheader("Accidents by State")
        state_sev = (
            df.groupby(["State", "Accident_Severity"])
            .size().reset_index(name="Count")
        )
        fig_state = px.bar(
            state_sev, x="State", y="Count",
            color="Accident_Severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
        )
        fig_state.update_layout(
            xaxis_tickangle=-30, legend_title="Severity",
            margin=dict(l=0, r=0, t=10, b=0), height=350,
        )
        st.plotly_chart(fig_state, use_container_width=True)

    with r1c2:
        st.subheader("Year-on-Year Trend")
        year_sev = (
            df.groupby(["Year", "Accident_Severity"])
            .size().reset_index(name="Count")
        )
        fig_year = px.line(
            year_sev, x="Year", y="Count",
            color="Accident_Severity",
            color_discrete_map=SEVERITY_COLORS,
            markers=True,
        )
        fig_year.update_layout(
            legend_title="Severity",
            margin=dict(l=0, r=0, t=10, b=0), height=350,
        )
        st.plotly_chart(fig_year, use_container_width=True)

    # Row 2: Hour heatmap | Weather × Severity
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        st.subheader("Accidents by Hour of Day")
        hour_sev = (
            df.groupby(["Hour", "Accident_Severity"])
            .size().reset_index(name="Count")
        )
        fig_hour = px.bar(
            hour_sev, x="Hour", y="Count",
            color="Accident_Severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
        )
        fig_hour.update_layout(
            xaxis=dict(tickmode="linear", dtick=2),
            legend_title="Severity",
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            xaxis_title="Hour (0–23)", yaxis_title="Count",
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with r2c2:
        st.subheader("Weather Condition vs Severity")
        weather_sev = (
            df.groupby(["Weather_Condition", "Accident_Severity"])
            .size().reset_index(name="Count")
        )
        fig_weather = px.bar(
            weather_sev, x="Weather_Condition", y="Count",
            color="Accident_Severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="group",
        )
        fig_weather.update_layout(
            legend_title="Severity",
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            xaxis_title="Weather", yaxis_title="Count",
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    # Row 3: Vehicle type | Cause of accident
    r3c1, r3c2 = st.columns(2, gap="medium")

    with r3c1:
        st.subheader("Vehicle Type Distribution")
        vehicle_counts = df["Vehicle_Type"].value_counts().reset_index()
        vehicle_counts.columns = ["Vehicle_Type", "Count"]
        fig_veh = px.pie(
            vehicle_counts, names="Vehicle_Type", values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_veh.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=320,
        )
        st.plotly_chart(fig_veh, use_container_width=True)

    with r3c2:
        st.subheader("Top Causes of Accidents")
        cause_sev = (
            df.groupby(["Cause_of_Accident", "Accident_Severity"])
            .size().reset_index(name="Count")
        )
        fig_cause = px.bar(
            cause_sev, x="Count", y="Cause_of_Accident",
            color="Accident_Severity",
            color_discrete_map=SEVERITY_COLORS,
            orientation="h", barmode="stack",
        )
        fig_cause.update_layout(
            legend_title="Severity",
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            yaxis_title="", xaxis_title="Count",
        )
        st.plotly_chart(fig_cause, use_container_width=True)

    # Row 4: Month heatmap
    st.subheader("Monthly Accident Heatmap (Fatal only)")
    pivot = (
        df[df["Accident_Severity"] == "Fatal"]
        .groupby(["State", "Month"])
        .size()
        .reset_index(name="Fatal Count")
    )
    pivot["Month Name"] = pivot["Month"].map(MONTH_NAMES)
    month_order = list(MONTH_NAMES.values())
    fig_heat = px.density_heatmap(
        pivot, x="Month Name", y="State", z="Fatal Count",
        color_continuous_scale="Reds",
        category_orders={"Month Name": month_order},
    )
    fig_heat.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=340,
        xaxis_title="Month", yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Road type breakdown
    st.subheader("Road Type vs Severity")
    road_sev = (
        df.groupby(["Road_Type", "Accident_Severity"])
        .size().reset_index(name="Count")
    )
    road_pct = road_sev.copy()
    totals = road_pct.groupby("Road_Type")["Count"].transform("sum")
    road_pct["Percentage"] = (road_pct["Count"] / totals * 100).round(1)

    fig_road = px.bar(
        road_pct, x="Road_Type", y="Percentage",
        color="Accident_Severity",
        color_discrete_map=SEVERITY_COLORS,
        barmode="stack",
        text="Percentage",
    )
    fig_road.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig_road.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=340,
        xaxis_title="Road Type", yaxis_title="Percentage (%)",
        legend_title="Severity",
    )
    st.plotly_chart(fig_road, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Risk Explorer":
    st.title("🔍 Risk Explorer")
    st.markdown(
        "Select a combination of factors to explore **real historical accident statistics** "
        "for that scenario from the dataset."
    )

    col_filters, col_results = st.columns([1, 1.6], gap="large")

    with col_filters:
        st.subheader("Filter Conditions")
        f_state = st.selectbox("State", ["All"] + sorted(df["State"].unique()))
        f_weather = st.selectbox("Weather", ["All"] + sorted(df["Weather_Condition"].unique()))
        f_road = st.selectbox("Road Type", ["All"] + sorted(df["Road_Type"].unique()))
        f_vehicle = st.selectbox("Vehicle Type", ["All"] + sorted(df["Vehicle_Type"].unique()))
        f_cause = st.selectbox("Cause", ["All"] + sorted(df["Cause_of_Accident"].unique()))
        hour_range = st.slider("Hour Range", 0, 23, (0, 23))

    filtered = df.copy()
    if f_state != "All":
        filtered = filtered[filtered["State"] == f_state]
    if f_weather != "All":
        filtered = filtered[filtered["Weather_Condition"] == f_weather]
    if f_road != "All":
        filtered = filtered[filtered["Road_Type"] == f_road]
    if f_vehicle != "All":
        filtered = filtered[filtered["Vehicle_Type"] == f_vehicle]
    if f_cause != "All":
        filtered = filtered[filtered["Cause_of_Accident"] == f_cause]
    filtered = filtered[
        (filtered["Hour"] >= hour_range[0]) & (filtered["Hour"] <= hour_range[1])
    ]

    with col_results:
        st.subheader("Historical Statistics for Your Selection")

        if filtered.empty:
            st.warning("No records match this combination. Try relaxing some filters.")
        else:
            total_f = len(filtered)
            sev_counts = filtered["Accident_Severity"].value_counts()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", total_f)
            m2.metric("Fatal", sev_counts.get("Fatal", 0),
                      delta=f"{sev_counts.get('Fatal', 0)/total_f*100:.1f}%")
            m3.metric("Grievous", sev_counts.get("Grievous Injury", 0),
                      delta=f"{sev_counts.get('Grievous Injury', 0)/total_f*100:.1f}%")
            m4.metric("Minor", sev_counts.get("Minor Injury", 0),
                      delta=f"{sev_counts.get('Minor Injury', 0)/total_f*100:.1f}%")

            # Severity breakdown pie
            sev_df = sev_counts.reset_index()
            sev_df.columns = ["Severity", "Count"]
            fig_sev = px.pie(
                sev_df, names="Severity", values="Count",
                color="Severity", color_discrete_map=SEVERITY_COLORS,
                hole=0.45,
            )
            fig_sev.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
            st.plotly_chart(fig_sev, use_container_width=True)

            # Hourly distribution within filter
            st.markdown("**Accidents by Hour (filtered)**")
            hour_dist = (
                filtered.groupby(["Hour", "Accident_Severity"])
                .size().reset_index(name="Count")
            )
            fig_hd = px.bar(
                hour_dist, x="Hour", y="Count",
                color="Accident_Severity",
                color_discrete_map=SEVERITY_COLORS,
                barmode="stack",
            )
            fig_hd.update_layout(
                margin=dict(l=0, r=0, t=6, b=0), height=240,
                xaxis_title="Hour", yaxis_title="Count",
                legend_title="Severity",
                xaxis=dict(tickmode="linear", dtick=2),
            )
            st.plotly_chart(fig_hd, use_container_width=True)

            # Dominant risk insight
            dominant_sev = sev_counts.idxmax()
            dominant_pct = round(sev_counts.max() / total_f * 100, 1)
            top_cause_filtered = filtered["Cause_of_Accident"].value_counts().idxmax()
            color_map = {"Fatal": "🔴", "Grievous Injury": "🟠", "Minor Injury": "🟢"}

            st.info(
                f"{color_map[dominant_sev]} **{dominant_pct}%** of accidents in this scenario "
                f"result in **{dominant_sev}**. "
                f"The leading cause is **{top_cause_filtered}**."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":
    st.title("📋 Dataset Explorer")
    st.markdown("Browse, filter, and summarise the 10,000 accident records.")

    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            e_state = st.multiselect("State", sorted(df["State"].unique()),
                                     default=sorted(df["State"].unique()))
            e_year = st.multiselect("Year", sorted(df["Year"].unique()),
                                    default=sorted(df["Year"].unique()))
        with fc2:
            e_weather = st.multiselect("Weather", sorted(df["Weather_Condition"].unique()),
                                       default=sorted(df["Weather_Condition"].unique()))
            e_road = st.multiselect("Road Type", sorted(df["Road_Type"].unique()),
                                    default=sorted(df["Road_Type"].unique()))
        with fc3:
            e_vehicle = st.multiselect("Vehicle Type", sorted(df["Vehicle_Type"].unique()),
                                       default=sorted(df["Vehicle_Type"].unique()))
            e_severity = st.multiselect("Severity", sorted(df["Accident_Severity"].unique()),
                                        default=sorted(df["Accident_Severity"].unique()))

    exp_df = df[
        df["State"].isin(e_state) &
        df["Year"].isin(e_year) &
        df["Weather_Condition"].isin(e_weather) &
        df["Road_Type"].isin(e_road) &
        df["Vehicle_Type"].isin(e_vehicle) &
        df["Accident_Severity"].isin(e_severity)
    ]

    st.markdown(f"**Showing {len(exp_df):,} of {len(df):,} records**")

    # Summary stats
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Severity Breakdown**")
        sv = exp_df["Accident_Severity"].value_counts().reset_index()
        sv.columns = ["Severity", "Count"]
        sv["Pct"] = (sv["Count"] / len(exp_df) * 100).round(1).astype(str) + "%"
        st.dataframe(sv, use_container_width=True, hide_index=True)
    with s2:
        st.markdown("**Top Causes**")
        tc = exp_df["Cause_of_Accident"].value_counts().head(6).reset_index()
        tc.columns = ["Cause", "Count"]
        st.dataframe(tc, use_container_width=True, hide_index=True)
    with s3:
        st.markdown("**Vehicle Breakdown**")
        vb = exp_df["Vehicle_Type"].value_counts().reset_index()
        vb.columns = ["Vehicle", "Count"]
        st.dataframe(vb, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.dataframe(
        exp_df.reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    csv_data = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_accidents.csv",
        mime="text/csv",
    )
