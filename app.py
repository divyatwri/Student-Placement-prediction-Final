import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# Page Setting
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)


# ---------------------------------------------------------
# Custom Theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #32104B;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #555555;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #4B145F;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }

    .success-box {
        background-color: #EAF8F1;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #178F83;
        font-weight: 700;
        color: #0B5345;
    }

    .warning-box {
        background-color: #FFF6E6;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #F2A900;
        font-weight: 700;
        color: #7D6608;
    }

    .danger-box {
        background-color: #FDEDEC;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #D95C59;
        font-weight: 700;
        color: #922B21;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #32104B, #4B145F, #111111);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pkl file not found. Please keep model.pkl in the same folder as app.py.")
    st.stop()
except Exception as error:
    st.error("Model could not be loaded. Please check model.pkl and requirements.txt.")
    st.write(error)
    st.stop()


# ---------------------------------------------------------
# Feature Engineering Function
# This must match the final notebook feature engineering
# ---------------------------------------------------------
def create_features(data):
    data = data.copy()

    data["academic_average"] = data[
        ["ssc_p", "hsc_p", "degree_p", "mba_p"]
    ].mean(axis=1)

    data["employability_average"] = data[
        ["degree_p", "etest_p", "mba_p"]
    ].mean(axis=1)

    data["academic_growth"] = data["degree_p"] - data["ssc_p"]

    data["low_score_count"] = (
        (data["ssc_p"] < 60).astype(int) +
        (data["hsc_p"] < 60).astype(int) +
        (data["degree_p"] < 60).astype(int) +
        (data["etest_p"] < 60).astype(int) +
        (data["mba_p"] < 60).astype(int)
    )

    return data


# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.title("🎓 Student Details")
st.sidebar.write("Enter student information for prediction.")

gender = st.sidebar.selectbox("Gender", ["M", "F"])

ssc_p = st.sidebar.number_input(
    "SSC Percentage",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.5
)

ssc_b = st.sidebar.selectbox("SSC Board", ["Central", "Others"])

hsc_p = st.sidebar.number_input(
    "HSC Percentage",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.5
)

hsc_b = st.sidebar.selectbox("HSC Board", ["Central", "Others"])

hsc_s = st.sidebar.selectbox(
    "HSC Stream",
    ["Commerce", "Science", "Arts"]
)

degree_p = st.sidebar.number_input(
    "Degree Percentage",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.5
)

degree_t = st.sidebar.selectbox(
    "Degree Type",
    ["Comm&Mgmt", "Sci&Tech", "Others"]
)

workex = st.sidebar.selectbox("Work Experience", ["Yes", "No"])

etest_p = st.sidebar.number_input(
    "Employability Test Percentage",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=0.5
)

specialisation = st.sidebar.selectbox(
    "Specialisation",
    ["Mkt&Fin", "Mkt&HR", "Business Analytics"]
)

mba_p = st.sidebar.number_input(
    "MBA Percentage",
    min_value=0.0,
    max_value=100.0,
    value=62.0,
    step=0.5
)

predict_button = st.sidebar.button("Predict Placement")


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">Student Placement Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">A placement intelligence dashboard for predicting student placement readiness using machine learning.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
    <b>Business Objective:</b> This app helps placement teams evaluate student placement probability,
    readiness category, strengths, weaknesses and improvement recommendations.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Create Input Data
# ---------------------------------------------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "ssc_p": [ssc_p],
    "ssc_b": [ssc_b],
    "hsc_p": [hsc_p],
    "hsc_b": [hsc_b],
    "hsc_s": [hsc_s],
    "degree_p": [degree_p],
    "degree_t": [degree_t],
    "workex": [workex],
    "etest_p": [etest_p],
    "specialisation": [specialisation],
    "mba_p": [mba_p]
})

model_input = create_features(input_data)

academic_average = round(model_input["academic_average"].iloc[0], 2)
employability_average = round(model_input["employability_average"].iloc[0], 2)
academic_growth = round(model_input["academic_growth"].iloc[0], 2)
low_score_count = int(model_input["low_score_count"].iloc[0])


# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------
if predict_button:

    prediction = model.predict(model_input)[0]

    probability_values = model.predict_proba(model_input)[0]
    class_names = list(model.classes_)

    if "Placed" in class_names:
        placed_index = class_names.index("Placed")
    else:
        placed_index = 1

    placement_probability = round(probability_values[placed_index] * 100, 2)

    if placement_probability >= 75:
        readiness = "High Placement Readiness"
        risk = "Low Risk"
        box_class = "success-box"
    elif placement_probability >= 45:
        readiness = "Moderate Placement Readiness"
        risk = "Medium Risk"
        box_class = "warning-box"
    else:
        readiness = "High Support Required"
        risk = "High Risk"
        box_class = "danger-box"

    # -----------------------------------------------------
    # KPI Cards
    # -----------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Placement Probability", f"{placement_probability}%")
    col2.metric("Academic Average", f"{academic_average}%")
    col3.metric("Employability Average", f"{employability_average}%")
    col4.metric("Risk Level", risk)

    # -----------------------------------------------------
    # Prediction Result
    # -----------------------------------------------------
    st.subheader("Prediction Result")

    if prediction == "Placed":
        st.markdown(
            f'<div class="success-box">Prediction: {prediction}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="danger-box">Prediction: {prediction}</div>',
            unsafe_allow_html=True
        )

    st.subheader("Readiness Category")
    st.markdown(
        f'<div class="{box_class}">{readiness}</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Tabs
    # -----------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Summary",
        "Visual Analytics",
        "Recommendations",
        "Final Report"
    ])

    # -----------------------------------------------------
    # Tab 1: Executive Summary
    # -----------------------------------------------------
    with tab1:
        st.write("### Executive Summary")

        st.write(
            f"""
            The model predicts the student as **{prediction}** with a placement probability of 
            **{placement_probability}%**. The student falls under the **{readiness}** category.
            
            This output can help placement teams identify student readiness, support needs and improvement areas.
            """
        )

        summary = pd.DataFrame({
            "Metric": [
                "Prediction",
                "Placement Probability",
                "Readiness Category",
                "Risk Level",
                "Academic Average",
                "Employability Average",
                "Academic Growth",
                "Low Score Count",
                "Work Experience",
                "Specialisation"
            ],
            "Value": [
                prediction,
                f"{placement_probability}%",
                readiness,
                risk,
                f"{academic_average}%",
                f"{employability_average}%",
                academic_growth,
                low_score_count,
                workex,
                specialisation
            ]
        })

        st.dataframe(summary, use_container_width=True)

    # -----------------------------------------------------
    # Tab 2: Visual Analytics
    # -----------------------------------------------------
    with tab2:
        st.write("### Placement Probability Gauge")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=placement_probability,
            title={"text": "Placement Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4B145F"},
                "steps": [
                    {"range": [0, 45], "color": "#FDEDEC"},
                    {"range": [45, 75], "color": "#FFF6E6"},
                    {"range": [75, 100], "color": "#EAF8F1"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.write("### Student Score vs Benchmark")

        benchmark = pd.DataFrame({
            "Area": ["SSC", "HSC", "Degree", "Employability Test", "MBA"],
            "Student Score": [ssc_p, hsc_p, degree_p, etest_p, mba_p],
            "Benchmark": [65, 65, 65, 70, 62]
        })

        benchmark_long = benchmark.melt(
            id_vars="Area",
            value_vars=["Student Score", "Benchmark"],
            var_name="Measure",
            value_name="Score"
        )

        benchmark_fig = px.bar(
            benchmark_long,
            x="Area",
            y="Score",
            color="Measure",
            barmode="group",
            text_auto=".1f",
            title="Student Score Compared with Benchmark"
        )

        benchmark_fig.update_layout(
            yaxis_range=[0, 100],
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(benchmark_fig, use_container_width=True)

        st.write("### Strength and Weakness View")

        strength = pd.DataFrame({
            "Area": ["SSC", "HSC", "Degree", "Employability Test", "MBA"],
            "Score": [ssc_p, hsc_p, degree_p, etest_p, mba_p]
        })

        strength["Status"] = np.where(
            strength["Score"] >= 70,
            "Strong",
            np.where(strength["Score"] >= 60, "Moderate", "Needs Improvement")
        )

        strength_fig = px.bar(
            strength,
            x="Score",
            y="Area",
            orientation="h",
            color="Status",
            text="Score",
            title="Student Strength and Weakness Analysis"
        )

        strength_fig.update_layout(
            xaxis_range=[0, 100],
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(strength_fig, use_container_width=True)

    # -----------------------------------------------------
    # Tab 3: Recommendations
    # -----------------------------------------------------
    with tab3:
        st.write("### Personalized Recommendations")

        recommendations = []

        if degree_p < 65:
            recommendations.append([
                "High",
                "Improve degree-level academic performance because degree score strongly affects placement readiness."
            ])

        if etest_p < 70:
            recommendations.append([
                "High",
                "Focus on aptitude, reasoning and employability test preparation."
            ])

        if workex == "No":
            recommendations.append([
                "Medium",
                "Add internship, live project or industry exposure to improve practical profile."
            ])

        if mba_p < 62:
            recommendations.append([
                "Medium",
                "Strengthen MBA academic performance and business fundamentals."
            ])

        if ssc_p < 60 or hsc_p < 60:
            recommendations.append([
                "Medium",
                "Improve academic foundation and communication confidence."
            ])

        if specialisation == "Mkt&HR":
            recommendations.append([
                "Low",
                "Add HR analytics, Excel, Power BI or business analytics skills for stronger placement profile."
            ])

        if len(recommendations) == 0:
            recommendations.append([
                "Low",
                "Profile is strong. Continue interview preparation, resume improvement and LinkedIn optimization."
            ])

        recommendation_df = pd.DataFrame(
            recommendations,
            columns=["Priority", "Recommendation"]
        )

        st.dataframe(recommendation_df, use_container_width=True)

        st.markdown(
            """
            ### Suggested Skill Roadmap
            
            - Resume and LinkedIn improvement  
            - Mock interview and group discussion practice  
            - Aptitude and logical reasoning preparation  
            - Excel, SQL, Power BI or Tableau project  
            - Internship or live project experience  
            """
        )

    # -----------------------------------------------------
    # Tab 4: Final Report
    # -----------------------------------------------------
    with tab4:
        st.write("### Final Prediction Report")

        report = pd.DataFrame({
            "Field": [
                "Prediction",
                "Placement Probability",
                "Readiness Category",
                "Risk Level",
                "Academic Average",
                "Employability Average",
                "Academic Growth",
                "Low Score Count",
                "Work Experience",
                "Specialisation"
            ],
            "Value": [
                prediction,
                f"{placement_probability}%",
                readiness,
                risk,
                f"{academic_average}%",
                f"{employability_average}%",
                academic_growth,
                low_score_count,
                workex,
                specialisation
            ]
        })

        st.dataframe(report, use_container_width=True)

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Final Report",
            data=csv,
            file_name="student_placement_prediction_report.csv",
            mime="text/csv"
        )

else:
    st.info("Enter student details in the sidebar and click Predict Placement.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Academic project only. Prediction should support placement decisions and should not replace human judgement."
)