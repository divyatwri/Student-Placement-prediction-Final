# ============================================================
# STUDENT PLACEMENT PREDICTION SYSTEM
# COMPLETE STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Student Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. APP THEME
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #2F123D;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #5A5A5A;
        margin-bottom: 20px;
    }

    .objective-card {
        background-color: white;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #4B145F;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .success-card {
        background-color: #EAF8F1;
        color: #0B5345;
        border-left: 6px solid #178F83;
        padding: 18px;
        border-radius: 12px;
        font-weight: 700;
    }

    .warning-card {
        background-color: #FFF6E6;
        color: #7D6608;
        border-left: 6px solid #F2A900;
        padding: 18px;
        border-radius: 12px;
        font-weight: 700;
    }

    .danger-card {
        background-color: #FDEDEC;
        color: #922B21;
        border-left: 6px solid #D95C59;
        padding: 18px;
        border-radius: 12px;
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111111,
            #291234,
            #4B145F
        );
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. SAFE TABLE FUNCTION
# FIXES PYARROW MIXED DATA-TYPE ERROR
# ============================================================

def show_safe_table(dataframe):
    safe_dataframe = dataframe.copy()

    for column in safe_dataframe.columns:
        safe_dataframe[column] = (
            safe_dataframe[column]
            .fillna("")
            .astype(str)
        )

    st.dataframe(
        safe_dataframe,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "model.pkl was not found. Keep model.pkl in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error("The trained model could not be loaded.")
    st.write(str(error))
    st.stop()


# ============================================================
# 5. FEATURE ENGINEERING
# MUST MATCH THE NOTEBOOK
# ============================================================

def create_features(data):
    data = data.copy()

    data["academic_average"] = data[
        [
            "ssc_p",
            "hsc_p",
            "degree_p",
            "mba_p"
        ]
    ].mean(axis=1)

    data["employability_average"] = data[
        [
            "degree_p",
            "etest_p",
            "mba_p"
        ]
    ].mean(axis=1)

    data["academic_growth"] = (
        data["degree_p"] - data["ssc_p"]
    )

    data["low_score_count"] = (
        (data["ssc_p"] < 60).astype(int)
        + (data["hsc_p"] < 60).astype(int)
        + (data["degree_p"] < 60).astype(int)
        + (data["etest_p"] < 60).astype(int)
        + (data["mba_p"] < 60).astype(int)
    )

    return data


# ============================================================
# 6. SIDEBAR INPUTS
# ============================================================

st.sidebar.title("🎓 Student Details")

st.sidebar.write(
    "Enter academic and employability information."
)

gender = st.sidebar.selectbox(
    "Gender",
    ["M", "F"]
)

ssc_p = st.sidebar.number_input(
    "SSC Percentage",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.5
)

ssc_b = st.sidebar.selectbox(
    "SSC Board",
    ["Central", "Others"]
)

hsc_p = st.sidebar.number_input(
    "HSC Percentage",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.5
)

hsc_b = st.sidebar.selectbox(
    "HSC Board",
    ["Central", "Others"]
)

hsc_s = st.sidebar.selectbox(
    "HSC Stream",
    [
        "Commerce",
        "Science",
        "Arts"
    ]
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
    [
        "Comm&Mgmt",
        "Sci&Tech",
        "Others"
    ]
)

workex = st.sidebar.selectbox(
    "Work Experience",
    ["Yes", "No"]
)

etest_p = st.sidebar.number_input(
    "Employability Test Percentage",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=0.5
)

specialisation = st.sidebar.selectbox(
    "Specialisation",
    [
        "Mkt&Fin",
        "Mkt&HR",
        "Business Analytics"
    ]
)

mba_p = st.sidebar.number_input(
    "MBA Percentage",
    min_value=0.0,
    max_value=100.0,
    value=62.0,
    step=0.5
)

predict_button = st.sidebar.button(
    "Predict Placement",
    use_container_width=True
)


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    Student Placement Prediction System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    A placement intelligence dashboard for predicting student
    placement readiness using machine learning.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="objective-card">
    <b>Business Objective:</b>
    Evaluate student placement probability, readiness category,
    strengths, weaknesses and improvement recommendations.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. CREATE MODEL INPUT
# ============================================================

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

academic_average = round(
    float(model_input["academic_average"].iloc[0]),
    2
)

employability_average = round(
    float(model_input["employability_average"].iloc[0]),
    2
)

academic_growth = round(
    float(model_input["academic_growth"].iloc[0]),
    2
)

low_score_count = int(
    model_input["low_score_count"].iloc[0]
)


# ============================================================
# 9. RUN PREDICTION
# ============================================================

if predict_button:

    try:
        prediction = model.predict(model_input)[0]

        probability_values = model.predict_proba(
            model_input
        )[0]

        model_classes = list(model.classes_)

        if "Placed" in model_classes:
            placed_index = model_classes.index("Placed")
        else:
            placed_index = 1

        placement_probability = round(
            float(probability_values[placed_index]) * 100,
            2
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.write(str(error))
        st.stop()


    # ========================================================
    # 10. READINESS CATEGORY
    # ========================================================

    if placement_probability >= 75:

        readiness = "High Placement Readiness"
        risk_level = "Low Risk"
        readiness_icon = "🟢"
        readiness_card = "success-card"

    elif placement_probability >= 45:

        readiness = "Moderate Placement Readiness"
        risk_level = "Medium Risk"
        readiness_icon = "🟡"
        readiness_card = "warning-card"

    else:

        readiness = "High Support Required"
        risk_level = "High Risk"
        readiness_icon = "🔴"
        readiness_card = "danger-card"


    # ========================================================
    # 11. KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Placement Probability",
        f"{placement_probability:.2f}%"
    )

    col2.metric(
        "Academic Average",
        f"{academic_average:.2f}%"
    )

    col3.metric(
        "Employability Average",
        f"{employability_average:.2f}%"
    )

    col4.metric(
        "Risk Level",
        risk_level
    )


    # ========================================================
    # 12. PREDICTION RESULT
    # ========================================================

    st.subheader("Prediction Result")

    if str(prediction) == "Placed":
        prediction_card = "success-card"
    else:
        prediction_card = "danger-card"

    st.markdown(
        f"""
        <div class="{prediction_card}">
        Prediction: {prediction}<br>
        Placement Probability: {placement_probability:.2f}%<br>
        Risk Level: {risk_level}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Readiness Category")

    st.markdown(
        f"""
        <div class="{readiness_card}">
        {readiness_icon} {readiness}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 13. TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Summary",
        "Visual Analytics",
        "Recommendations",
        "Final Report"
    ])


    # ========================================================
    # TAB 1 — EXECUTIVE SUMMARY
    # ========================================================

    with tab1:

        st.subheader("Executive Summary")

        if placement_probability >= 75:

            executive_message = (
                "The student demonstrates strong placement readiness. "
                "The recommended focus is interview preparation and "
                "targeted applications."
            )

        elif placement_probability >= 45:

            executive_message = (
                "The student demonstrates moderate placement readiness. "
                "Some academic or employability gaps should be improved."
            )

        else:

            executive_message = (
                "The student requires structured placement support. "
                "Academic, aptitude and practical-development actions "
                "are recommended."
            )

        st.write(executive_message)

        summary_table = pd.DataFrame({
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
                str(prediction),
                f"{placement_probability:.2f}%",
                str(readiness),
                str(risk_level),
                f"{academic_average:.2f}%",
                f"{employability_average:.2f}%",
                f"{academic_growth:.2f}",
                str(low_score_count),
                str(workex),
                str(specialisation)
            ]
        })

        show_safe_table(summary_table)


    # ========================================================
    # TAB 2 — VISUAL ANALYTICS
    # ========================================================

    with tab2:

        st.subheader("Placement Probability Gauge")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=placement_probability,

                number={
                    "suffix": "%"
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#4B145F"
                    },

                    "steps": [
                        {
                            "range": [0, 45],
                            "color": "#FDEDEC"
                        },
                        {
                            "range": [45, 75],
                            "color": "#FFF6E6"
                        },
                        {
                            "range": [75, 100],
                            "color": "#EAF8F1"
                        }
                    ],

                    "threshold": {
                        "line": {
                            "color": "#32104B",
                            "width": 4
                        },
                        "value": placement_probability
                    }
                }
            )
        )

        gauge.update_layout(
            height=400
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        st.subheader("Student Score vs Benchmark")

        benchmark_data = pd.DataFrame({
            "Area": [
                "SSC",
                "HSC",
                "Degree",
                "Employability Test",
                "MBA"
            ],

            "Student Score": [
                float(ssc_p),
                float(hsc_p),
                float(degree_p),
                float(etest_p),
                float(mba_p)
            ],

            "Benchmark": [
                65.0,
                65.0,
                65.0,
                70.0,
                62.0
            ]
        })

        benchmark_long = benchmark_data.melt(
            id_vars="Area",
            value_vars=[
                "Student Score",
                "Benchmark"
            ],
            var_name="Measure",
            value_name="Score"
        )

        benchmark_chart = px.bar(
            benchmark_long,
            x="Area",
            y="Score",
            color="Measure",
            barmode="group",
            text_auto=".1f",
            title="Student Performance Compared with Benchmark"
        )

        benchmark_chart.update_layout(
            yaxis_range=[0, 100],
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            benchmark_chart,
            use_container_width=True
        )


        st.subheader("Strength and Weakness Analysis")

        strength_data = pd.DataFrame({
            "Area": [
                "SSC",
                "HSC",
                "Degree",
                "Employability Test",
                "MBA"
            ],

            "Score": [
                float(ssc_p),
                float(hsc_p),
                float(degree_p),
                float(etest_p),
                float(mba_p)
            ]
        })

        strength_data["Status"] = np.where(
            strength_data["Score"] >= 70,
            "Strong",
            np.where(
                strength_data["Score"] >= 60,
                "Moderate",
                "Needs Improvement"
            )
        )

        strength_chart = px.bar(
            strength_data,
            x="Score",
            y="Area",
            orientation="h",
            color="Status",
            text="Score",
            title="Student Strength and Weakness Profile"
        )

        strength_chart.update_layout(
            xaxis_range=[0, 100],
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            strength_chart,
            use_container_width=True
        )


    # ========================================================
    # TAB 3 — RECOMMENDATIONS
    # ========================================================

    with tab3:

        st.subheader("Priority Gap Analysis")

        benchmark_values = {
            "SSC": (float(ssc_p), 65.0),
            "HSC": (float(hsc_p), 65.0),
            "Degree": (float(degree_p), 65.0),
            "Employability Test": (float(etest_p), 70.0),
            "MBA": (float(mba_p), 62.0)
        }

        gap_records = []

        for area, values in benchmark_values.items():

            current_score = values[0]
            benchmark_score = values[1]

            score_gap = round(
                current_score - benchmark_score,
                2
            )

            if score_gap < 0:

                if score_gap <= -10:
                    priority = "High"
                else:
                    priority = "Medium"

                gap_records.append({
                    "Priority": str(priority),
                    "Area": str(area),
                    "Current Score": f"{current_score:.2f}",
                    "Benchmark": f"{benchmark_score:.2f}",
                    "Gap": f"{score_gap:.2f}"
                })

        if workex == "No":

            gap_records.append({
                "Priority": "Medium",
                "Area": "Work Experience",
                "Current Score": "No",
                "Benchmark": "Yes",
                "Gap": "Practical exposure required"
            })

        gap_table = pd.DataFrame(gap_records)

        if gap_table.empty:

            st.success(
                "No major benchmark gaps were identified."
            )

        else:

            show_safe_table(gap_table)


        st.subheader("Personalized Recommendations")

        recommendation_records = []

        if degree_p < 65:

            recommendation_records.append({
                "Priority": "High",
                "Area": "Degree Performance",
                "Recommended Action":
                    "Improve subject knowledge and degree-level performance."
            })

        if etest_p < 70:

            recommendation_records.append({
                "Priority": "High",
                "Area": "Employability Test",
                "Recommended Action":
                    "Practice aptitude, reasoning and employability tests."
            })

        if workex == "No":

            recommendation_records.append({
                "Priority": "Medium",
                "Area": "Practical Exposure",
                "Recommended Action":
                    "Complete an internship, live project or industry assignment."
            })

        if mba_p < 62:

            recommendation_records.append({
                "Priority": "Medium",
                "Area": "MBA Performance",
                "Recommended Action":
                    "Strengthen business fundamentals and MBA subjects."
            })

        if specialisation == "Business Analytics":

            recommendation_records.append({
                "Priority": "Low",
                "Area": "Technical Skills",
                "Recommended Action":
                    "Develop Excel, SQL, Power BI and Python projects."
            })

        elif specialisation == "Mkt&Fin":

            recommendation_records.append({
                "Priority": "Low",
                "Area": "Domain Skills",
                "Recommended Action":
                    "Develop marketing and financial-analysis skills."
            })

        elif specialisation == "Mkt&HR":

            recommendation_records.append({
                "Priority": "Low",
                "Area": "Domain Skills",
                "Recommended Action":
                    "Develop HR analytics, Excel and reporting skills."
            })

        if not recommendation_records:

            recommendation_records.append({
                "Priority": "Low",
                "Area": "Placement Preparation",
                "Recommended Action":
                    "Continue interview and resume preparation."
            })

        recommendation_table = pd.DataFrame(
            recommendation_records
        )

        show_safe_table(recommendation_table)


        st.subheader("Suggested Entry-Level Roles")

        role_mapping = {
            "Business Analytics": [
                "Business Analyst Intern",
                "Junior Data Analyst",
                "Reporting Analyst",
                "MIS Analyst"
            ],

            "Mkt&Fin": [
                "Marketing Analyst",
                "Financial Analyst Trainee",
                "Sales Analyst",
                "Business Development Analyst"
            ],

            "Mkt&HR": [
                "HR Analyst",
                "Recruitment Coordinator",
                "HR Operations Associate",
                "Talent Acquisition Associate"
            ]
        }

        suggested_roles = role_mapping.get(
            specialisation,
            ["Graduate Management Trainee"]
        )

        for role in suggested_roles:
            st.write("•", role)

        st.caption(
            "Role suggestions are rule-based guidance, "
            "not machine-learning predictions."
        )


        st.subheader("Four-Week Improvement Plan")

        action_plan = pd.DataFrame({
            "Week": [
                "Week 1",
                "Week 2",
                "Week 3",
                "Week 4"
            ],

            "Action": [
                "Resume, LinkedIn and readiness review",
                "Academic, aptitude and reasoning preparation",
                "Live project or practical skill development",
                "Mock interviews and targeted applications"
            ]
        })

        show_safe_table(action_plan)


    # ========================================================
    # TAB 4 — FINAL REPORT
    # ========================================================

    with tab4:

        st.subheader("Final Placement Assessment Report")

        assessment_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )

        final_report = pd.DataFrame({
            "Field": [
                "Assessment Date",
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
                str(assessment_time),
                str(prediction),
                f"{placement_probability:.2f}%",
                str(readiness),
                str(risk_level),
                f"{academic_average:.2f}%",
                f"{employability_average:.2f}%",
                f"{academic_growth:.2f}",
                str(low_score_count),
                str(workex),
                str(specialisation)
            ]
        })

        show_safe_table(final_report)

        csv_report = final_report.astype(str).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Placement Report",
            data=csv_report,
            file_name="student_placement_report.csv",
            mime="text/csv",
            use_container_width=True
        )


else:

    st.info(
        "Enter student information in the sidebar "
        "and click Predict Placement."
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Academic decision-support application. "
    "Predictions should support, not replace, "
    "human placement decisions."
)
