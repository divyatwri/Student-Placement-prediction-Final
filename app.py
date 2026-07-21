from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 5%, rgba(79,70,229,.10), transparent 28%),
            radial-gradient(circle at 90% 5%, rgba(6,182,212,.08), transparent 25%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        color: #0f172a;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 15% 8%, rgba(124,58,237,.50), transparent 28%),
            linear-gradient(165deg, #0f172a 0%, #1e1b4b 55%, #312e81 100%);
        border-right: 1px solid rgba(255,255,255,.10);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] input {
        color: #111827 !important;
        background: #ffffff !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #111827 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        color: #111827 !important;
        fill: #111827 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button,
    section[data-testid="stSidebar"] button[kind="primary"] {
        width: 100% !important;
        min-height: 3.15rem !important;
        border: none !important;
        border-radius: 14px !important;
        background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 55%, #fb7185 100%) !important;
        color: #111827 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 26px rgba(245,158,11,.35) !important;
    }

    section[data-testid="stSidebar"] div.stButton > button p,
    section[data-testid="stSidebar"] button[kind="primary"] p {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        border-radius: 24px;
        background: linear-gradient(120deg, #111827 0%, #312e81 58%, #0369a1 100%);
        box-shadow: 0 22px 55px rgba(30,41,59,.20);
        margin-bottom: 1.2rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -110px;
        top: -120px;
        border-radius: 50%;
        background: rgba(255,255,255,.10);
    }

    .hero-badge {
        display: inline-block;
        padding: .38rem .78rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.25);
        background: rgba(255,255,255,.10);
        color: #dbeafe;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }

    .hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -.04em;
    }

    .hero p {
        max-width: 850px;
        color: #dbeafe;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: .9rem 0 0;
    }

    .glass-card {
        height: 100%;
        padding: 1.15rem 1.25rem;
        border-radius: 18px;
        background: rgba(255,255,255,.90);
        border: 1px solid rgba(148,163,184,.22);
        box-shadow: 0 12px 32px rgba(30,41,59,.08);
        backdrop-filter: blur(12px);
    }

    .glass-card h3 {
        color: #172554;
        margin: 0 0 .35rem;
        font-size: 1rem;
    }

    .glass-card p {
        color: #475569;
        margin: 0;
        line-height: 1.6;
        font-size: .90rem;
    }

    .kpi-card {
        padding: 1.1rem 1.15rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 27px rgba(15,23,42,.07);
    }

    .kpi-label {
        color: #64748b;
        font-weight: 700;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: .25rem;
    }

    .section-title {
        margin: 1.3rem 0 .8rem;
        color: #0f172a;
        font-size: 1.42rem;
        font-weight: 800;
        letter-spacing: -.02em;
    }

    .status-banner {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        margin: .4rem 0 1rem;
        font-weight: 800;
        font-size: 1.02rem;
        border: 1px solid transparent;
    }

    .status-high {
        color: #065f46;
        background: linear-gradient(90deg, #d1fae5, #ecfdf5);
        border-color: #6ee7b7;
    }

    .status-medium {
        color: #92400e;
        background: linear-gradient(90deg, #fef3c7, #fff7ed);
        border-color: #fbbf24;
    }

    .status-low {
        color: #991b1b;
        background: linear-gradient(90deg, #fee2e2, #fff1f2);
        border-color: #fca5a5;
    }

    .recommendation {
        padding: .9rem 1rem;
        margin-bottom: .7rem;
        border-radius: 14px;
        background: white;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 7px 20px rgba(15,23,42,.06);
        color: #334155;
        line-height: 1.55;
    }

    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #475569 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4338ca !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #4f46e5 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 900px) {
        .hero {padding: 1.4rem;}
        .hero h1 {font-size: 2rem;}
        .block-container {padding-left: .85rem; padding-right: .85rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================
MODEL_PATH = Path("model.pkl")

MODEL_FEATURES = [
    "gender",
    "ssc_p",
    "ssc_b",
    "hsc_p",
    "hsc_b",
    "hsc_s",
    "degree_p",
    "degree_t",
    "workex",
    "etest_p",
    "specialisation",
    "mba_p",
]

COLORS = {
    "indigo": "#4F46E5",
    "cyan": "#06B6D4",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "emerald": "#10B981",
    "slate": "#64748B",
}


# =========================================================
# MODEL HELPERS
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Keep model.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["academic_average"] = data[
        ["ssc_p", "hsc_p", "degree_p", "mba_p"]
    ].mean(axis=1)

    data["employability_average"] = data[
        ["degree_p", "etest_p", "mba_p"]
    ].mean(axis=1)

    data["academic_growth"] = data["degree_p"] - data["ssc_p"]

    score_columns = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    data["low_score_count"] = (data[score_columns] < 60).sum(axis=1).astype(int)

    return data


def get_placed_class_index(model) -> int:
    classes = None

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        if hasattr(final_estimator, "classes_"):
            classes = list(final_estimator.classes_)

    if classes:
        normalized = [str(item).strip().lower() for item in classes]
        for candidate in ["placed", "1", "yes", "true"]:
            if candidate in normalized:
                return normalized.index(candidate)

    return 1


def run_prediction(model, raw_df: pd.DataFrame):
    engineered_df = create_features(raw_df)
    prediction = str(model.predict(engineered_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(engineered_df)[0]
        probability = float(probabilities[get_placed_class_index(model)])

    if probability is None:
        probability = 0.75 if prediction.strip().lower() == "placed" else 0.25

    return prediction, probability, engineered_df


def classify_readiness(probability: float):
    if probability >= 0.75:
        return (
            "High Placement Readiness",
            "Low Risk",
            "status-high",
            "🟢",
            "The student demonstrates strong placement readiness and a competitive profile."
        )
    if probability >= 0.50:
        return (
            "Moderate Placement Readiness",
            "Medium Risk",
            "status-medium",
            "🟡",
            "The student shows moderate readiness. A few areas should be improved before placements."
        )
    return (
        "Low Placement Readiness",
        "High Risk",
        "status-low",
        "🔴",
        "The student currently needs a focused development plan before placement activities."
    )


def build_recommendations(values: dict, probability: float):
    items = []

    if values["degree_p"] < 65:
        items.append(
            "Strengthen degree-level performance through revision, applied projects and regular mock assessments."
        )

    if values["etest_p"] < 70:
        items.append(
            "Improve aptitude, logical reasoning, communication and timed employability-test practice."
        )

    if values["mba_p"] < 65:
        items.append(
            "Focus on MBA core concepts, case analysis, presentations and practical business assignments."
        )

    if values["workex"] == "No":
        items.append(
            "Gain practical exposure through an internship, live project, volunteering assignment or industry simulation."
        )

    if values["ssc_p"] < 60 or values["hsc_p"] < 60:
        items.append(
            "Build a stronger professional profile through certifications, portfolio projects and demonstrable skills."
        )

    if probability >= 0.75:
        items.append(
            "Maintain readiness through mock interviews, résumé refinement and role-specific preparation."
        )
    elif probability >= 0.50:
        items.append(
            "Follow a 30-day placement plan covering aptitude, résumé improvement, mock interviews and one portfolio project."
        )
    else:
        items.append(
            "Start a structured 60-day improvement plan and review progress weekly with a mentor."
        )

    return items[:5]


# =========================================================
# CHARTS
# =========================================================
def gauge_chart(probability: float):
    value = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 42, "color": "#0F172A"}},
            title={"text": "Placement Probability", "font": {"size": 17, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                "bar": {"color": COLORS["indigo"], "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 75], "color": "#FEF3C7"},
                    {"range": [75, 100], "color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 4},
                    "thickness": 0.78,
                    "value": value,
                },
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=25, r=25, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def radar_chart(values: dict):
    categories = [
        "SSC",
        "HSC",
        "Degree",
        "Employability",
        "MBA",
        "Experience",
    ]

    scores = [
        values["ssc_p"],
        values["hsc_p"],
        values["degree_p"],
        values["etest_p"],
        values["mba_p"],
        78 if values["workex"] == "Yes" else 42,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(79,70,229,0.20)",
            line=dict(color=COLORS["indigo"], width=3),
            marker=dict(size=7, color=COLORS["cyan"]),
            name="Student Profile",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[70] * (len(categories) + 1),
            theta=categories + [categories[0]],
            line=dict(color=COLORS["amber"], width=2, dash="dash"),
            name="Readiness Benchmark",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=50, r=50, t=40, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#334155"},
        polar=dict(
            bgcolor="rgba(255,255,255,.75)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#CBD5E1"),
            angularaxis=dict(gridcolor="#E2E8F0"),
        ),
        legend=dict(orientation="h", y=-0.13, x=0.08),
    )
    return fig


def benchmark_chart(values: dict):
    labels = ["SSC", "HSC", "Degree", "Employability Test", "MBA"]
    scores = [
        values["ssc_p"],
        values["hsc_p"],
        values["degree_p"],
        values["etest_p"],
        values["mba_p"],
    ]
    benchmark = [65, 65, 65, 70, 65]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=labels,
            x=scores,
            orientation="h",
            name="Student Score",
            marker=dict(
                color=[
                    COLORS["indigo"],
                    COLORS["cyan"],
                    COLORS["emerald"],
                    COLORS["amber"],
                    COLORS["rose"],
                ]
            ),
            text=[f"{score:.1f}%" for score in scores],
            textposition="inside",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=labels,
            x=benchmark,
            mode="markers",
            name="Suggested Benchmark",
            marker=dict(symbol="diamond", size=13, color="#0F172A"),
        )
    )

    fig.update_layout(
        height=400,
        xaxis=dict(title="Score (%)", range=[0, 100], gridcolor="#E2E8F0"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=35, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.80)",
        font={"family": "Inter", "color": "#334155"},
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def gap_chart(values: dict):
    targets = {
        "SSC": 65,
        "HSC": 65,
        "Degree": 65,
        "Employability": 70,
        "MBA": 65,
    }

    actuals = {
        "SSC": values["ssc_p"],
        "HSC": values["hsc_p"],
        "Degree": values["degree_p"],
        "Employability": values["etest_p"],
        "MBA": values["mba_p"],
    }

    gaps = {name: max(0, targets[name] - actuals[name]) for name in targets}
    labels = list(gaps.keys())
    gap_values = list(gaps.values())

    bar_colors = [
        COLORS["rose"] if gap >= 10 else
        COLORS["amber"] if gap > 0 else
        COLORS["emerald"]
        for gap in gap_values
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=gap_values,
            marker=dict(color=bar_colors),
            text=[f"{gap:.1f}" for gap in gap_values],
            textposition="outside",
        )
    )

    fig.update_layout(
        height=350,
        yaxis=dict(
            title="Improvement Gap (percentage points)",
            gridcolor="#E2E8F0",
            rangemode="tozero",
        ),
        margin=dict(l=20, r=20, t=30, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.80)",
        font={"family": "Inter", "color": "#334155"},
        showlegend=False,
    )
    return fig


def create_report_csv(values, prediction, probability, readiness, risk):
    report = pd.DataFrame(
        [{
            "Prediction": prediction,
            "Placement Probability": f"{probability * 100:.2f}%",
            "Readiness Category": readiness,
            "Risk Level": risk,
            "Academic Average": round(
                np.mean([values["ssc_p"], values["hsc_p"], values["degree_p"], values["mba_p"]]),
                2,
            ),
            "Employability Average": round(
                np.mean([values["degree_p"], values["etest_p"], values["mba_p"]]),
                2,
            ),
            "Academic Growth": round(values["degree_p"] - values["ssc_p"], 2),
            "Work Experience": values["workex"],
            "Specialisation": values["specialisation"],
        }]
    )

    buffer = io.StringIO()
    report.to_csv(buffer, index=False)
    return buffer.getvalue()


# =========================================================
# LOAD MODEL
# =========================================================
try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Upload model.pkl to the same GitHub folder as app.py and then reboot the Streamlit app."
    )
    st.stop()


# =========================================================
# SIDEBAR FORM
# =========================================================
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.caption("Enter academic and employability information.")

    with st.form("student_form"):
        st.markdown("### Personal & School Profile")

        gender = st.selectbox(
            "Gender",
            ["M", "F"],
            format_func=lambda value: "Male" if value == "M" else "Female",
        )

        ssc_p = st.number_input(
            "SSC Percentage",
            0.0,
            100.0,
            65.0,
            0.5,
        )

        ssc_b = st.selectbox(
            "SSC Board",
            ["Central", "Others"],
        )

        hsc_p = st.number_input(
            "HSC Percentage",
            0.0,
            100.0,
            65.0,
            0.5,
        )

        hsc_b = st.selectbox(
            "HSC Board",
            ["Central", "Others"],
        )

        hsc_s = st.selectbox(
            "HSC Stream",
            ["Commerce", "Science", "Arts"],
        )

        st.markdown("### Higher Education & Employability")

        degree_p = st.number_input(
            "Degree Percentage",
            0.0,
            100.0,
            65.0,
            0.5,
        )

        degree_t = st.selectbox(
            "Degree Type",
            ["Comm&Mgmt", "Sci&Tech", "Others"],
            format_func=lambda value: {
                "Comm&Mgmt": "Commerce & Management",
                "Sci&Tech": "Science & Technology",
                "Others": "Other",
            }[value],
        )

        workex = st.selectbox(
            "Work Experience",
            ["Yes", "No"],
        )

        etest_p = st.number_input(
            "Employability Test Percentage",
            0.0,
            100.0,
            70.0,
            0.5,
        )

        specialisation = st.selectbox(
            "Specialisation",
            ["Mkt&Fin", "Mkt&HR", "Business Analytics"],
            format_func=lambda value: {
                "Mkt&Fin": "Marketing & Finance",
                "Mkt&HR": "Marketing & HR",
                "Business Analytics": "Business Analytics",
            }[value],
        )

        mba_p = st.number_input(
            "MBA Percentage",
            0.0,
            100.0,
            62.0,
            0.5,
        )

        submitted = st.form_submit_button(
            "Predict Placement",
            type="primary",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "This is a decision-support tool. Final placement decisions should include human review."
    )


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Machine Learning • Placement Intelligence</div>
        <h1>Student Placement Prediction System</h1>
        <p>
            Evaluate placement probability, readiness level, student strengths,
            improvement gaps and actionable recommendations using a trained
            Random Forest model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="glass-card">
            <h3>🎯 Business Objective</h3>
            <p>Identify placement readiness early and provide targeted student support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="glass-card">
            <h3>🧠 ML Decision Engine</h3>
            <p>Uses academic, educational, employability and experience-related information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="glass-card">
            <h3>📈 Actionable Output</h3>
            <p>Provides prediction, probability, risk level, analytics and recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PREPARE INPUT
# =========================================================
input_values = {
    "gender": gender,
    "ssc_p": float(ssc_p),
    "ssc_b": ssc_b,
    "hsc_p": float(hsc_p),
    "hsc_b": hsc_b,
    "hsc_s": hsc_s,
    "degree_p": float(degree_p),
    "degree_t": degree_t,
    "workex": workex,
    "etest_p": float(etest_p),
    "specialisation": specialisation,
    "mba_p": float(mba_p),
}

raw_input_df = pd.DataFrame([input_values], columns=MODEL_FEATURES)


# =========================================================
# INITIAL SCREEN
# =========================================================
if not submitted:
    st.markdown(
        """
        <div style="
            margin-top:1.2rem;
            padding:1.1rem 1.2rem;
            border-radius:16px;
            background:linear-gradient(90deg,#e0f2fe,#eef2ff);
            border:1px solid #bfdbfe;
            color:#1e3a8a;
            font-weight:700;">
            👈 Enter the student's details in the sidebar and click
            <b>Predict Placement</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">What this application provides</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)
    cards = [
        ("01", "Placement Probability", "Model-generated probability of placement."),
        ("02", "Readiness Category", "High, moderate or low placement readiness."),
        ("03", "Visual Analytics", "Profile, benchmark and improvement-gap charts."),
        ("04", "Development Plan", "Personalised and actionable recommendations."),
    ]

    for column, (number, title, description) in zip([d1, d2, d3, d4], cards):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="color:#4f46e5;font-weight:800;font-size:.82rem;">{number}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.stop()


# =========================================================
# PREDICT
# =========================================================
try:
    with st.spinner("Analysing the student profile..."):
        prediction, probability, engineered_df = run_prediction(
            model,
            raw_input_df,
        )
except Exception as error:
    st.error("Prediction could not be completed.")
    st.exception(error)
    st.info(
        "Make sure your model.pkl was trained using the same feature names, "
        "category spellings and engineered features."
    )
    st.stop()


readiness, risk_level, status_class, icon, summary_text = classify_readiness(
    probability
)

recommendations = build_recommendations(input_values, probability)

academic_average = np.mean(
    [ssc_p, hsc_p, degree_p, mba_p]
)

employability_average = np.mean(
    [degree_p, etest_p, mba_p]
)

academic_growth = degree_p - ssc_p

low_score_count = sum(
    score < 60
    for score in [ssc_p, hsc_p, degree_p, etest_p, mba_p]
)


# =========================================================
# RESULTS HEADER
# =========================================================
st.markdown(
    f"""
    <div class="status-banner {status_class}">
        {icon} {readiness}
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

kpis = [
    ("Prediction", prediction),
    ("Placement Probability", f"{probability * 100:.2f}%"),
    ("Readiness Level", readiness.replace(" Placement Readiness", "")),
    ("Risk Level", risk_level),
]

for column, (label, value) in zip([k1, k2, k3, k4], kpis):
    with column:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# REPORT TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Visual Analytics",
        "Recommendations",
        "Final Report",
    ]
)


with tab1:
    st.markdown(
        '<div class="section-title">Executive Summary</div>',
        unsafe_allow_html=True,
    )

    st.write(summary_text)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.plotly_chart(
            gauge_chart(probability),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Prediction",
                    "Placement Probability",
                    "Readiness Category",
                    "Risk Level",
                    "Academic Average",
                    "Employability Average",
                    "Academic Growth",
                    "Low Score Count",
                ],
                "Value": [
                    prediction,
                    f"{probability * 100:.2f}%",
                    readiness,
                    risk_level,
                    f"{academic_average:.2f}%",
                    f"{employability_average:.2f}%",
                    f"{academic_growth:.2f}",
                    int(low_score_count),
                ],
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    st.markdown(
        """
        <div class="glass-card">
            <h3>How to interpret the result</h3>
            <p>
                The probability is a model-based estimate, not a guarantee.
                It should be interpreted together with the student's complete
                profile and human judgement from the placement team.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab2:
    st.markdown(
        '<div class="section-title">Student Profile Analytics</div>',
        unsafe_allow_html=True,
    )

    chart1, chart2 = st.columns(2, gap="large")

    with chart1:
        st.markdown("#### Multi-Dimensional Readiness Profile")
        st.plotly_chart(
            radar_chart(input_values),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with chart2:
        st.markdown("#### Score vs Suggested Benchmark")
        st.plotly_chart(
            benchmark_chart(input_values),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("#### Priority Improvement Gaps")
    st.plotly_chart(
        gap_chart(input_values),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    score_map = {
        "SSC": ssc_p,
        "HSC": hsc_p,
        "Degree": degree_p,
        "Employability Test": etest_p,
        "MBA": mba_p,
    }

    strongest_area = max(score_map, key=score_map.get)
    development_area = min(score_map, key=score_map.get)

    a1, a2, a3 = st.columns(3)
    insights = [
        ("Strongest Area", strongest_area),
        ("Priority Development Area", development_area),
        ("Experience Advantage", "Available" if workex == "Yes" else "Not Yet Available"),
    ]

    for column, (label, value) in zip([a1, a2, a3], insights):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:1.18rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with tab3:
    st.markdown(
        '<div class="section-title">Personalised Recommendations</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "These recommendations are based on academic scores, employability score, "
        "work experience and the predicted readiness level."
    )

    for index, recommendation in enumerate(recommendations, start=1):
        st.markdown(
            f"""
            <div class="recommendation">
                <b>{index}. Development Action</b><br>
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Suggested 4-Week Action Plan")

    plan_df = pd.DataFrame(
        {
            "Timeline": ["Week 1", "Week 2", "Week 3", "Week 4"],
            "Primary Focus": [
                "Profile diagnosis and goal setting",
                "Aptitude and employability preparation",
                "Résumé, LinkedIn and portfolio improvement",
                "Mock interviews and final assessment",
            ],
            "Expected Output": [
                "Clear improvement priorities",
                "Improved speed and accuracy",
                "Stronger professional profile",
                "Interview-readiness report",
            ],
        }
    )

    st.dataframe(
        plan_df,
        use_container_width=True,
        hide_index=True,
    )


with tab4:
    st.markdown(
        '<div class="section-title">Final Placement Readiness Report</div>',
        unsafe_allow_html=True,
    )

    report_left, report_right = st.columns([1.2, 1], gap="large")

    with report_left:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>Decision Summary</h3>
                <p><b>Model Prediction:</b> {prediction}</p>
                <p><b>Placement Probability:</b> {probability * 100:.2f}%</p>
                <p><b>Readiness Category:</b> {readiness}</p>
                <p><b>Risk Level:</b> {risk_level}</p>
                <p style="margin-top:.65rem;">{summary_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Input Profile")

        profile_df = pd.DataFrame(
            {
                "Profile Attribute": [
                    "Gender",
                    "SSC",
                    "HSC",
                    "HSC Stream",
                    "Degree",
                    "Degree Type",
                    "Work Experience",
                    "Employability Test",
                    "Specialisation",
                    "MBA",
                ],
                "Value": [
                    "Male" if gender == "M" else "Female",
                    f"{ssc_p:.2f}%",
                    f"{hsc_p:.2f}%",
                    hsc_s,
                    f"{degree_p:.2f}%",
                    degree_t,
                    workex,
                    f"{etest_p:.2f}%",
                    specialisation,
                    f"{mba_p:.2f}%",
                ],
            }
        )

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True,
        )

    with report_right:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Responsible Use Statement</h3>
                <p>
                    This result is generated from historical patterns. It must not
                    be used as the sole basis for rejection, ranking or exclusion.
                    Final decisions should include human review and student context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Download Report")

        csv_data = create_report_csv(
            input_values,
            prediction,
            probability,
            readiness,
            risk_level,
        )

        st.download_button(
            "Download Result as CSV",
            data=csv_data,
            file_name="student_placement_readiness_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("#### Technical Information")
        st.info(
            "Backend: Scikit-learn pipeline\n\n"
            "Model: Tuned Random Forest classifier\n\n"
            "Frontend: Streamlit\n\n"
            "Outputs: Prediction, probability, readiness, analytics and recommendations"
        )


st.markdown("---")
st.caption(
    "Academic decision-support application • The output should support, not replace, human placement decisions."
)
