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

    .career-shell {margin:1.1rem 0;padding:1.35rem;border-radius:22px;background:linear-gradient(135deg,#ffffff 0%,#eef2ff 100%);border:1px solid #c7d2fe;box-shadow:0 14px 34px rgba(79,70,229,.10);}
    .career-title {color:#312e81;font-size:1.35rem;font-weight:800;margin-bottom:.25rem;}
    .career-subtitle {color:#64748b;line-height:1.6;margin:0;}
    .career-score-card {padding:1.25rem;border-radius:20px;background:linear-gradient(135deg,#111827 0%,#312e81 100%);color:#fff;box-shadow:0 16px 34px rgba(30,41,59,.18);text-align:center;}
    .career-score-value {font-size:2.5rem;line-height:1;font-weight:800;margin-top:.25rem;}
    .career-score-label {color:#c7d2fe;font-weight:700;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;}
    .career-level {display:inline-block;margin-top:.75rem;padding:.38rem .7rem;border-radius:999px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);color:#fff;font-size:.78rem;font-weight:800;}
    .career-insight {padding:.85rem .95rem;border-radius:14px;background:#fff;border:1px solid #e2e8f0;box-shadow:0 7px 18px rgba(15,23,42,.05);margin-bottom:.65rem;}
    .career-insight-title {color:#0f172a;font-weight:800;margin-bottom:.18rem;}
    .career-insight-copy {color:#64748b;font-size:.88rem;line-height:1.55;margin:0;}
    .source-note {padding:.85rem 1rem;border-radius:14px;background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;font-size:.86rem;line-height:1.55;margin:.7rem 0;}

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



def calculate_career_readiness(career: dict):
    components = {
        "Internships": min(career["internships"], 3) / 3 * 15,
        "Certifications": min(career["certifications"], 5) / 5 * 10,
        "Projects": min(career["projects"], 5) / 5 * 15,
        "Technical Skills": min(len(career["technical_skills"]), 6) / 6 * 15,
        "Communication": career["communication_rating"] / 5 * 10,
        "Resume": {"Basic": 4, "Good": 7, "Excellent": 10}[career["resume_quality"]],
        "Mock Interview": career["mock_interview_score"] / 100 * 10,
        "LinkedIn": 5 if career["linkedin_profile"] == "Yes" else 0,
        "GitHub": 5 if career["github_profile"] == "Yes" else 0,
        "Leadership": min(career["leadership_activities"], 3) / 3 * 5,
    }
    score = round(min(sum(components.values()), 100), 2)
    if score >= 80:
        level, risk = "Excellent Career Readiness", "Low Career Risk"
    elif score >= 65:
        level, risk = "Strong Career Readiness", "Moderate-Low Career Risk"
    elif score >= 50:
        level, risk = "Developing Career Readiness", "Moderate Career Risk"
    else:
        level, risk = "Needs Career Development", "High Career Risk"
    return score, level, risk, components


def build_career_recommendations(career: dict):
    items=[]
    if career["internships"] == 0: items.append("Complete at least one internship or live industry project to build practical exposure.")
    if career["certifications"] < 2: items.append("Complete two role-relevant certifications aligned with the target career path.")
    if career["projects"] < 2: items.append("Build at least two portfolio projects that demonstrate practical problem-solving.")
    if len(career["technical_skills"]) < 3: items.append("Strengthen the technical skill portfolio with at least three job-relevant tools or technologies.")
    if career["communication_rating"] < 4: items.append("Improve communication through presentations, group discussions and interview practice.")
    if career["resume_quality"] == "Basic": items.append("Redesign the resume with quantified achievements, project outcomes and role-specific keywords.")
    if career["mock_interview_score"] < 70: items.append("Complete regular mock interviews and track improvements in confidence and answer quality.")
    if career["linkedin_profile"] == "No": items.append("Create and optimise a LinkedIn profile with a clear headline, skills, projects and experience.")
    if career["github_profile"] == "No" and career["technical_skills"]: items.append("Create a GitHub portfolio and upload selected analytics, coding or machine-learning projects.")
    if career["leadership_activities"] == 0: items.append("Join a leadership, club, volunteering or event-management activity to demonstrate initiative.")
    if not items: items.append("The career profile is strong. Focus on company-specific preparation, networking and interview conversion.")
    return items[:6]


def career_breakdown_chart(breakdown: dict):
    labels=list(breakdown.keys()); values=list(breakdown.values())
    palette=["#4F46E5","#06B6D4","#10B981","#F59E0B","#F43F5E","#8B5CF6","#0EA5E9","#14B8A6","#6366F1","#EC4899"]
    fig=go.Figure(go.Bar(x=values,y=labels,orientation="h",marker=dict(color=palette[:len(values)]),text=[f"{v:.1f}" for v in values],textposition="outside"))
    fig.update_layout(height=470,xaxis=dict(title="Readiness Contribution",range=[0,16],gridcolor="#E2E8F0"),yaxis=dict(autorange="reversed"),margin=dict(l=20,r=35,t=25,b=35),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.82)",font={"family":"Inter","color":"#334155"},showlegend=False)
    return fig


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

        st.markdown("### Career Readiness Profile")
        st.caption("These fields improve the career assessment, not the trained ML prediction.")
        internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1, step=1)
        certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=1, step=1)
        projects = st.number_input("Projects Completed", min_value=0, max_value=20, value=2, step=1)
        technical_skills = st.multiselect("Technical Skills", ["Excel","SQL","Power BI","Tableau","Python","R","Machine Learning","Data Visualization","Statistics","Communication"], default=["Excel","SQL","Power BI"])
        communication_rating = st.slider("Communication Rating",1,5,3)
        resume_quality = st.selectbox("Resume Quality", ["Basic","Good","Excellent"], index=1)
        mock_interview_score = st.slider("Mock Interview Score",0,100,65,5)
        linkedin_profile = st.selectbox("LinkedIn Profile", ["Yes","No"], index=0)
        github_profile = st.selectbox("GitHub Profile", ["Yes","No"], index=1)
        leadership_activities = st.number_input("Leadership / Extracurricular Activities", min_value=0, max_value=10, value=1, step=1)

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

career_values = {
    "internships": int(internships),
    "certifications": int(certifications),
    "projects": int(projects),
    "technical_skills": technical_skills,
    "communication_rating": int(communication_rating),
    "resume_quality": resume_quality,
    "mock_interview_score": int(mock_interview_score),
    "linkedin_profile": linkedin_profile,
    "github_profile": github_profile,
    "leadership_activities": int(leadership_activities),
}


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

career_score, career_level, career_risk, career_breakdown = calculate_career_readiness(career_values)
career_recommendations = build_career_recommendations(career_values)


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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Executive Summary",
        "Visual Analytics",
        "Recommendations",
        "Career Readiness Analyzer",
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
    st.markdown('<div class="section-title">Career Readiness Analyzer</div>', unsafe_allow_html=True)
    st.markdown("<div class='source-note'><b>Important:</b> This is a transparent rule-based career assessment. These extra fields are not used by the trained ML model because they were not present in the original training dataset.</div>", unsafe_allow_html=True)
    score_col, insight_col = st.columns([0.8,1.2], gap="large")
    with score_col:
        st.markdown(f"<div class='career-score-card'><div class='career-score-label'>Career Readiness Score</div><div class='career-score-value'>{career_score:.0f}/100</div><div class='career-level'>{career_level}</div></div>", unsafe_allow_html=True)
        st.markdown("#### Career Risk")
        st.info(career_risk)
    with insight_col:
        st.markdown("<div class='career-shell'><div class='career-title'>Professional Profile Assessment</div><p class='career-subtitle'>This complements the ML prediction by evaluating practical exposure, professional branding, technical preparation and interview readiness.</p></div>", unsafe_allow_html=True)
        for label, value in [("Internships",f"{career_values['internships']} completed"),("Certifications",f"{career_values['certifications']} completed"),("Projects",f"{career_values['projects']} completed"),("Technical Skills",f"{len(career_values['technical_skills'])} selected"),("Communication",f"{career_values['communication_rating']}/5"),("Mock Interview",f"{career_values['mock_interview_score']}/100")]:
            st.markdown(f"<div class='career-insight'><div class='career-insight-title'>{label}</div><p class='career-insight-copy'>{value}</p></div>", unsafe_allow_html=True)
    st.markdown("#### Career Readiness Contribution")
    st.plotly_chart(career_breakdown_chart(career_breakdown), use_container_width=True, config={"displayModeBar":False})
    st.markdown("#### Career Development Recommendations")
    for index, recommendation in enumerate(career_recommendations, start=1):
        st.markdown(f"<div class='recommendation'><b>{index}. Career Action</b><br>{recommendation}</div>", unsafe_allow_html=True)

with tab5:
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

        st.markdown("#### Combined Decision View")
        combined_df = pd.DataFrame({"Assessment":["ML Placement Prediction","ML Placement Probability","ML Readiness Category","Career Readiness Score","Career Readiness Level","Career Risk"],"Result":[prediction,f"{probability*100:.2f}%",readiness,f"{career_score:.2f}/100",career_level,career_risk]})
        st.dataframe(combined_df,use_container_width=True,hide_index=True)
        st.markdown("#### Download Report")

        full_report = pd.DataFrame([{"Prediction":prediction,"Placement Probability":f"{probability*100:.2f}%","ML Readiness Category":readiness,"ML Risk Level":risk_level,"Academic Average":round(academic_average,2),"Employability Average":round(employability_average,2),"Academic Growth":round(academic_growth,2),"Low Score Count":int(low_score_count),"Internships":career_values["internships"],"Certifications":career_values["certifications"],"Projects":career_values["projects"],"Technical Skills":", ".join(career_values["technical_skills"]),"Communication Rating":career_values["communication_rating"],"Resume Quality":career_values["resume_quality"],"Mock Interview Score":career_values["mock_interview_score"],"LinkedIn Profile":career_values["linkedin_profile"],"GitHub Profile":career_values["github_profile"],"Leadership Activities":career_values["leadership_activities"],"Career Readiness Score":career_score,"Career Readiness Level":career_level,"Career Risk":career_risk}])
        csv_data=full_report.to_csv(index=False)
        st.download_button("Download Complete Result as CSV",data=csv_data,file_name="student_placement_and_career_readiness_report.csv",mime="text/csv",use_container_width=True)

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
