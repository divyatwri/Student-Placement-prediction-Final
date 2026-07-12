# Student Placement Prediction System

## Project Overview

The Student Placement Prediction System is an end-to-end machine learning project that predicts whether a student is likely to be placed or not based on academic, specialization, work experience and employability-related information.

The project is designed as a Placement Intelligence Dashboard, not only as a prediction tool. It also provides placement probability, readiness category, student strengths, risk areas and personalized recommendations.

## Business Objective

The objective of this project is to help placement cells identify students who are placement-ready and students who may require additional support before campus recruitment.

The system supports data-driven decision-making by analyzing student academic performance, employability test score, specialization and work experience.

## Dataset

The dataset used in this project is a synthetic but real-world-inspired student placement dataset. It contains academic and employability-related fields such as:

- Gender
- SSC Percentage
- HSC Percentage
- Degree Percentage
- Work Experience
- Employability Test Score
- Specialisation
- MBA Percentage
- Placement Status
- Salary

Salary is not used for model training because it is available only after placement and may cause data leakage.

## Project Phases

### Phase 1: Business Understanding
Defined the placement prediction problem, objective, business impact and college/HR analytics use case.

### Phase 2: Data Understanding and EDA
Performed data inspection, missing value analysis, target variable analysis, score distributions, placement rate analysis and correlation analysis.

### Phase 3: Data Preprocessing
Handled missing values, encoded categorical variables and scaled numerical variables.

### Phase 4: Feature Engineering
Created business-focused features such as academic average, employability average, academic growth and low score count.

### Phase 5: Model Building
Trained and compared multiple machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest

### Phase 6: Model Optimization
Optimized the best-performing model and saved the final model as `model.pkl`.

### Phase 7: Deployment
Deployed the final model using Streamlit Community Cloud.

## App Features

- Student placement prediction
- Placement probability score
- Readiness category
- Risk level
- Student score vs benchmark graph
- Strength and weakness analysis
- Personalized recommendations
- Downloadable prediction report

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Plotly

## Final Deliverables

- Jupyter Notebook
- Trained model file: `model.pkl`
- Streamlit app file: `app.py`
- Requirements file: `requirements.txt`
- GitHub repository
- Streamlit deployed app link

## Disclaimer

This project is created for academic and learning purposes. The prediction should support placement decision-making and should not replace human judgement.