import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('student_performance_model.pkl')

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Math Score Predictor")
st.write("Enter student details to predict **Math Score**")

# ---- Categorical Inputs ----
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A","group B","group C","group D","group E"])
parent_edu = st.selectbox("Parental Education", [
    "some high school","high school","some college",
    "associate's degree","bachelor's degree","master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# ---- Numerical Inputs ----
reading = st.number_input("Reading Score (0-100)", min_value=0, max_value=100, value=60)
writing = st.number_input("Writing Score (0-100)", min_value=0, max_value=100, value=60)

# ---- Predict Button ----
if st.button("Predict Math Score"):
    input_df = pd.DataFrame({
        'gender':[gender],
        'race/ethnicity':[race],
        'parental level of education':[parent_edu],
        'lunch':[lunch],
        'test preparation course':[prep],
        'reading score':[reading],
        'writing score':[writing]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"📊 Predicted Math Score: **{round(prediction,2)}**")

    if prediction >= 50:
        st.balloons()
        st.write("✅ Status: PASS")
    else:
        st.write("❌ Status: FAIL")
