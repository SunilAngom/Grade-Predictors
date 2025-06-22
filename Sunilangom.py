import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# Page config
st.set_page_config(page_title="Student Final Grade Predictor", page_icon="🎓")
st.title("🎓 Students Grade Predictor")
st.subheader("Developed by Sunil Angom")

# Optional banner image
try:
    image = Image.open("student_banner.jpg")  # Replace with your image filename
    st.image(image, use_column_width=True)
except:
    st.warning("📷 'student_banner.jpg' not found. You can add it to enhance the UI.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
parental_education = st.selectbox("Parental Education", ["High School", "Bachelors", "Masters"])
hours_of_study = st.slider("Hours of Study", 0.0, 10.0, 2.0)
attendance_percentage = st.slider("Attendance %", 0.0, 100.0, 75.0)
extra_curricular_participation = st.selectbox("Extra Curricular Participation", ["Yes", "No"])
internet_access_home = st.selectbox("Internet Access at Home", ["Yes", "No"])
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
study_method = st.selectbox("Study Method", ["Self Study", "Group Study", "Coaching"])
class_participation = st.selectbox("Class Participation", ["Low", "Moderate", "High"])
family_support = st.selectbox("Family Support", ["Yes", "No"])
previous_grade = st.slider("Previous Grade", 0.0, 100.0, 70.0)

# Suggestion logic
def get_suggestion(grade):
    if grade >= 90:
        return "🌟 Excellent: Keep up the great work and aim for consistency!"
    elif grade >= 75:
        return "✅ Good: Try to improve weak areas to reach excellence."
    elif grade >= 60:
        return "📘 Average: Focus on difficult subjects and manage time wisely."
    elif grade >= 45:
        return "⚠️ Below Average: Increase study hours and seek help if needed."
    else:
        return "❗ Poor: Consider personalized coaching or extra tutoring."

# Predict button
if st.button("Predict Final Grade"):
    input_data = pd.DataFrame([[
        gender, parental_education, hours_of_study, attendance_percentage,
        extra_curricular_participation, internet_access_home, part_time_job,
        study_method, class_participation, family_support, previous_grade
    ]], columns=[
        "gender", "parental_education", "hours_of_study", "attendance_percentage",
        "extra_curricular_participation", "internet_access_home", "part_time_job",
        "study_method", "class_participation", "family_support", "previous_grade"
    ])

    try:
        # Encode only categorical columns
        categorical_cols = [
            "gender", "parental_education", "extra_curricular_participation",
            "internet_access_home", "part_time_job", "study_method",
            "class_participation", "family_support"
        ]

        for col in categorical_cols:
            input_data[col] = encoders[col].transform([input_data[col][0]])

        # Predict final grade
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Final Grade: {prediction:.2f}")

        # Show suggestion
        suggestion = get_suggestion(prediction)
        st.info(f"💡 Suggestion: {suggestion}")

    except Exception as e:
        st.error(f"Encoding or prediction failed: {e}")
