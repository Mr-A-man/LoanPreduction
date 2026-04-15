import streamlit as st
import numpy as np
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(current_dir, "model.pkl"))
model = bundle['model']
imputer = bundle['imputer']
feature_cols = bundle['feature_cols']
threshold = bundle.get('threshold', 0.30)

st.set_page_config(page_title="Loan Predictor")
st.title("Loan Prediction")
st.caption("Powered by Aman kumar (CSDS-20)")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, value=35)
    income = st.number_input("Annual Income (₹)", min_value=0, value=60000, step=5000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=20000, step=1000)
    credit_score = st.number_input("Credit Score", 300, 850, value=650)

with col2:
    employment_years = st.number_input("Employment Years", 0, 50, value=5)
    education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    housing = st.selectbox("Housing Status", ["Own", "Rent", "Mortgage"])

edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

if st.button("Predict Loan Risk", use_container_width=True):
    input_dict = {
        'Age': age,
        'Income': income,
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Employment_Years': employment_years,
        'Education_Level': edu_map[education],
        'Housing_Own': 1 if housing == 'Own' else 0,
        'Housing_Rent': 1 if housing == 'Rent' else 0,
        'Housing_Mortgage': 1 if housing == 'Mortgage' else 0,
    }

    input_arr = np.array([[input_dict[col] for col in feature_cols]])
    input_imputed = imputer.transform(input_arr)

    prob_default = model.predict_proba(input_imputed)[0][1]
    is_default = prob_default >= threshold

    st.divider()
    if is_default:
        st.error(f"High Risk")
        st.metric("Default Probability", f"{prob_default:.1%}")
        st.warning("Loan not recommended based on current profile.")
    else:
        st.success(f"Low Risk")
        st.metric("Approval", f"{(1 - prob_default):.1%}")

    # Risk bar
    st.progress(float(prob_default), text=f"Risk Score: {prob_default:.1%}")

