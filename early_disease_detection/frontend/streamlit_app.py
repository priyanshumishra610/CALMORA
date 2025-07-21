import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import Any, List
import base64
import io

# --- Load environment variables ---
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Tailwind CSS (via CDN) ---
TAILWIND_CDN = "https://cdn.tailwindcss.com"
st.markdown(f"<script src=\"{TAILWIND_CDN}\"></script>", unsafe_allow_html=True)

# --- Session State for Auth ---
if 'jwt_token' not in st.session_state:
    st.session_state['jwt_token'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# --- Auth Functions ---
def login(username, password):
    try:
        response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state['jwt_token'] = token
            st.session_state['username'] = username
            # Get role from username (simple logic)
            st.session_state['role'] = 'doctor' if username == 'doctor' else 'patient'
            st.success(f"Logged in as {username}")
        else:
            st.error("Login failed. Check credentials.")
    except Exception as e:
        st.error(f"Login error: {e}")

def logout():
    st.session_state['jwt_token'] = None
    st.session_state['role'] = None
    st.session_state['username'] = ''

# --- Patient Upload & Prediction ---
def patient_view():
    st.title("🩺 Early Disease Detection - Patient Portal")
    uploaded_file = st.file_uploader("Upload your health data (CSV)")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        if st.button("Get Prediction"):
            data = df.values.tolist()
            headers = {"Authorization": f"Bearer {st.session_state['jwt_token']}"} if st.session_state['jwt_token'] else {}
            try:
                response = requests.post(f"{API_URL}/predict", json={"data": data}, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['predictions']}")
                    if result.get('confidences'):
                        st.info(f"Confidence scores: {result['confidences']}")
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    # LLM Symptom Checker Placeholder
    st.info("🤖 LLM-based symptom checker coming soon.")

# --- Doctor Dashboard ---
def doctor_view():
    st.title("👨‍⚕️ Doctor Dashboard")
    st.write(f"Welcome, Dr. {st.session_state['username']}")
    # Patient data upload and prediction
    uploaded_file = st.file_uploader("Upload patient data (CSV)")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        if st.button("Get Prediction & SHAP Explanation"):
            data = df.values.tolist()
            headers = {"Authorization": f"Bearer {st.session_state['jwt_token']}"}
            try:
                pred_resp = requests.post(f"{API_URL}/predict", json={"data": data}, headers=headers)
                explain_resp = requests.post(f"{API_URL}/explain", json={"data": data}, headers=headers)
                if pred_resp.status_code == 200:
                    result = pred_resp.json()
                    st.success(f"Prediction: {result['predictions']}")
                    if result.get('confidences'):
                        st.info(f"Confidence scores: {result['confidences']}")
                else:
                    st.error(f"Prediction failed: {pred_resp.text}")
                if explain_resp.status_code == 200:
                    explain = explain_resp.json()
                    st.write("### SHAP Values (first row):")
                    st.json(explain['shap_values'][0])
                    st.write("### Feature Names:")
                    st.json(explain['feature_names'])
                    # SHAP plot placeholder (could render image if backend returns it)
                    st.info("SHAP summary plot coming soon.")
                else:
                    st.error(f"Explain failed: {explain_resp.text}")
            except Exception as e:
                st.error(f"Doctor prediction/explain error: {e}")
    # Patient history placeholder
    st.info("Patient history and SHAP plots coming soon.")
    if st.button("Logout"):
        logout()

# --- Login Form ---
def login_form():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="Early Disease Detection", layout="centered")
    st.sidebar.title("Navigation")
    if st.session_state['jwt_token']:
        st.sidebar.success(f"Logged in as {st.session_state['username']} ({st.session_state['role']})")
        if st.sidebar.button("Logout"):
            logout()
    role = st.sidebar.radio("Select Role", ("Patient", "Doctor"))
    if not st.session_state['jwt_token']:
        login_form()
    elif role == "Doctor" and st.session_state['role'] == 'doctor':
        doctor_view()
    elif role == "Patient" and st.session_state['role'] == 'patient':
        patient_view()
    else:
        st.warning("You do not have access to this view. Please login with the correct role.")

st.title("Calmora: Symptom Checker")
st.markdown("Enter your symptoms below. Calm, explainable, and actionable health insights.")

with st.form("symptom_form"):
    user_text = st.text_area("Describe your symptoms", "")
    consent = st.checkbox("I accept this is not medical advice and agree to the consent policy.", value=False)
    st.markdown('[Terms of Service](../../TERMS.md) | [Privacy Policy](../../PRIVACY.md)', unsafe_allow_html=True)
    submitted = st.form_submit_button("Check My Risk")

if submitted:
    if not consent:
        st.error("You must accept the consent policy to proceed.")
    elif not user_text.strip():
        st.error("Please enter your symptoms.")
    else:
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/symptoms",
                json={"text": user_text},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                st.success(data.get("message", "All good!"))
                # Display risk predictions
                if data.get("risk"):
                    st.subheader("Risk Predictions")
                    for r in data["risk"]:
                        st.write(f"{r['disease']}: {r['risk_score']:.2f}")
                # Display SHAP explanation
                if data.get("shap") and data["shap"].get("plot_base64"):
                    st.subheader("Why? (Explainability)")
                    img_bytes = base64.b64decode(data["shap"]["plot_base64"])
                    st.image(io.BytesIO(img_bytes), caption="SHAP Explanation")
                elif data.get("shap") and data["shap"].get("shap_values"):
                    st.subheader("Why? (Explainability)")
                    st.write(f"SHAP values: {data['shap']['shap_values']}")
                # Display lifestyle tips
                if data.get("lifestyle"):
                    st.subheader("Lifestyle & Health Tips")
                    for tip in data["lifestyle"]:
                        st.info(tip["tip"])
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Exception: {str(e)}")

# TODO: Add loading spinner, error boundary, and improved UI/UX

if __name__ == "__main__":
    main()
