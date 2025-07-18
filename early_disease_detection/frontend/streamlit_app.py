import streamlit as st
import requests
import pandas as pd
import json
from typing import Optional

# --- Session State for role management ---
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# --- Backend API endpoints ---
API_URL = "http://localhost:8000"

# --- Modular components (placeholders) ---
def login_component():
    st.subheader("Doctor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # TODO: Replace with secure backend auth
        if username == "doctor" and password == "password":
            st.session_state['role'] = 'doctor'
            st.session_state['username'] = username
            st.success("Logged in as Doctor.")
        else:
            st.error("Invalid credentials.")

def upload_component():
    st.subheader("Upload Patient Data")
    uploaded_file = st.file_uploader("Choose a data file (CSV/Excel/JSON)")
    if uploaded_file:
        file_type = uploaded_file.type
        if file_type == 'text/csv':
            df = pd.read_csv(uploaded_file)
        elif file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            df = pd.read_excel(uploaded_file)
        elif file_type == 'application/json':
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return
        st.write("Preview:", df.head())
        if st.button("Get Prediction"):
            # Send to FastAPI backend
            data = df.values.tolist()
            response = requests.post(f"{API_URL}/predict", json={"data": data})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['predictions']}")
                # Get explanation
                explain = requests.post(f"{API_URL}/explain", json={"data": data})
                if explain.status_code == 200:
                    st.info(f"Explanation: {explain.json().get('message', 'N/A')}")
            else:
                st.error(f"Prediction failed: {response.text}")

# --- Doctor dashboard component ---
def doctor_dashboard():
    st.title("Doctor Dashboard")
    st.write(f"Welcome, {st.session_state['username']}")
    # Placeholder: Fetch patient history from backend
    st.info("Patient history and confidence scores coming soon.")
    # Placeholder: SHAP plots
    st.info("SHAP plots for model explainability coming soon.")
    if st.button("Logout"):
        st.session_state['role'] = None
        st.session_state['username'] = ''

# --- Patient view ---
def patient_view():
    st.title("Early Disease Detection - Patient Portal")
    upload_component()
    # Placeholder for LLM symptom checker
    st.info("LLM-based symptom checker coming soon.")

# --- Main app logic ---
def main():
    st.set_page_config(page_title="Early Disease Detection", layout="centered")
    st.sidebar.title("Navigation")
    role = st.sidebar.radio("Select Role", ("Patient", "Doctor"))
    if role == "Doctor" and st.session_state['role'] != 'doctor':
        login_component()
    elif role == "Doctor" and st.session_state['role'] == 'doctor':
        doctor_dashboard()
    else:
        patient_view()

if __name__ == "__main__":
    main()
