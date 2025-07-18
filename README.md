# 🚑 Early Disease Detection AI System

> **Modern, trustworthy, production-grade MLOps pipeline for early disease prediction and risk detection — built with ZenML, MLflow, BentoML, Seldon Core, FastAPI, Streamlit, Prometheus, Evidently AI, and CI/CD best practices.**

---

## 🎯 **Project Mission**

Healthcare needs proactive tools to detect diseases *early* — saving lives and costs.  
This project is an **end-to-end AI system** that uses state-of-the-art MLOps, explainability, privacy, and human-centered design to help patients and doctors make faster, more informed decisions.

---

## 🛠️ **Tech Stack**

| Layer              | Tools & Libraries                                          |
|--------------------|------------------------------------------------------------|
| 🧩 **ML Pipelines** | ZenML, DVC, Pandas, scikit-learn, XGBoost                  |
| 🔍 **Tracking**    | MLflow (Tracking + Registry)                               |
| 📦 **Serving**     | BentoML, Seldon Core, FastAPI                              |
| 🎨 **Frontend**    | Streamlit, Tailwind (optional), Figma (design system)      |
| 🔑 **MLOps**       | Docker, Kubernetes, Helm, GitHub Actions, Prometheus, Grafana |
| 📈 **Monitoring**  | Evidently AI (data drift)                                  |
| 🔒 **Security**    | .env configs, role-based access (planned)                  |

---

## 🗂️ **Key Features**

✅ Modular pipelines (ingestion, preprocessing, training, deployment)  
✅ Data & model versioning with DVC & MLflow  
✅ Real-time API serving with FastAPI + BentoML  
✅ Explainable AI placeholders (SHAP/LIME)  
✅ Streamlit frontend for patients & doctors  
✅ Monitoring with Prometheus, Grafana & Evidently AI  
✅ CI/CD pipeline with GitHub Actions  
✅ Designed for cloud-native deployment (Kubernetes, Helm)

---

## 🏛️ **High-Level Architecture**

```plaintext
            +------------------+
            |  Patient / Doctor|
            +--------+---------+
                     |
                 (Frontend)
            +--------+---------+
            |   Streamlit App  |
            +--------+---------+
                     |
                 (API Call)
            +--------+---------+
            |   FastAPI Server |
            +--------+---------+
                     |
                 (Model Serving)
        +------------+-------------+
        | BentoML Service + Seldon |
        +------------+-------------+
                     |
               (MLflow Registry)
            +--------+---------+
            |   ZenML Pipelines |
            +------------------+
                     |
                  (Data Store)
⚙️ Local Setup
bash
Copy
Edit
# Clone the repo
git clone https://github.com/priyanshumishra610/early-disease-detection.git
cd early-disease-detection

# Create virtual environment & activate
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run ZenML pipeline (example)
python pipelines/ingestion_pipeline.py
python pipelines/preprocessing_pipeline.py
python pipelines/training_pipeline.py

# Serve API
uvicorn api.fastapi_app:app --reload

# Run Streamlit frontend
streamlit run frontend/streamlit_app.py
🚀 Run with Docker Compose
bash
Copy
Edit
docker-compose up --build
✅ Testing
bash
Copy
Edit
pytest tests/
📊 Monitoring
Access MLflow Tracking UI at http://localhost:5000

Prometheus and Grafana dashboards are included in docker-compose.yml (placeholders).

Drift reports generated with Evidently AI.

🔐 Environment Variables
All secrets and configs are managed through .env.
DO NOT commit real credentials — use .env.example as a template.

🎉 What’s Next
 Add real SHAP/LIME explainability.

 Connect frontend to backend endpoints.

 Implement real authentication (RBAC).

 Add Helm chart for Kubernetes.

 Finalize monitoring dashboards.

 Deploy to cloud (GKE, EKS, AKS).

🤝 Contributing
Fork the repo

Create a new branch (git checkout -b feature/my-feature)

Commit changes (git commit -m 'Add feature')

Push to your branch (git push origin feature/my-feature)

Create a Pull Request!

🏷️ License
MIT — use it, share it, make it better.

🔗 Built with ❤️ by Priyanshu Mishra

