# 🚑 Early Disease Detection AI System

**Modern, trustworthy, production-grade MLOps pipeline for early disease prediction and risk detection — built with ZenML, MLflow, BentoML, Seldon Core, FastAPI, Streamlit, Prometheus, Evidently AI, Tailwind, and full CI/CD best practices.**

---

## 🎯 Project Mission

Healthcare needs proactive tools to detect diseases *early* — saving lives, reducing costs, and empowering doctors with data-driven insights.  
This project is an end-to-end AI system that combines advanced **MLOps**, **explainability**, **privacy**, and **human-centered design** to deliver actionable predictions and transparent risk explanations.

---

## 🛠️ Tech Stack

| Layer              | Tools & Libraries                                          |
|--------------------|------------------------------------------------------------|
| 🧩 **ML Pipelines** | ZenML, DVC, Pandas, scikit-learn, XGBoost                  |
| 🔍 **Experiment Tracking** | MLflow (Tracking & Registry)                          |
| 📦 **Serving & Deployment** | BentoML, Seldon Core, FastAPI                     |
| 🎨 **Frontend**    | Streamlit, Tailwind CSS, Figma (design system)              |
| 🔑 **MLOps & Orchestration** | Docker, Kubernetes, Helm, GitHub Actions, Prometheus, Grafana |
| 📈 **Monitoring**  | Evidently AI (data drift)                                   |
| 🔒 **Security**    | .env configs, role-based access (RBAC planned)              |

---

## 🗂️ Key Features

✅ Modular ML pipelines (ingestion, preprocessing, training, deployment)  
✅ Versioned data & model tracking with DVC and MLflow  
✅ Real-time, scalable API serving with FastAPI + BentoML + Seldon  
✅ Explainability integration with SHAP and LIME  
✅ Streamlit + Tailwind frontend for patient & doctor dashboards  
✅ Robust monitoring with Prometheus, Grafana & Evidently AI  
✅ Fully automated CI/CD with GitHub Actions  
✅ Cloud-native ready: Docker, Kubernetes, Helm

---

## 🏛️ High-Level Architecture

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

# Create a virtual environment & activate
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run ZenML pipelines (example)
python pipelines/ingestion_pipeline.py
python pipelines/preprocessing_pipeline.py
python pipelines/training_pipeline.py

# Serve the FastAPI backend
uvicorn api.fastapi_app:app --reload

# Launch the Streamlit frontend
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
MLflow Tracking UI: http://localhost:5000

Prometheus and Grafana dashboards included in docker-compose.yml

Evidently AI for drift detection reports

🔐 Environment Variables
All secrets and configuration keys are managed securely via .env files.
Never commit real credentials — always use .env.example as your template.

🎯 Roadmap
✅ Phase 1: Core project structure, pipelines, environment configs, CI/CD setup, Docker Compose orchestration, and initial tests.
🔄 Phase 2: Implement end-to-end pipelines with real data, robust MLflow tracking, BentoML packaging, and live API serving.
🚀 Phase 3: Securely connect frontend to backend, integrate SHAP/LIME explainability, add user authentication (RBAC).
☸️ Phase 4: Deploy stack with Helm to Kubernetes (GKE, EKS, or AKS) with monitoring dashboards for live performance and drift.
🌐 Phase 5: Extend UX with refined Figma design system, Tailwind styling, advanced doctor/patient workflows, and mobile-friendly version.

💡 Future Vision
Integrate a feature store (Feast) for advanced feature management.

Expand to support multiple disease models and plugins.

Enable community-driven extensions and contributions.

🤝 Contributing
Your ideas, issues, and pull requests are welcome!

Fork this repo

Create a branch: git checkout -b feature/your-feature

Commit your changes: git commit -m 'Add new feature'

Push to your branch: git push origin feature/your-feature

Open a Pull Request!

🏷️ License
MIT License — use it, share it, make it better.

🔗 Built with ❤️ by Priyanshu Mishra
If you find this project valuable, star ⭐️ it, share it, and contribute!

