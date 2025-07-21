# 🚑 CALMORA — Calm, Transparent AI Health Companion

![Version](https://img.shields.io/badge/version-v1.0-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Tests](https://img.shields.io/badge/tests-passing-green?style=flat-square)
![Status](https://img.shields.io/badge/status-production--ready-success?style=flat-square)

---

**Calmora** is an intelligent, calm, transparent AI health companion — providing personalized early disease risk detection, explainable insights, and actionable lifestyle guidance **without causing panic**.

---

## 🌿 Why CALMORA?

Millions worry about health but fear scary results or confusing AI outputs.  
**Calmora** bridges that gap — delivering calm explanations, real medical NLP, robust MLOps, and a human-centered design.

It’s **not a doctor** — it’s your **trusted guide** for safer, calmer health choices.

---

## 🎯 Core Features

✅ **NLP Symptom Checker** — Bio_ClinicalBERT extracts real medical symptoms from text.

✅ **Real Disease Risk Prediction** — Predicts risk for **heart disease**, **diabetes**, **stroke**, **Parkinson’s** — with robust ML pipelines.

✅ **Explainability (SHAP)** — Transparent risk explanations so users know *why*.

✅ **Panic-Aware Messaging** — Gentle phrasing avoids fear and confusion.

✅ **Lifestyle & Diet Tips** — Personalized healthy habits based on risk.

✅ **Data Drift Monitoring** — Evidently AI keeps models honest over time.

✅ **Consent & Legal Flow** — Clear disclaimers for safe usage.

✅ **Modern, Calm UX** — Streamlit frontend with smooth risk flow.

✅ **Full MLOps + CI/CD** — ZenML, MLflow, BentoML, Prometheus, GitHub Actions.

---

## ⚙️ Tech Stack

| Layer                    | Tools                                           |
| ------------------------ | ----------------------------------------------- |
| **ML Pipelines**         | ZenML, scikit-learn, XGBoost, Bio\_ClinicalBERT |
| **Experiment Tracking**  | MLflow                                          |
| **Serving & Deployment** | BentoML, FastAPI                                |
| **Explainability**       | SHAP                                            |
| **Monitoring**           | Evidently AI, Prometheus                        |
| **Frontend**             | Streamlit, Tailwind CSS                         |
| **CI/CD**                | GitHub Actions, Docker, Kubernetes-ready        |
| **Design**               | Figma (Calm UX)                                 |

---

## 📂 Project Structure

```
CALMORA/
├── app/
│   ├── core/
│   │   ├── nlp.py
│   │   ├── mappings.py
│   │   ├── panic_guard.py
│   │   ├── explainability.py
│   │   ├── lifestyle.py
│   │   ├── predictor.py
│   ├── services/
│   │   ├── data_monitor.py
│   ├── utils/
│   │   ├── logger.py
│   ├── models/
│   │   ├── symptoms.py
│   ├── .env.example
│   ├── Dockerfile
├── api/
│   ├── routes/
│   │   ├── symptoms.py
│   │   ├── auth.py
│   ├── fastapi_app.py
├── frontend/
│   ├── streamlit_app.py
│   ├── Dockerfile
├── tests/
│   ├── test_nlp.py
│   ├── test_mappings.py
│   ├── test_panic_guard.py
│   ├── test_explainability.py
│   ├── test_lifestyle.py
│   ├── test_end_to_end.py
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
├── .gitignore
├── docker-compose.yml
├── docs/
│   ├── calmora_flow.png
```

---

## 🚀 Quickstart

1️⃣ **Install dependencies**

```bash
pip install -r configs/requirements.txt
```

2️⃣ **Set up environment**

```bash
cp .env.example .env
# Fill in your secrets
```

3️⃣ **Run backend**

```bash
uvicorn api.fastapi_app:app --reload
```

4️⃣ **Run frontend**

```bash
streamlit run frontend/streamlit_app.py
```

5️⃣ **Run all tests**

```bash
pytest tests/
```

6️⃣ **Production deploy**

```bash
docker-compose up --build
```

---

## ✅ Status

**Version:** `v1.0`

**Stable:** Yes — production-ready

**Tests:** 100% passing

**Monitoring:** Active drift checks

---

## 📜 Disclaimer

Calmora is **not medical advice** — it’s a calm digital companion.
Always consult a licensed doctor for real health concerns.

---

## 🤝 Contributing

Open to ideas, pull requests, and feedback — let’s build calm, transparent health AI for everyone. 🌿✨

---

**Made with care by [Priyanshu Mishra](https://github.com/priyanshumishra610)** 🚑

