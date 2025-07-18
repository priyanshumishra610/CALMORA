# Early Disease Detection ML System

![CI/CD](https://github.com/priyanshumishra610/early-disease-detection/actions/workflows/main.yml/badge.svg)

## Mission Statement
A modular, production-grade MLOps system for early disease detection, combining robust data pipelines, explainable AI, and modern DevOps best practices.

---

## Architecture Diagram
![Architecture Diagram](docs/architecture_diagram.png) <!-- Replace with your diagram -->

---

## Project Structure
See the [README in docs/](docs/README.md) for a detailed folder breakdown.

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/priyanshumishra610/early-disease-detection.git
cd early-disease-detection
```

### 2. Create and activate a virtual environment (Python 3.10 recommended)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r early_disease_detection/configs/requirements.txt
```

### 4. Set up environment variables
```bash
cp early_disease_detection/configs/.env.example early_disease_detection/.env
# Edit .env with your secrets
```

### 5. Run ZenML pipelines
```bash
zenml init
python early_disease_detection/pipelines/ingestion_pipeline.py
python early_disease_detection/pipelines/preprocessing_pipeline.py
python early_disease_detection/pipelines/training_pipeline.py
```

### 6. Start the FastAPI backend
```bash
cd early_disease_detection/api
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

### 7. Start the Streamlit frontend
```bash
cd early_disease_detection/frontend
streamlit run streamlit_app.py
```

---

## Example ZenML Pipeline Commands
```bash
zenml pipeline run ingestion_pipeline -- --file_path data/raw/sample.csv
zenml pipeline run preprocessing_pipeline -- --input_path data/versioned/ingested_data.csv
zenml pipeline run training_pipeline -- --input_path data/processed/processed_data.csv
```

---

## Example FastAPI Run Command
```bash
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

---

## Example Streamlit Run Command
```bash
streamlit run streamlit_app.py
```

---

## Contribution Guidelines
- Fork the repo and create a feature branch.
- Write clear, well-documented code.
- Add/Update tests for new features.
- Open a pull request with a clear description.
- Follow the code of conduct in [docs/compliance.md](docs/compliance.md).

---

## License
[MIT](LICENSE) 