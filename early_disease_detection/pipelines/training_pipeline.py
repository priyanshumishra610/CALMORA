import os
import logging
import pandas as pd
from zenml import pipeline, step
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import shap
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed')))
MODEL_REGISTRY_DIR = os.getenv("MODEL_REGISTRY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/registry')))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@step
def load_processed_step(filename: str = 'processed_data.csv') -> pd.DataFrame:
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df = pd.read_csv(file_path)
    logging.info(f"Loaded processed data from {file_path} with shape {df.shape}")
    return df

@step
def split_data_step(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Split data: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

@step
def tune_model_step(X_train, y_train):
    """
    Hyperparameter tuning with GridSearchCV for RandomForest.
    """
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5]
    }
    clf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    logging.info(f"Best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_

@step
def evaluate_model_step(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

@step
def log_and_save_model_step(model, metrics, best_params, X_train, X_test, y_train, y_test, model_name: str = 'disease_predictor') -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        # Save model artifact locally
        os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
        model_save_path = os.path.join(MODEL_REGISTRY_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_save_path)
        logging.info(f"Saved trained model to {model_save_path}")
        # DVC: Run `dvc add {model_save_path}` to version this file
        # Register model in MLflow Model Registry
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)
        return model_save_path, run.info.run_id

@step
def shap_explainability_step(model, X_train, X_test, run_id: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    shap_path = os.path.join(MODEL_REGISTRY_DIR, f"shap_summary_{run_id}.png")
    import matplotlib.pyplot as plt
    plt.savefig(shap_path)
    logging.info(f"Saved SHAP summary plot to {shap_path}")
    # Optionally log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.log_artifact(shap_path)
    return shap_path

@pipeline
def training_pipeline(processed_filename: str = 'processed_data.csv', target_col: str = 'target', model_name: str = 'disease_predictor'):
    df = load_processed_step(filename=processed_filename)
    X_train, X_test, y_train, y_test = split_data_step(df, target_col=target_col)
    model, best_params = tune_model_step(X_train, y_train)
    metrics = evaluate_model_step(model, X_test, y_test)
    model_path, run_id = log_and_save_model_step(model, metrics, best_params, X_train, X_test, y_train, y_test, model_name=model_name)
    shap_explainability_step(model, X_train, X_test, run_id)
