from zenml import pipeline, step
import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PROCESSED_DIR = os.path.abspath(os.path.join(DATA_DIR, 'processed'))
MODEL_REGISTRY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/registry'))

@step
def load_processed_data_step(filename: str = 'processed_data.csv') -> pd.DataFrame:
    """
    Load processed data for training.
    """
    file_path = os.path.join(PROCESSED_DIR, filename)
    logging.info(f"Loading processed data from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Loaded data shape: {df.shape}")
    return df

@step
def split_data_step(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets.
    """
    logging.info(f"Splitting data with test_size={test_size}")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

@step
def train_model_step(X_train, y_train, X_test, y_test, mlflow_tracking_uri: str, model_name: str = 'disease_predictor') -> str:
    """
    Train a RandomForestClassifier, log to MLflow, and save model artifact.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_name=model_name) as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_param('n_estimators', 100)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        logging.info(f"Model accuracy: {acc}")
        # Register model in MLflow Model Registry
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=model_name
        )
        # Save model artifact to /models/registry/
        os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
        model_save_path = os.path.join(MODEL_REGISTRY_DIR, f"{model_name}.pkl")
        import joblib
        joblib.dump(clf, model_save_path)
        logging.info(f"Saved trained model to {model_save_path}")
        # DVC: Run `dvc add {model_save_path}` to version this file
        return model_save_path

@step
def explainability_placeholder_step(model_path: str, X_test):
    """
    Placeholder for SHAP explainability (to be implemented).
    """
    logging.info("Explainability step (SHAP) placeholder.")
    # TODO: Add SHAP explainability here
    pass

@pipeline
def training_pipeline(
    processed_filename: str = 'processed_data.csv',
    target_col: str = 'target',
    mlflow_tracking_uri: str = 'http://localhost:5000',
    model_name: str = 'disease_predictor'):
    df = load_processed_data_step(filename=processed_filename)
    X_train, X_test, y_train, y_test = split_data_step(df, target_col=target_col)
    model_path = train_model_step(X_train, y_train, X_test, y_test, mlflow_tracking_uri, model_name=model_name)
    explainability_placeholder_step(model_path, X_test)

# Placeholder for test function
def test_training_pipeline():
    """
    Test the training pipeline (to be implemented).
    """
    pass
