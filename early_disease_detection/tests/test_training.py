import pandas as pd
import os
from early_disease_detection.pipelines.training_pipeline import train_model_step

def test_model_artifact(tmp_path):
    # Create dummy data
    X_train = [[0, 1], [1, 0]]
    y_train = [0, 1]
    X_test = [[1, 1]]
    y_test = [1]
    model_path = train_model_step(X_train, y_train, X_test, y_test, mlflow_tracking_uri='sqlite:///mlruns.db', model_name='test_model')
    assert os.path.exists(model_path) 