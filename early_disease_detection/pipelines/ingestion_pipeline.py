from zenml import pipeline, step
import logging
import pandas as pd
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DIR = os.path.abspath(os.path.join(DATA_DIR, 'raw'))
VERSIONED_DIR = os.path.abspath(os.path.join(DATA_DIR, 'versioned'))

@step
def load_data_step(file_path: str, file_type: str = 'csv', db_conn: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV, Excel, JSON, or a database connection.
    """
    logging.info(f"Loading data from {file_path} as {file_type}")
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'excel':
        df = pd.read_excel(file_path)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'db' and db_conn:
        import sqlalchemy
        engine = sqlalchemy.create_engine(db_conn)
        df = pd.read_sql('SELECT * FROM data', engine)
    else:
        raise ValueError('Unsupported file type or missing db_conn')
    logging.info(f"Loaded data shape: {df.shape}")
    return df

@step
def validate_schema_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate schema (placeholder: checks for null columns).
    """
    logging.info("Validating schema...")
    if df.isnull().all().any():
        raise ValueError("Some columns are completely null!")
    logging.info("Schema validation passed.")
    return df

@step
def save_versioned_step(df: pd.DataFrame, filename: str = 'ingested_data.csv') -> str:
    """
    Save the ingested data to the versioned folder.
    """
    os.makedirs(VERSIONED_DIR, exist_ok=True)
    save_path = os.path.join(VERSIONED_DIR, filename)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved versioned data to {save_path}")
    # DVC: Run `dvc add {save_path}` to version this file
    return save_path

@pipeline
def ingestion_pipeline(file_path: str, file_type: str = 'csv', db_conn: Optional[str] = None, filename: str = 'ingested_data.csv'):
    df = load_data_step(file_path=file_path, file_type=file_type, db_conn=db_conn)
    validated_df = validate_schema_step(df)
    save_versioned_step(validated_df, filename=filename)

# Placeholder for test function
def test_ingestion_pipeline():
    """
    Test the ingestion pipeline (to be implemented).
    """
    pass
