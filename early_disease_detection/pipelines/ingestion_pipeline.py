import os
import logging
import pandas as pd
from typing import Optional
from zenml import pipeline, step
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Example Pydantic schema for validation
def get_schema(df: pd.DataFrame):
    # Dynamically create a schema based on columns (customize for your use case)
    class DataSchema(BaseModel):
        # Example: expects columns 'age' (int), 'sex' (str), 'lab_result' (float)
        age: int
        sex: str
        lab_result: float
    return DataSchema

@step
def load_data_step(file_path: str, file_type: str = 'csv', db_conn: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV or database.
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'db' and db_conn:
            import sqlalchemy
            engine = sqlalchemy.create_engine(db_conn)
            df = pd.read_sql('SELECT * FROM data', engine)
        else:
            raise ValueError('Unsupported file type or missing db_conn')
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

@step
def validate_schema_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate schema using Pydantic. Logs rows, columns, dtypes.
    """
    schema = get_schema(df)
    errors = []
    for idx, row in df.iterrows():
        try:
            schema(**row.to_dict())
        except ValidationError as e:
            errors.append((idx, e.errors()))
    if errors:
        logging.error(f"Schema validation failed for {len(errors)} rows.")
        raise ValueError(f"Schema validation errors: {errors[:3]} ...")
    logging.info(f"Schema validation passed. Rows: {df.shape[0]}, Columns: {df.shape[1]}, Dtypes: {df.dtypes.to_dict()}")
    return df

@step
def save_raw_step(df: pd.DataFrame, filename: str = 'validated_raw.csv') -> str:
    """
    Save validated data to raw folder. DVC: Run `dvc add` on this file.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    save_path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved validated data to {save_path}")
    # DVC: Run `dvc add {save_path}` to version this file
    return save_path

@pipeline
def ingestion_pipeline(file_path: str, file_type: str = 'csv', db_conn: Optional[str] = None, filename: str = 'validated_raw.csv'):
    df = load_data_step(file_path=file_path, file_type=file_type, db_conn=db_conn)
    validated_df = validate_schema_step(df)
    save_raw_step(validated_df, filename=filename)

# Placeholder for test function
def test_ingestion_pipeline():
    """
    Test the ingestion pipeline (to be implemented).
    """
    pass
