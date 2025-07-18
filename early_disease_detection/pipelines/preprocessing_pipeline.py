from zenml import pipeline, step
import logging
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PROCESSED_DIR = os.path.abspath(os.path.join(DATA_DIR, 'processed'))

@step
def clean_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: drop duplicates.
    """
    logging.info("Cleaning data: dropping duplicates...")
    df_clean = df.drop_duplicates()
    logging.info(f"Data shape after cleaning: {df_clean.shape}")
    return df_clean

@step
def handle_nulls_step(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle null values using SimpleImputer.
    """
    logging.info(f"Handling nulls with strategy: {strategy}")
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    logging.info("Null handling complete.")
    return df_imputed

@step
def normalize_step(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize or standardize features.
    """
    logging.info(f"Normalizing data with method: {method}")
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    logging.info("Normalization/standardization complete.")
    return df_scaled

@step
def encode_categorical_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using one-hot encoding.
    """
    logging.info("Encoding categorical features...")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        logging.info("No categorical columns to encode.")
        return df
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    df = df.drop(cat_cols, axis=1)
    df_final = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    logging.info(f"Data shape after encoding: {df_final.shape}")
    return df_final

@step
def save_processed_step(df: pd.DataFrame, filename: str = 'processed_data.csv') -> str:
    """
    Save the processed data to the processed folder.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved processed data to {save_path}")
    # DVC: Run `dvc add {save_path}` to version this file
    return save_path

@pipeline
def preprocessing_pipeline(df: pd.DataFrame, null_strategy: str = 'mean', norm_method: str = 'standard', filename: str = 'processed_data.csv'):
    df_clean = clean_data_step(df)
    df_imputed = handle_nulls_step(df_clean, strategy=null_strategy)
    df_encoded = encode_categorical_step(df_imputed)
    df_norm = normalize_step(df_encoded, method=norm_method)
    save_processed_step(df_norm, filename=filename)

# Placeholder for test function
def test_preprocessing_pipeline():
    """
    Test the preprocessing pipeline (to be implemented).
    """
    pass
