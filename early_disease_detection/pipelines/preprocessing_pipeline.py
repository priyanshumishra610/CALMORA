import os
import logging
import pandas as pd
from zenml import pipeline, step
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw')))
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@step
def load_raw_step(filename: str = 'validated_raw.csv') -> pd.DataFrame:
    """
    Load validated raw data.
    """
    file_path = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(file_path)
    logging.info(f"Loaded raw data from {file_path} with shape {df.shape}")
    return df

@step
def clean_missing_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values using mean imputation.
    """
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    logging.info(f"Missing values imputed. Any nulls left: {df_imputed.isnull().sum().sum()}")
    return df_imputed

@step
def handle_outliers_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers by capping at 1st and 99th percentiles.
    """
    df_capped = df.copy()
    for col in df_capped.select_dtypes(include=['float', 'int']).columns:
        lower = df_capped[col].quantile(0.01)
        upper = df_capped[col].quantile(0.99)
        df_capped[col] = df_capped[col].clip(lower, upper)
    logging.info("Outliers capped at 1st and 99th percentiles.")
    return df_capped

@step
def feature_engineering_step(df: pd.DataFrame, target_col: str = 'target', k_best: int = 5) -> pd.DataFrame:
    """
    Feature engineering: scaling, encoding, feature selection.
    """
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    # Scale numeric features
    num_cols = X.select_dtypes(include=['float', 'int']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    # Encode categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = X.drop(list(cat_cols), axis=1)
        X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(k_best, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support(indices=True)]
    X = pd.DataFrame(X_selected, columns=selected_cols)
    # Re-attach target
    X[target_col] = y.values
    logging.info(f"Feature engineering complete. Features: {list(X.columns)}")
    return X

@step
def save_processed_step(df: pd.DataFrame, filename: str = 'processed_data.csv') -> str:
    """
    Save processed data to processed folder. DVC: Run `dvc add` on this file.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved processed data to {save_path}")
    # DVC: Run `dvc add {save_path}` to version this file
    return save_path

@pipeline
def preprocessing_pipeline(raw_filename: str = 'validated_raw.csv', processed_filename: str = 'processed_data.csv', target_col: str = 'target', k_best: int = 5):
    df = load_raw_step(filename=raw_filename)
    df_clean = clean_missing_step(df)
    df_outlier = handle_outliers_step(df_clean)
    df_features = feature_engineering_step(df_outlier, target_col=target_col, k_best=k_best)
    save_processed_step(df_features, filename=processed_filename)
