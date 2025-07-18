import pandas as pd
from early_disease_detection.pipelines.preprocessing_pipeline import clean_data_step, handle_nulls_step

def test_preprocessing_shape():
    df = pd.DataFrame({'a': [1, 2, 2, None], 'b': [3, 4, 4, 5]})
    df_clean = clean_data_step(df)
    df_imputed = handle_nulls_step(df_clean, strategy='mean')
    assert df_imputed.isnull().sum().sum() == 0
    assert df_imputed.shape[0] <= df.shape[0] 