import pandas as pd
from early_disease_detection.pipelines.ingestion_pipeline import load_data_step

def test_load_data_csv(tmp_path):
    # Create a sample CSV
    sample = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    file_path = tmp_path / "sample.csv"
    sample.to_csv(file_path, index=False)
    df = load_data_step(str(file_path), file_type='csv')
    assert df.shape == (2, 2) 