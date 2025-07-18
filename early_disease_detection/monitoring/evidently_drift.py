import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import sys

# Usage: python evidently_drift.py past_data.csv new_data.csv report.html
if len(sys.argv) != 4:
    print("Usage: python evidently_drift.py <past_data.csv> <new_data.csv> <report.html>")
    sys.exit(1)

past_data_path = sys.argv[1]
new_data_path = sys.argv[2]
report_path = sys.argv[3]

# Load data
past_df = pd.read_csv(past_data_path)
new_df = pd.read_csv(new_data_path)

# Run drift detection
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=past_df, current_data=new_df)

# Save report
report.save_html(report_path)
print(f"Drift report saved to {report_path}")

# Extend: Schedule this script to run on new batches (e.g., via CI/CD or Airflow) 