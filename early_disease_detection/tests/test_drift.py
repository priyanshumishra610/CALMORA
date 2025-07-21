"""
Test DataMonitor drift detection and logging.
"""
import os
import tempfile
from app.services.data_monitor import DataMonitor

def test_drift_detection_and_logging():
    # Use a temp log file
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "drift.log")
        # Simulate reference data (e.g., all zeros)
        ref_path = os.path.join(tmpdir, "ref.csv")
        with open(ref_path, "w") as f:
            f.write("a,b,c\n0,0,0\n0,0,0\n")
        monitor = DataMonitor(reference_data_path=ref_path, drift_log_path=log_path, drift_threshold=0.1)
        # Simulate current data that is very different (e.g., all ones)
        input_data = [{"a": 1, "b": 1, "c": 1} for _ in range(2)]
        result = monitor.check_drift(input_data)
        assert "drift" in result
        assert result["drift"] is True or result["drift_score"] > 0.1
        # Check log file was written
        with open(log_path) as f:
            log_content = f.read()
        assert "Drift Detected" in log_content or "Drift Score" in log_content 