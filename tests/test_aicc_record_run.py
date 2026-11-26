import importlib.util
import json
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/aicc_record_run.py"
spec = importlib.util.spec_from_file_location("aicc_record_run", MODULE_PATH)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        source = MODULE_PATH.parents[1] / "experiments/aicc/cosmos-cosmic-run/smoke_report.json"
        self.report = json.loads(source.read_text())

    def test_executed_report_is_accepted(self):
        recorder.validate_report(self.report)

    def test_missing_submission_checks_is_rejected(self):
        self.report["submission_checks"] = []
        with self.assertRaises(ValueError):
            recorder.validate_report(self.report)

    def test_nonfinite_measurement_is_rejected(self):
        self.report["experiments"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            recorder.validate_report(self.report)

    def test_biome_leakage_is_rejected(self):
        row = next(r for r in self.report["experiments"] if r["split"] == "unseen_biome_holdout")
        row["overlapping_biomes"] = 1
        with self.assertRaises(ValueError):
            recorder.validate_report(self.report)

    def test_unmeasured_selected_model_is_rejected(self):
        self.report["selected_variant"] = "never_ran"
        with self.assertRaises(ValueError):
            recorder.validate_report(self.report)

    def test_failed_run_is_rejected(self):
        self.report["status"] = "failed"
        with self.assertRaises(ValueError):
            recorder.validate_report(self.report)

    def test_notebook_url_with_session_parameter_is_rejected(self):
        with self.assertRaises(ValueError):
            recorder.validate_notebook_url("https://www.kaggle.com/code/user/run?session=private")

    def test_offsite_notebook_url_is_rejected(self):
        with self.assertRaises(ValueError):
            recorder.validate_notebook_url("https://example.com/code/user/run")


if __name__ == "__main__":
    unittest.main()

