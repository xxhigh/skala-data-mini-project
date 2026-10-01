"""보고서 피처 정의와 전처리 경계를 검증한다.

실행: python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from src.features import FEATURES, VOLTAGE_GRID, extract_delta_q, extract_record
from src.features import load_features
from src.preprocess import make_preprocessor


class ReportFeaturesTests(unittest.TestCase):
    def make_summary(self):
        values = np.r_[np.arange(1.0, 101.0), 1e12]
        return {"cycle": np.arange(1, 102), "chargetime": values, "Tavg": values}

    def test_means_only_use_first_100_cycles(self):
        record, reason = extract_record(self.make_summary(), 400, 7)
        self.assertIsNone(reason)
        self.assertEqual(record["mean_chargetime"], 50.5)
        self.assertEqual(record["mean_Tavg"], 50.5)
        self.assertEqual(record["cycle_life"], 400)

    def test_incomplete_observation_and_invalid_target_are_excluded(self):
        summary = self.make_summary()
        summary["cycle"][99] = 102
        self.assertEqual(extract_record(summary, 400, 7)[1], "incomplete_early_cycles")
        self.assertEqual(
            extract_record(self.make_summary(), np.nan, 7)[1],
            "invalid_or_insufficient_cycle_life",
        )

    def test_different_descending_grid_is_interpolated(self):
        voltage = np.linspace(3.5, 2.0, 80)
        features, issue = extract_delta_q(voltage, np.zeros(80), -voltage)
        self.assertIsNone(issue)
        self.assertAlmostEqual(features["delta_q_min"], -3.5)
        self.assertAlmostEqual(features["log_delta_q_var"], np.log10(np.var(VOLTAGE_GRID)))

    def test_nonfinite_pairs_are_removed_together(self):
        voltage = np.linspace(2.0, 3.5, 80)
        q10 = np.zeros(80)
        q100 = -voltage.copy()
        q10[10] = np.inf
        q100[15] = np.nan
        features, issue = extract_delta_q(voltage, q10, q100)
        self.assertIsNone(issue)
        self.assertAlmostEqual(features["log_delta_q_var"], np.log10(np.var(VOLTAGE_GRID)))

    def test_zero_variance_keeps_minimum_and_marks_log_missing(self):
        features, issue = extract_delta_q(VOLTAGE_GRID, np.zeros(1000), np.zeros(1000))
        self.assertEqual(issue, "zero_delta_q_variance")
        self.assertEqual(features["delta_q_min"], 0)
        self.assertTrue(np.isnan(features["log_delta_q_var"]))

    def test_invalid_curves_do_not_extrapolate(self):
        for voltage, q10, q100, reason in (
            (VOLTAGE_GRID, np.zeros(5), np.zeros(1000), "voltage_capacity_length_mismatch"),
            ([2.1, 3.5], [0, 0], [1, 1], "incomplete_voltage_range"),
            ([2, 2, 3.5], [0, 0, 0], [1, 1, 1], "duplicate_voltage_points"),
            (VOLTAGE_GRID, None, None, "missing_q10_or_q100"),
        ):
            with self.subTest(reason=reason):
                features, issue = extract_delta_q(voltage, q10, q100)
                self.assertEqual(issue, reason)
                self.assertTrue(all(np.isnan(value) for value in features.values()))

    def test_mat_v5_loader_and_table_schema(self):
        curves = [{"Qdlin": np.zeros(1000)} for _ in range(101)]
        curves[99] = {"Qdlin": -VOLTAGE_GRID}
        # 101사이클 곡선의 극단값은 피처에 포함되지 않는다.
        curves[100] = {"Qdlin": np.full(1000, 1e12)}
        cell = {
            "summary": self.make_summary(), "cycle_life": 400,
            "Vdlin": VOLTAGE_GRID, "cycles": curves,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.mat"
            savemat(path, {"batch": [cell]})
            frame, skipped, issues = load_features(path, batch_id="Batch1")
        self.assertFalse(skipped)
        self.assertFalse(issues)
        self.assertEqual(list(frame.columns), ["batch_id", "cell_id", *FEATURES, "cycle_life"])
        self.assertEqual(frame.iloc[0]["batch_id"], "Batch1")
        self.assertAlmostEqual(frame.iloc[0]["delta_q_min"], -3.5)

    def test_preprocessor_reuses_training_statistics(self):
        train = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
        preprocess = make_preprocessor().fit(train)
        np.testing.assert_allclose(preprocess["imputer"].statistics_, [3, 20])
        np.testing.assert_allclose(preprocess["scaler"].mean_, [3, 20])
        preprocess.transform([[10000, np.nan]])
        np.testing.assert_allclose(preprocess["scaler"].mean_, [3, 20])


if __name__ == "__main__":
    unittest.main()
