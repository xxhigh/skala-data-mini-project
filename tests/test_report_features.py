"""피처 정의와 전처리 기준을 검증한다.

실행: python -m unittest discover -s tests
"""

import unittest

import numpy as np
import pandas as pd

from src.features import FEATURES, make_delta_q_features, make_features
from src.preprocess import clean_cells, fill_missing, scale_features


def make_cell(cycle_life=400.0):
    return {
        "cell_id": 7, "cycle_life": cycle_life,
        "chargetime": np.r_[0.0, np.full(98, 10.0), 3000.0],
        "Tavg": np.full(100, 30.0),
        "q10": np.zeros(1000), "q100": -np.linspace(2.0, 3.5, 1000),
    }


class ReportFeaturesTests(unittest.TestCase):
    def test_delta_q_features(self):
        log_var, minimum = make_delta_q_features(np.zeros(1000), -np.linspace(2.0, 3.5, 1000))
        self.assertAlmostEqual(minimum, -3.5)
        self.assertAlmostEqual(log_var, np.log10(np.var(np.linspace(2.0, 3.5, 1000))))

    def test_zero_variance_marks_log_missing(self):
        log_var, minimum = make_delta_q_features(np.zeros(1000), np.zeros(1000))
        self.assertTrue(np.isnan(log_var))
        self.assertEqual(minimum, 0)

    def test_clean_cells_drops_missing_life_and_abnormal_values(self):
        cells = clean_cells([make_cell(), make_cell(np.nan)])
        self.assertEqual(len(cells), 1)
        frame = make_features(cells, "Batch1")
        self.assertEqual(list(frame.columns), ["batch_id", "cell_id", *FEATURES, "cycle_life"])
        self.assertEqual(frame.loc[0, "mean_chargetime"], 10.0)

    def test_preprocessing_uses_training_statistics(self):
        train = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [10.0, np.nan, 30.0]})
        test = pd.DataFrame({"a": [10000.0], "b": [np.nan]})
        train, test = fill_missing(train, test)
        self.assertEqual(test.loc[0, "b"], 20.0)
        train_scaled, test_scaled = scale_features(train, test)
        np.testing.assert_allclose(train_scaled.mean(axis=0), [0, 0], atol=1e-12)
        self.assertEqual(test_scaled[0, 1], 0)


if __name__ == "__main__":
    unittest.main()
