"""Check the CPU solver against sklearn and guard validation-selection tradeoffs."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "ens"))
from weighted_fusion import choose_candidate, fit_logistic, folds_by_video, sample_weights


class FusionFitTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = (X[:, 0] - X[:, 2] + rng.normal(size=100) > 0).astype(int)
        return X, y, rng.uniform(0.2, 3, size=100)

    def test_weighted_objective_stationary(self):
        X, y, w = self.fixture()
        coef, intercept = fit_logistic(X, y, w, C=0.3)
        p = 1 / (1 + np.exp(-(X @ coef + intercept)))
        np.testing.assert_allclose(X.T @ (w * (p - y)) + coef / 0.3, 0, atol=1e-6)
        self.assertAlmostEqual(float(np.dot(w, p - y)), 0, places=6)

    def test_matches_sklearn(self):
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            self.skipTest("sklearn unavailable locally; required in CI")
        X, y, w = self.fixture()
        for C in (0.3, 1, 3):
            coef, intercept = fit_logistic(X, y, w, C)
            ref = LogisticRegression(C=C, solver="lbfgs", tol=1e-10, max_iter=1000).fit(X, y, sample_weight=w)
            np.testing.assert_allclose(X @ coef + intercept, ref.decision_function(X), atol=1e-6)

    def test_invalid_training_data_rejected(self):
        X, y, w = self.fixture()
        with self.assertRaises(ValueError):
            fit_logistic(X, np.zeros_like(y), w)
        with self.assertRaises(ValueError):
            fit_logistic(X, y, -w)

    def test_degradations_stay_in_same_fold(self):
        vids = np.repeat([f"video-{i}" for i in range(20)], 3)
        folds = folds_by_video(vids)
        self.assertEqual(len(set(folds)), 5)
        for vid in set(vids):
            self.assertEqual(len(set(folds[vids == vid])), 1)

    def test_reweighting_only_changes_ffpp_reals(self):
        y = np.array([0, 1, 0, 1] * 3)
        src = np.array(["ffpp", "ffpp", "mavos", "mavos"] * 3)
        cond = np.repeat(["clean", "blur", "comp"], 4)
        before = sample_weights(y, src, cond)
        after = sample_weights(y, src, cond, real_weight=3)
        mask = (src == "ffpp") & (y == 0)
        np.testing.assert_allclose(after[mask], 3 * before[mask])
        np.testing.assert_allclose(after[~mask], before[~mask])

    def test_rejects_false_alarm_gain_that_misses_more_fakes(self):
        baseline = dict(C=1, real_weight=1, bacc=0.80, caught=0.8, ffpp_clean_false_alarm=0.1)
        worse = dict(C=1, real_weight=3, bacc=0.81, caught=0.75, ffpp_clean_false_alarm=0.02)
        self.assertIsNone(choose_candidate([baseline, worse]))
        good = dict(C=1, real_weight=2, bacc=0.81, caught=0.80, ffpp_clean_false_alarm=0.05)
        self.assertEqual(choose_candidate([baseline, worse, good]), good)


if __name__ == "__main__":
    unittest.main()
